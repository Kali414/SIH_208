"""
rake_dss.py

End-to-end reference: priority scoring (TF) + stockyard selection + route/mode scoring
+ allocation optimization (OR-Tools CP-SAT) + simple dispatch scheduling.

This is a reference prototype — adapt constraints/objectives for your business.
"""

from __future__ import annotations
import math, json, datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd

# TensorFlow priority model
import tensorflow as tf
from tensorflow.keras import layers, models

# OR-Tools CP-SAT
from ortools.sat.python import cp_model

# simple graph routing using networkx for multi-hop (optional)
import networkx as nx

# -------------------- Data classes --------------------
@dataclass
class Order:
    order_id: str
    material: str
    quantity_tons: int
    destination: str
    sla_hours: int
    priority: str  # High/Medium/Low
    preferred_mode: str = "Any"
    penalty_per_hour: float = 0.0
    max_splits: int = 2

@dataclass
class Stockyard:
    yard_id: str
    location: str
    inventory: Dict[str,int]            # material -> tons
    loading_capacity_tph: Dict[str,int] # material -> tons/hour
    siding_capacity_rakes: int
    operational_cost_per_ton: float = 0.0

@dataclass
class Rake:
    rake_id: str
    capacity_tons: int
    allowed_materials: List[str]
    available_from: datetime.datetime

@dataclass
class RouteOption:
    src: str
    dest: str
    mode: str  # Rail / Road
    distance_km: float
    cost_per_ton: float
    travel_time_hr: float
    reliability: float = 0.95

# -------------------- Priority model (TF MLP) --------------------
class PriorityModel:
    def __init__(self, input_dim:int = 4):
        self.input_dim = input_dim
        self.model = self._build()
    def _build(self):
        inp = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(32, activation='relu')(inp)
        x = layers.Dense(16, activation='relu')(x)
        out = layers.Dense(1, activation='sigmoid')(x)
        m = models.Model(inp, out)
        m.compile(optimizer='adam', loss='mse')
        return m
    def featurize(self, o:Order) -> np.ndarray:
        # q scaled, inverse sla, priority encode, penalty scaled
        q = o.quantity_tons / 5000.0
        sla = 1.0 / max(o.sla_hours,1)
        pr = 1.0 if o.priority.lower()=="high" else (0.6 if o.priority.lower()=="medium" else 0.2)
        pen = o.penalty_per_hour / 10000.0
        return np.array([q, sla, pr, pen], dtype=float)
    def train_dummy(self, orders:List[Order], epochs:int=40):
        X = np.vstack([self.featurize(o) for o in orders])
        # synthetic label: more quantity & tighter SLA => higher
        y = np.clip(0.4*X[:,0] + 0.4*X[:,1] + 0.2*X[:,2], 0, 1)
        self.model.fit(X,y,epochs=epochs,verbose=0)
    def predict(self, orders:List[Order]) -> Dict[str,float]:
        X = np.vstack([self.featurize(o) for o in orders])
        preds = self.model.predict(X,verbose=0).flatten()
        return {orders[i].order_id: float(preds[i]) for i in range(len(orders))}

# -------------------- Route & Mode selection --------------------
def select_best_route(route_options: List[RouteOption], sla_hours:int) -> RouteOption:
    # score = cost + time_penalty - reliability_bonus
    best = None
    best_score = 1e18
    for r in route_options:
        time_pen = 0.0 if r.travel_time_hr <= sla_hours else (r.travel_time_hr - sla_hours) * 100.0
        score = r.cost_per_ton + time_pen - (r.reliability * 50.0)
        if score < best_score:
            best_score = score
            best = r
    return best

# -------------------- Stockyard selection (per order) --------------------
def candidate_stockyards_for_order(order:Order, stockyards:List[Stockyard]) -> List[Stockyard]:
    cands = []
    for s in stockyards:
        if s.inventory.get(order.material,0) > 0:
            cands.append(s)
    return cands

def best_stockyard_for_order(order:Order, stockyards:List[Stockyard], routes:List[RouteOption]) -> Tuple[Optional[Stockyard], Optional[RouteOption]]:
    # choose stockyard that minimizes estimated (transport cost_per_ton + operational_cost)
    best = None
    best_route = None
    best_score = 1e18
    for s in candidate_stockyards_for_order(order, stockyards):
        # find route options s.location -> order.destination
        ropts = [r for r in routes if r.src==s.location and r.dest==order.destination]
        if not ropts:
            continue
        # pick best route among modes
        rbest = select_best_route(ropts, order.sla_hours)
        score = rbest.cost_per_ton + s.operational_cost_per_ton/100.0  # small weight
        if score < best_score:
            best_score = score
            best = s
            best_route = rbest
    return best, best_route

# -------------------- Allocation optimization (MILP/CPSAT) --------------------
def optimize_allocation(orders:List[Order], stockyards:List[Stockyard], rakes:List[Rake], routes:List[RouteOption], priority_scores:Dict[str,float], constraints:Dict[str,Any]={}) -> Dict[str,Any]:
    """
    Variables:
      x[o,s] = integer tons from stockyard s to order o  (0..inventory)
    Constraints:
      - demand satisfaction: sum_s x[o,s] == qty_o
      - supply limit: sum_o x[o,s] <= inventory[s][material]
      - max_splits: number of used sources per order <= max_splits
      - rake capacity aggregated: total assigned <= sum(capacity*used_rake)
    Objective:
      minimize sum (x[o,s] * transport_cost_estimate) - sum(priority_bonus * qty)
    """
    # maps
    orders_map = {o.order_id:o for o in orders}
    stock_map = {s.yard_id:s for s in stockyards}
    # precompute transport_cost per (s,o) using cheapest route option
    trans_cost = {}
    route_choice = {}
    for s in stockyards:
        for o in orders:
            # candidate routes:
            cand = [r for r in routes if r.src==s.location and r.dest==o.destination]
            if not cand:
                continue
            # pick cost per ton adjusted by reliability
            best_r = select_best_route(cand, o.sla_hours)
            trans_cost[(o.order_id, s.yard_id)] = int(best_r.cost_per_ton)
            route_choice[(o.order_id, s.yard_id)] = best_r
    # Build CP-SAT model
    model = cp_model.CpModel()
    # integer variables x[o,s]
    x = {}
    for o in orders:
        for s in stockyards:
            if (o.order_id, s.yard_id) in trans_cost:
                ub = min(s.inventory.get(o.material,0), o.quantity_tons)
                var = model.NewIntVar(0, ub, f"x_{o.order_id}_{s.yard_id}")
                x[(o.order_id,s.yard_id)] = var
    # demand satisfaction
    for o in orders:
        vars_o = [v for (oid,yid),v in x.items() if oid==o.order_id]
        if not vars_o:
            return {"status":"infeasible","msg":f"No source for order {o.order_id}"}
        model.Add(sum(vars_o) == o.quantity_tons)
        # max splits via indicator booleans
        inds = []
        for s in stockyards:
            if (o.order_id,s.yard_id) in x:
                b = model.NewBoolVar(f"ind_{o.order_id}_{s.yard_id}")
                # link: if b==0 then x==0, if b==1 then x>=1
                model.Add(x[(o.order_id,s.yard_id)] >= 1).OnlyEnforceIf(b)
                model.Add(x[(o.order_id,s.yard_id)] == 0).OnlyEnforceIf(b.Not())
                inds.append(b)
        model.Add(sum(inds) <= o.max_splits)
    # supply constraint
    for s in stockyards:
        for mat in s.inventory.keys():
            vars_s = [x[(o.order_id,s.yard_id)] for o in orders if (o.order_id,s.yard_id) in x and o.material==mat]
            if vars_s:
                model.Add(sum(vars_s) <= s.inventory[mat])
    # rake capacity aggregate: sum assigned <= sum(capacity * used)
    total_assigned = []
    for (oid,sid),var in x.items():
        total_assigned.append(var)
    total_assigned_sum = model.NewIntVar(0, sum([r.capacity_tons for r in rakes]), "total_assigned_sum")
    model.Add(total_assigned_sum == sum(total_assigned))
    # y_r usage booleans and cap_vars
    cap_vars = []
    for r in rakes:
        y = model.NewBoolVar(f"use_{r.rake_id}")
        cap = model.NewIntVar(0, r.capacity_tons, f"cap_{r.rake_id}")
        # cap == capacity * y
        model.Add(cap == r.capacity_tons).OnlyEnforceIf(y)
        model.Add(cap == 0).OnlyEnforceIf(y.Not())
        cap_vars.append(cap)
    model.Add(sum(cap_vars) >= total_assigned_sum)
    # Objective: minimize transport cost - priority bonus
    terms = []
    for (oid,sid),var in x.items():
        c = trans_cost[(oid,sid)]
        terms.append(c * var)
    # priority bonus: encourage serving high priority by subtracting weight*qty
    for o in orders:
        score = int(priority_scores.get(o.order_id,0) * 1000)
        vars_o = [x[(o.order_id,s.yard_id)] for s in stockyards if (o.order_id,s.yard_id) in x]
        if vars_o:
            terms.append(- score * sum(vars_o))
    model.Minimize(sum(terms))
    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15
    solver.parameters.num_search_workers = 8
    result = solver.Solve(model)
    if result in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        assignment = []
        for (oid,sid),var in x.items():
            v = solver.Value(var)
            if v>0:
                assignment.append({"order":oid,"stockyard":sid,"tons":v,"route": route_choice[(oid,sid)].__dict__})
        used_rakes = []
        for r in rakes:
            y_var_name = f"use_{r.rake_id}"
            # find var by name (quick hack)
            try:
                val = solver.Value(model.GetVarFromProtoName(y_var_name))
                if val == 1:
                    used_rakes.append(r.rake_id)
            except Exception:
                pass
        return {"status":"ok","assignments":assignment,"used_rakes":used_rakes}
    else:
        return {"status":"infeasible","msg":"No feasible solution"}

# -------------------- Dispatch scheduling (simple) --------------------
def compute_loading_time(tons:int, loading_rate_tph:int) -> float:
    if loading_rate_tph<=0: return 9999.0
    return tons / loading_rate_tph

def simple_dispatch_schedule(assignments:List[Dict[str,Any]], stockyards:List[Stockyard], rakes:List[Rake]) -> List[Dict[str,Any]]:
    """
    assignments: list of {order, stockyard, tons, route}
    returns simple schedule: group assignments by stockyard and produce departure times sequentially.
    """
    schedules = []
    # group by stockyard
    by_sy = {}
    for a in assignments:
        by_sy.setdefault(a['stockyard'],[]).append(a)
    now = datetime.datetime.now().replace(minute=0,second=0,microsecond=0)
    for syid, items in by_sy.items():
        sy = next((s for s in stockyards if s.yard_id==syid), None)
        # naive: schedule items sequentially using loading rate of material (average)
        t_cursor = now
        for it in items:
            # find material from order -> need order info? assume same material for chunk
            tons = it['tons']
            # choose loading_rate for material if known else average
            # We guess material from route? route doesn't carry material. So require order mapping.
            # For prototype, use first loading_rate value
            loading_rate = list(sy.loading_capacity_tph.values())[0] if sy and sy.loading_capacity_tph else 200
            load_hours = compute_loading_time(tons, loading_rate)
            depart = t_cursor + datetime.timedelta(hours=load_hours)
            # travel time:
            travel_hrs = it['route']['travel_time_hr']
            arrive = depart + datetime.timedelta(hours=travel_hrs)
            schedules.append({
                "order": it['order'],
                "stockyard": syid,
                "tons": tons,
                "depart_time": depart.isoformat(),
                "arrival_time": arrive.isoformat(),
                "route_mode": it['route']['mode']
            })
            # next start after depart (rake frees after arrive+unload in real world; naive next here)
            t_cursor = depart + datetime.timedelta(hours=1)  # gap between rakes
    return schedules

# -------------------- Sample dataset and run pipeline --------------------
def sample_data():
    orders = [
        Order("O101","Aluminium",2000,"Kolkata",72,"High","Any",200.0,2),
        Order("O102","TMT",1500,"Ranchi",24,"High","Rail",500.0,1),
        Order("O103","Aluminium",800,"Patna",48,"Medium","Any",100.0,2),
    ]
    stockyards = [
        Stockyard("SY1","Bokaro Main",{"Aluminium":1200,"TMT":2000},{"Aluminium":150,"TMT":300},2,60.0),
        Stockyard("SY2","CMO-AL-1",{"Aluminium":1500,"TMT":0},{"Aluminium":120},1,65.0),
    ]
    now = datetime.datetime.now()
    rakes = [
        Rake("R1",3500,["Aluminium","TMT"], now),
        Rake("R2",2800,["TMT"], now),
    ]
    routes = [
        RouteOption("Bokaro Main","Kolkata","Rail",420,1300,14,0.93),
        RouteOption("CMO-AL-1","Kolkata","Rail",400,1250,13,0.94),
        RouteOption("Bokaro Main","Patna","Road",310,1900,9,0.9),
        RouteOption("Bokaro Main","Ranchi","Rail",180,1200,10,0.96),
    ]
    return orders, stockyards, rakes, routes

def run_pipeline():
    print("Loading sample data...")
    orders, stockyards, rakes, routes = sample_data()
    # 1. Priority scores
    pm = PriorityModel()
    pm.train_dummy(orders)
    pri = pm.predict(orders)
    print("Priority scores:", json.dumps(pri, indent=2))
    # 2. stockyard suggestions per order
    sy_suggestions = {}
    route_suggestions = {}
    for o in orders:
        sy, r = best_stockyard_for_order(o, stockyards, routes)
        sy_suggestions[o.order_id] = sy.yard_id if sy else None
        route_suggestions[o.order_id] = r.__dict__ if r else None
    print("Stockyard suggestions:", json.dumps(sy_suggestions, indent=2))
    print("Route suggestions (best):", json.dumps(route_suggestions, indent=2))
    # 3. Run allocation optimization
    result = optimize_allocation(orders, stockyards, rakes, routes, pri)
    print("Optimization result:", json.dumps(result, indent=2))
    if result.get("status")=="ok":
        assignments = result["assignments"]
        # 4. dispatch schedule
        schedule = simple_dispatch_schedule(assignments, stockyards, rakes)
        print("Dispatch schedule (simple):", json.dumps(schedule, indent=2))
    else:
        print("No feasible allocation:", result)
if __name__ == "__main__":
    run_pipeline()
