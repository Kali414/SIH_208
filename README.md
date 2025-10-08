## URL
https://sih-208.onrender.com/optimize

POST Method

## Input
```
{
  "orders": [
    {"order_id": "ORD101", "material": "Aluminium", "quantity": 1200, "priority": "High", "destination": "Kolkata", "sla_hours": 72},
    {"order_id": "ORD102", "material": "Steel", "quantity": 3000, "priority": "Medium", "destination": "Bhubaneswar", "sla_hours": 120},
    {"order_id": "ORD103", "material": "Copper", "quantity": 800, "priority": "Low", "destination": "Ranchi", "sla_hours": 96},
    {"order_id": "ORD104", "material": "Aluminium", "quantity": 1500, "priority": "High", "destination": "Kolkata", "sla_hours": 72}
  ],
  "stockyards": [
    {"yard_id": "SY1", "location": "Bokaro", "inventory": {"Aluminium": 1500, "Steel": 2000, "Copper": 500}, "loading_capacity": 200, "siding_capacity_rakes": 3, "unloading_ease": 0.9},
    {"yard_id": "SY2", "location": "Durgapur", "inventory": {"Aluminium": 500, "Steel": 1500, "Copper": 800}, "loading_capacity": 150, "siding_capacity_rakes": 2, "unloading_ease": 0.8},
    {"yard_id": "SY3", "location": "Rourkela", "inventory": {"Aluminium": 700, "Steel": 1000, "Copper": 1200}, "loading_capacity": 180, "siding_capacity_rakes": 2, "unloading_ease": 0.85}
  ],
  "rakes": [
    {"rake_id": "R1", "capacity": 2000, "allowed_materials": ["Aluminium", "Steel"], "available_from": "2025-10-08T06:00:00"},
    {"rake_id": "R2", "capacity": 1500, "allowed_materials": ["Aluminium", "Copper"], "available_from": "2025-10-08T08:00:00"},
    {"rake_id": "R3", "capacity": 2500, "allowed_materials": ["Steel", "Copper"], "available_from": "2025-10-08T10:00:00"}
  ],
  "routes": [
    {"src": "Bokaro", "dest": "Kolkata", "mode": "Rail", "distance_km": 420, "cost_per_ton": 1300, "travel_time_hr": 14, "reliability": 0.93},
    {"src": "Durgapur", "dest": "Bhubaneswar", "mode": "Rail", "distance_km": 550, "cost_per_ton": 1500, "travel_time_hr": 18, "reliability": 0.9},
    {"src": "Rourkela", "dest": "Ranchi", "mode": "Rail", "distance_km": 250, "cost_per_ton": 1100, "travel_time_hr": 7, "reliability": 0.95},
    {"src": "Bokaro", "dest": "Ranchi", "mode": "Road", "distance_km": 180, "cost_per_ton": 900, "travel_time_hr": 5, "reliability": 0.88}
  ],
  "constraints": {
    "max_loading_per_hour_per_yard": {"SY1": 200, "SY2": 150, "SY3": 180},
    "max_siding_rakes": {"SY1": 3, "SY2": 2, "SY3": 2},
    "priority_weights": {"High": 3, "Medium": 2, "Low": 1}
  }
}

```


## Output
```
{
    "optimization_result": {
        "rakes": [
            {
                "assigned_orders": [
                    "ORD101",
                    "ORD104"
                ],
                "destinations": [
                    "Kolkata"
                ],
                "mode": "Rail",
                "rake_id": "R1",
                "selected_route": [
                    "Bokaro to Kolkata"
                ],
                "source_stockyard": "SY1",
                "total_load": 2700
            },
            {
                "assigned_orders": [
                    "ORD103"
                ],
                "destinations": [
                    "Ranchi"
                ],
                "mode": "Rail",
                "rake_id": "R2",
                "selected_route": [
                    "Durgapur to Ranchi"
                ],
                "source_stockyard": "SY2",
                "total_load": 800
            },
            {
                "assigned_orders": [
                    "ORD102"
                ],
                "destinations": [
                    "Bhubaneswar"
                ],
                "mode": "Rail",
                "rake_id": "R3",
                "selected_route": [
                    "Bokaro to Bhubaneswar"
                ],
                "source_stockyard": "SY1",
                "total_load": 3000
            }
        ],
        "remaining_orders": []
    },
    "status": "success"
}

```
