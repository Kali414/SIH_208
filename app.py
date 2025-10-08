from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# -------------------- Load Environment Variables --------------------
load_dotenv()

# -------------------- Flask App --------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, allow_headers="*")
# -------------------- Groq Client --------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file or environment variables.")

# Initialize ChatGroq client
client = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

# -------------------- Prompt Template --------------------
PROMPT_TEMPLATE = """
You are an expert logistics optimization AI. Generate a logistics plan in a single valid JSON object ONLY.

Orders:
{orders}

Stockyards:
{stockyards}

Rakes:
{rakes}

Routes:
{routes}

Task & Logic:
1. Match orders to stockyards based on material availability.
2. Prioritize high-priority or larger orders first.
3. Assign available rakes ensuring total quantity ≤ rake capacity.
4. For each rake, determine the best source stockyard.
5. List all unique destinations for orders assigned to a rake.
6. Construct a multi-stop route if a rake has multiple destinations.
7. If an order cannot be fulfilled, include it in "remaining_orders" with a reason.

Output JSON format:
{{
  "rakes": [
    {{
      "rake_id": "string",
      "assigned_orders": ["list", "of", "order_ids"],
      "source_stockyard": "string",
      "destinations": ["list", "of", "destination_cities"],
      "total_load": "integer",
      "mode": "string",
      "selected_route": ["list", "of", "route_legs_as_strings"]
    }}
  ],
  "remaining_orders": [
    {{
      "order_id": "string",
      "reason": "string"
    }}
  ]
}}
"""

# -------------------- Helper Functions --------------------
def create_prompt(data: dict) -> str:
    """Format input data into prompt for Groq LLM."""
    orders = json.dumps(data.get("orders", []))
    stockyards = json.dumps(data.get("stockyards", []))
    rakes = json.dumps(data.get("rakes", []))
    routes = json.dumps(data.get("routes", []))
    return PROMPT_TEMPLATE.format(
        orders=orders,
        stockyards=stockyards,
        rakes=rakes,
        routes=routes
    )

def query_groq(prompt: str) -> str:
    """Send prompt to Groq LLM and return raw text."""
    try:
        response = client.invoke([{"role": "system", "content": prompt}])
        # Check if response is a list or AIMessage
        if isinstance(response, list) and len(response) > 0:
            return response[0].content
        elif hasattr(response, "content"):
            return response.content
        return str(response)
    except Exception as e:
        return f"Error querying Groq: {str(e)}"

def parse_json_safe(llm_output: str) -> dict:
    """Attempt to parse JSON from LLM output safely."""
    try:
        start = llm_output.find("{")
        end = llm_output.rfind("}") + 1
        if start != -1 and end != -1:
            json_str = llm_output[start:end]
            return json.loads(json_str)
    except Exception:
        pass
    # fallback
    return {"raw_output": llm_output}

# -------------------- API Routes --------------------
@app.route("/")
def home():
    return "Rake Optimization DSS API with LangChain + ChatGroq"

@app.route("/optimize", methods=["POST"])
def optimize():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # Create prompt
    prompt = create_prompt(data)

    # Query Groq LLM
    llm_output = query_groq(prompt)

    # Parse JSON safely
    result_json = parse_json_safe(llm_output)

    return jsonify({"status": "success", "optimization_result": result_json})

# -------------------- Run Server --------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
