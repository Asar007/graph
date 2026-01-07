import json
import datetime
import os

# --- CONFIGURATION (From User Request) ---
DAILY_BUDGET = 1.00  # $1.00 Limit
USAGE_FILE = "daily_usage.json"

# Pricing (User Provided)
COST_PER_1M_INPUT = 0.30
COST_PER_1M_OUTPUT = 2.50

def get_todays_cost():
    """Reads the current day's cost from a local file."""
    today_str = datetime.date.today().isoformat()
    
    if not os.path.exists(USAGE_FILE):
        return 0.0, today_str
        
    try:
        with open(USAGE_FILE, "r") as f:
            data = json.load(f)
            
        # If the date in the file is not today, reset cost to 0
        if data.get("date") != today_str:
            return 0.0, today_str
            
        return data.get("cost", 0.0), today_str
    except Exception:
        return 0.0, today_str

def update_cost(new_cost):
    """Updates the total cost in the local file."""
    current_cost, today_str = get_todays_cost()
    total_cost = current_cost + new_cost
    
    try:
        with open(USAGE_FILE, "w") as f:
            json.dump({"date": today_str, "cost": total_cost}, f)
    except Exception as e:
        print(f"Error saving usage: {e}")
    
    return total_cost

def check_budget_limit():
    """Checks if we have hit the limit. Raises exception if so."""
    current_spending, _ = get_todays_cost()
    if current_spending >= DAILY_BUDGET:
        raise Exception(f"KILL SWITCH ENGAGED: Daily limit of ${DAILY_BUDGET} reached! (Current: ${current_spending:.4f})")
    return current_spending

def track_usage(response):
    """Calculates cost from response metadata and updates tracker."""
    try:
        # Check if usage_metadata exists (it should for successful generation)
        if not hasattr(response, 'usage_metadata'):
            return 0.0, 0.0
            
        usage = response.usage_metadata
        in_tokens = usage.prompt_token_count
        out_tokens = usage.candidates_token_count
        
        cost_input = (in_tokens / 1_000_000) * COST_PER_1M_INPUT
        cost_output = (out_tokens / 1_000_000) * COST_PER_1M_OUTPUT
        call_cost = cost_input + cost_output
        
        new_total = update_cost(call_cost)
        return call_cost, new_total
    except Exception as e:
        print(f"Billing Error: {e}")
        return 0.0, 0.0
