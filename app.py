import streamlit as st
import google.generativeai as genai
import json
import os
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
import time
import difflib
from streamlit_ace import st_ace
import billing # Import Billing Module

# Load environment variables
load_dotenv(override=True)

# Set page configuration - Must be the first st command
st.set_page_config(layout="wide", page_title="Ai Graph Generator", initial_sidebar_state="collapsed")

def load_prompt_template():
    try:
        with open("json_only_prompt.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        st.error("Error: json_only_prompt.txt not found.")
        return None

def load_modification_prompt():
    try:
        with open("modification_prompt.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        st.error("Error: modification_prompt.txt not found.")
        return None

def load_html_template():
    try:
        with open("template.html", "r", encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        st.error("Error: template.html not found.")
        return None

def load_mindmap_prompt():
    try:
        with open("mindmap_prompt.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        return None

def load_sequence_prompt():
    try:
        with open("sequence_prompt.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        return None

def load_sequence_template():
    try:
        with open("sequence_template.html", "r", encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return None

def load_timeline_prompt():
    try:
        with open("timeline_prompt.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        return None

def load_timeline_template():
    try:
        with open("timeline_template.html", "r", encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return None

def validate_json(json_str):
    try:
        # Regex to find JSON block enclosed in ```json ... ``` or just { ... }
        match = re.search(r'```json\s*(\{.*?\})\s*```', json_str, re.DOTALL)
        if match:
             json_str = match.group(1)
        else:
             # Fallback: Try to find the first outer {} block
             match_fallback = re.search(r'\{.*\}', json_str, re.DOTALL)
             if match_fallback:
                 json_str = match_fallback.group(0)
        
        data = json.loads(json_str)
        
        # Determine schema validation type
        if 'participants' in data and 'events' in data:
            # Sequence Diagram Schema
            if not isinstance(data.get('participants'), list):
                 print("Validation Error: 'participants' must be a list")
                 return None
            if not isinstance(data.get('events'), list):
                 print("Validation Error: 'events' must be a list")
                 return None

            return data

        # Timeline Schema
        if 'mermaid_syntax' in data:
            return data

        # Graph/Mindmap Schema (Fallback)
        required_fields = ['nodes', 'hierarchy', 'edges']
        for field in required_fields:
            if field not in data:
                print(f"Validation Error: Missing field '{field}'")
                return None
                
        if not isinstance(data.get('nodes'), list):
             print("Validation Error: 'nodes' must be a list")
             return None

        # Validate each node has an ID
        for i, node in enumerate(data['nodes']):
            if not isinstance(node, dict) or 'id' not in node:
                print(f"Validation Error: Node at index {i} missing 'id' or not an object")
                return None
                
        if not isinstance(data.get('edges'), list):
             print("Validation Error: 'edges' must be a list")
             return None

        # Validate each edge has source/target
        for i, edge in enumerate(data['edges']):
            if not isinstance(edge, dict) or 'source' not in edge or 'target' not in edge:
                print(f"Validation Error: Edge at index {i} missing source/target")
                return None
        
        return data
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return None

def inject_data_into_html(html_content, json_data):
    json_str = json.dumps(json_data, indent=2)
    # Escape </script> to prevent breaking HTML
    json_str = json_str.replace("</", "<\\/")

    start_marker = "/* [INJECTION_START] */"
    end_marker = "/* [INJECTION_END] */"
    
    start_idx = html_content.find(start_marker)
    end_idx = html_content.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        # Calculate split points
        # Keep the content before the start marker (and the marker itself for clarity if desired, but let's replace OUT the marker to avoid clutter or keep it?)
        # Let's keep the markers so we can verify inspection later if needed.
        
        pre_content = html_content[:start_idx]
        post_content = html_content[end_idx + len(end_marker):]
        
        # We need to construct the new block.
        # Note: We are replacing everything BETWEEN start_idx and end_idx + len(end_marker).
        # Actually, let's just replace the whole block.
        
        new_block = f"{start_marker}\n            const architectureData = {json_str};\n            {end_marker}"
        return pre_content + new_block + post_content
    
    else:
        st.error("Could not find the insertion point in the HTML template.")
        return html_content

def generate_graph(topic, api_key, graph_type="Graph"):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt_template = None
    html_loader = load_html_template

    if graph_type == "Mindmap":
        prompt_template = load_mindmap_prompt()
        if not prompt_template:
            return None, None, "Mindmap prompt template (mindmap_prompt.txt) not found."
    elif graph_type == "Sequence":
        prompt_template = load_sequence_prompt()
        html_loader = load_sequence_template
        if not prompt_template:
            return None, None, "Sequence prompt template (sequence_prompt.txt) not found."
    elif graph_type == "Timeline":
        prompt_template = load_timeline_prompt()
        html_loader = load_timeline_template
        if not prompt_template:
            return None, None, "Timeline prompt template (timeline_prompt.txt) not found."
    else:
        prompt_template = load_prompt_template()
        
    if prompt_template:
        final_prompt = prompt_template.replace("[INSERT TOPIC HERE]", topic)
        try:
            # Upsert reliability logic even for the older model
            response = call_gemini_with_retry(model, final_prompt)
            json_data = validate_json(response.text)
            if json_data:
                html_template = html_loader()
                if html_template:
                    new_html = inject_data_into_html(html_template, json_data)
                    return new_html, json_data, None
            else:
                # Include a snippet of the raw text for debugging
                raw_snippet = response.text[:500].replace('\n', ' ')
                return None, None, f"Failed to generate valid JSON. Model Output start: {raw_snippet}..."
        except Exception as e:
            return None, None, f"Exception: {str(e)}"
    return None, None, "Prompt template not found."

def load_mindmap_modification_prompt():
    try:
        with open("mindmap_modification_prompt.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        st.error("Error: mindmap_modification_prompt.txt not found.")
        return None

def modify_graph(current_json, prompt, api_key, graph_type="Graph"):
    genai.configure(api_key=api_key)
    # Using the user's preferred model, but now wrapped in retry logic
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    template = None
    if graph_type == "Mindmap":
        template = load_mindmap_modification_prompt()
    else:
        template = load_modification_prompt()

    if not template:
        return None, None, "Modification prompt file missing."
        
    # Replace placeholders in the text file
    final_prompt = template.replace("[INSERT CURRENT JSON DATA HERE]", json.dumps(current_json))
    final_prompt = final_prompt.replace("[INSERT CHANGE REQUEST HERE]", prompt)
    
    try:
        # Robust execution
        response = call_gemini_with_retry(model, final_prompt)
        new_json_data = clean_json_output(response.text)
        
        if not new_json_data:
             return None, None, "Failed to parse JSON from AI response."
        
        html_loader = load_html_template
        if graph_type == "Sequence":
            html_loader = load_sequence_template
        elif graph_type == "Timeline":
            html_loader = load_timeline_template
            
        html_template = html_loader()
        
        if html_template:
            new_html = inject_data_into_html(html_template, new_json_data)
            return new_html, new_json_data, None
            
    except Exception as e:
        return None, None, str(e)
        
    return None, None, "Unknown error during modification."

# --- Helper Functions for Simple Editor ---
def json_to_dsl(json_data):
    """Converts JSON graph data to Simple DSL Format."""
    if not json_data:
        return ""
    
    lines = []
    # 1. Edges first (simplest to read)
    if 'edges' in json_data:
        for edge in json_data['edges']:
            src = edge.get('source', '?')
            tgt = edge.get('target', '?')
            label = edge.get('label', '')
            
            line = f"{src} -> {tgt}"
            if label:
                line += f" : {label}"
            lines.append(line)
            
    # 2. Nodes that might not be in edges (orphans)
    if 'nodes' in json_data:
        existing_nodes = set()
        if 'edges' in json_data:
            for edge in json_data['edges']:
                existing_nodes.add(edge.get('source'))
                existing_nodes.add(edge.get('target'))
        
        for node in json_data['nodes']:
            node_id = node.get('id')
            if node_id and node_id not in existing_nodes:
                lines.append(node_id)
                
    return "\n".join(lines)

def dsl_to_json(dsl_text, current_json=None):
    """Converts Simple DSL text back to JSON structure, preserving formatting if possible."""
    new_nodes = {} # Map id -> node_obj
    new_edges = []
    
    # helper to ensure node exists
    def get_or_create_node(node_id):
        if node_id not in new_nodes:
            # Try to preserve existing node data if available
            existing_data = None
            if current_json and 'nodes' in current_json:
                existing_data = next((n for n in current_json['nodes'] if n['id'] == node_id), None)
            
            if existing_data:
                new_nodes[node_id] = existing_data
            else:
                new_nodes[node_id] = {"id": node_id, "label": node_id}
        return new_nodes[node_id]

    lines = dsl_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'): continue
        
        # Parse Arrow "->"
        if "->" in line:
            parts = line.split("->")
            source_id = parts[0].strip()
            remainder = parts[1].strip()
            
            target_id = remainder
            label = None
            
            # Parse Label ":"
            if ":" in remainder:
                r_parts = remainder.split(":")
                target_id = r_parts[0].strip()
                label = r_parts[1].strip()
            
            get_or_create_node(source_id)
            get_or_create_node(target_id)
            
            edge_obj = {
                "id": f"e-{source_id}-{target_id}-{len(new_edges)}",
                "source": source_id,
                "target": target_id
            }
            if label:
                edge_obj["label"] = label
            new_edges.append(edge_obj)
            
        else:
            # Just a node
            get_or_create_node(line)
            
    return {
        "nodes": list(new_nodes.values()),
        "edges": new_edges,
        "hierarchy": {}, # Reset hierarchy for now as DSL doesn't easily capture it
        "details": current_json.get('details', {}) if current_json else {}
    }

# --- Sequence DSL Helpers ---

def sequence_json_to_dsl(json_data):
    """Converts Sequence JSON to Mermaid-like DSL."""
    if not json_data: return ""
    lines = []
    
    # 1. Participants
    if 'participants' in json_data:
        for p in json_data['participants']:
            p_type = p.get('type', 'Participant')
            p_id = p.get('id')
            p_label = p.get('label', p_id)
            if p_type == 'Actor':
                lines.append(f"actor {p_label} as {p_id}")
            else:
                lines.append(f"participant {p_label} as {p_id}")
    
    lines.append("") # Spacer
    
    # 2. Events & Fragments mixed by step
    # We need to reconstruct the flow. 
    # Events have 'step'. Fragments have 'startStep' and 'endStep'.
    # Activations have 'startStep' and 'endStep'.
    
    events = sorted(json_data.get('events', []), key=lambda x: x.get('step', 0))
    fragments = json_data.get('fragments', [])
    activations = json_data.get('activations', [])
    
    # Map steps to actions
    max_step = 0
    if events: max_step = max(e.get('step', 0) for e in events)
    
    # Create a map of what happens at each step
    # This is tricky because fragments span steps.
    # Simplified approach: Iterate steps 1 to N.
    
    # Pre-process fragments starting/ending
    frag_starts = {} # step -> list of frags
    frag_ends = {}   # step -> list of frags
    for f in fragments:
        s = f.get('startStep')
        e = f.get('endStep')
        if s: frag_starts.setdefault(s, []).append(f)
        if e: frag_ends.setdefault(e, []).append(f)
        max_step = max(max_step, e or 0)

    # Pre-process activations
    act_starts = {}
    act_ends = {}
    for a in activations:
        s = a.get('startStep')
        e = a.get('endStep')
        if s: act_starts.setdefault(s, []).append(a)
        if e: act_ends.setdefault(e, []).append(a)
        max_step = max(max_step, e or 0)
        
    indent_level = 0
    indent_str = "  "
    
    for step in range(1, max_step + 2):
        # Check Fragment Starts
        if step in frag_starts:
            for f in frag_starts[step]:
                lines.append(f"{indent_str * indent_level}alt {f.get('condition', '')}")
                indent_level += 1
        
        # Check Events at this step
        step_events = [e for e in events if e.get('step') == step]
        for e in step_events:
            src = e.get('source')
            tgt = e.get('target')
            lbl = e.get('label')
            l_type = e.get('lineType', 'solid')
            a_type = e.get('arrowType', 'solid')
            
            arrow = "->"
            if l_type == 'dotted':
                arrow = "-->"
            elif a_type == 'open_arrow': # Async usually solid line open arrow
                 arrow = "->>"
            
            lines.append(f"{indent_str * indent_level}{src} {arrow} {tgt} : {lbl}")
            
        # Check Activation Starts (after event usually)
        if step in act_starts:
            for a in act_starts[step]:
                lines.append(f"{indent_str * indent_level}activate {a.get('participant')}")
                
        # Check Activation Ends
        if step in act_ends:
            for a in act_ends[step]:
                lines.append(f"{indent_str * indent_level}deactivate {a.get('participant')}")

        # Check Fragment Ends
        if step in frag_ends:
            for f in frag_ends[step]:
                indent_level = max(0, indent_level - 1)
                lines.append(f"{indent_str * indent_level}end")
                
    return "\n".join(lines)

def sequence_dsl_to_json(dsl_text):
    """Parses Mermaid-like Sequence DSL to JSON."""
    lines = dsl_text.split('\n')
    
    participants = []
    participant_ids = set()
    events = []
    activations = []
    fragments = []
    
    fragment_stack = [] # Stack of {type, startStep, condition}
    activation_stack = {} # map participant_id -> startStep
    
    current_step = 1
    
    def get_or_create_participant(name_or_id, label=None, p_type="Participant"):
        p_id = name_or_id.replace(" ", "_") # Simple ID sanitization
        if label:
            display_label = label
            final_id = p_id # definition uses ID
        else:
            # Usage like A -> B. ID is A. Label is A.
            display_label = name_or_id
            final_id = p_id
            
        if final_id not in participant_ids:
            participant_ids.add(final_id)
            participants.append({
                "id": final_id,
                "label": display_label,
                "type": p_type,
                "description": ""
            })
        return final_id

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'): continue
        
        # 1. Definitions
        # participant Label as ID
        # actor Label as ID
        if line.startswith('participant ') or line.startswith('actor '):
            parts = line.split(' ')
            p_type = "Actor" if parts[0] == 'actor' else "Participant"
            
            # handle "as"
            if " as " in line:
                # participant "My API" as api
                # regex might be better but let's do simple split
                segments = line.split(" as ")
                name_part = segments[0].replace(parts[0] + " ", "").strip().replace('"', '')
                id_part = segments[1].strip()
                get_or_create_participant(id_part, label=name_part, p_type=p_type)
            else:
                # participant User
                name = line.replace(parts[0] + " ", "").strip().replace('"', '')
                get_or_create_participant(name, label=name, p_type=p_type)
            continue
            
        # 2. Fragments
        if line.startswith('alt '):
            cond = line[4:].strip()
            fragment_stack.append({
                "type": "alt", 
                "condition": cond, 
                "startStep": current_step
            })
            current_step += 1 # Ensure content starts on next line
            continue
        
        if line == 'end':
            if fragment_stack:
                frag = fragment_stack.pop()
                frag['endStep'] = current_step
                fragments.append(frag)
                current_step += 1 # Ensure subsequent content starts on next line
            continue
            
        # 3. Activations
        if line.startswith('activate '):
            p_id = line.split(' ')[1].strip()
            # Ensure participant exists
            p_id = get_or_create_participant(p_id)
            
            # Backdate logic: If the LAST event was a message TO this participant
            # and it happened at the previous step, assume this activation belongs to it.
            start_step_candidate = current_step
            if events:
                 last_evt = events[-1]
                 # events are {step, source, target...}
                 # Check if last event target is this participant AND it was one step ago
                 # (Since we increment current_step after adding event)
                 if last_evt['target'] == p_id and last_evt['step'] == current_step - 1:
                     start_step_candidate = current_step - 1
            
            activation_stack[p_id] = start_step_candidate
            continue
            
        if line.startswith('deactivate '):
            p_id = line.split(' ')[1].strip()
            p_id = get_or_create_participant(p_id)
            if p_id in activation_stack:
                start_s = activation_stack.pop(p_id)
                activations.append({
                    "participant": p_id,
                    "startStep": start_s,
                    "endStep": current_step 
                })
            continue

        # 4. Messages
        # A -> B : Label
        # A --> B : Label
        # A ->> B : Label
        arrow_map = {
            "->>": ("open_arrow", "solid"), # Async
            "-->": ("open_arrow", "dotted"), # Return
            "->": ("solid", "solid")      # Sync
        }
        
        matched_arrow = None
        for arrow, props in arrow_map.items():
            if arrow in line:
                # Check for longer matches first (--> before ->)
                # But dict order isn't guaranteed.
                # Actually "-->" contains "->", so we must be careful.
                # Regex is safer.
                pass

        # Simple parsing logic relies on finding the split point
        # Check "-->" first, then "->>", then "->"
        
        parts = None
        arrow_type = "solid"
        line_type = "solid"
        
        if "-->" in line:
            parts = line.split("-->")
            arrow_type, line_type = "open_arrow", "dotted"
        elif "->>" in line:
             parts = line.split("->>")
             arrow_type, line_type = "open_arrow", "solid"
        elif "->" in line:
             parts = line.split("->")
             arrow_type, line_type = "solid", "solid"
             
        if parts:
            src = parts[0].strip()
            remainder = parts[1].strip()
            tgt = remainder
            label = ""
            
            if ":" in remainder:
                r_parts = remainder.split(":", 1)
                tgt = r_parts[0].strip()
                label = r_parts[1].strip()
            
            src_id = get_or_create_participant(src)
            tgt_id = get_or_create_participant(tgt)
            
            # --- IMPLICIT ACTIVATION LOGIC ---
            # Rule 1: A -> B (Sync Call) implies Activate B
            if arrow_type == "solid" and line_type == "solid":
                # Check if B is already active (explicitly or implicitly)
                if tgt_id not in activation_stack:
                    # Start implicit activation
                    activation_stack[tgt_id] = current_step
                    # We mark it as 'implicit' in a separate set if we wanted to be strict,
                    # but for now, treating it same as explicit is fine.
            
            # Rule 2: B --> A (Return) implies Deactivate B
            if arrow_type == "open_arrow" and line_type == "dotted":
                # Check if B is active
                if src_id in activation_stack:
                     start_s = activation_stack.pop(src_id)
                     activations.append({
                        "participant": src_id,
                        "startStep": start_s,
                        "endStep": current_step
                     })

            events.append({
                "step": current_step,
                "type": "message",
                "source": src_id,
                "target": tgt_id,
                "label": label,
                "arrowType": arrow_type,
                "lineType": line_type
            })
            
            current_step += 1 # Increment step after event

    return {
        "participants": participants,
        "events": events,
        "activations": activations,
        "fragments": fragments,
        "metadata": {"title": "Sequence Diagram", "summary": "Generated via DSL"}
    }
def clean_json_output(text):
    """Extracts JSON from text, handling markdown blocks."""
    try:
        # regex for ```json ... ```
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            text = match.group(1)
        else:
            # fallback to first brace pair
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                text = match.group(0)
        return json.loads(text)
    except Exception:
        return None

def call_gemini_with_retry(model, prompt, max_retries=5, retry_delay=5):
    """Executes a Gemini generation with robust rate-limit handling AND Billing Enforcement."""
    
    # 1. CHECK BUDGET
    billing.check_budget_limit()
    
    for attempt in range(max_retries):
        try:
            # 2. CALL API
            response = model.generate_content(prompt)
            
            # 3. TRACK USAGE
            billing.track_usage(response)
            
            return response
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < max_retries - 1:
                # rate limit hit
                wait_time = retry_delay
                # Try to extract wait time from error message
                match = re.search(r'retry in (\d+(\.\d+)?)s', error_str)
                if match:
                    wait_time = float(match.group(1)) + 1 # Add buffer
                
                # Use max of calculated backoff or requested wait
                actual_wait = max(wait_time, retry_delay)
                time.sleep(actual_wait)
                
                retry_delay *= 2 # Exponential backoff
                continue
            else:
                raise e
    raise Exception("Max retries reached")

def generate_event_from_line(line_text, api_key):
    """Uses LLM to parse a single DSL line into a JSON event object."""
    # Tiny prompt for fast, cheap execution
    prompt = f"""
    Parse to JSON event object (source, target, label, arrowType, lineType):
    "{line_text}"
    
    schema: {{"type": "message", "source": "id", "target": "id", "label": "str", "arrowType": "solid|open_arrow", "lineType": "solid|dotted"}}
    """
    genai.configure(api_key=api_key)
    # Switch to stable model for reliability
    model = genai.GenerativeModel('gemini-2.5-flash') 
    try:
        # Use retry logic here too!
        resp = call_gemini_with_retry(model, prompt)
        return clean_json_output(resp.text)
    except:
        return None

def apply_smart_patch(old_dsl, new_dsl, current_json, api_key):
    """
    Attempts to update current_json based on a small diff.
    Returns new_json if successful, or None if full regen is needed.
    """
    if not old_dsl or not current_json: return None
    
    old_lines = old_dsl.splitlines()
    new_lines = new_dsl.splitlines()
    
    # We only handle 1:1 modifications for now (same line count)
    if len(old_lines) != len(new_lines):
        return None
        
    diff = list(difflib.ndiff(old_lines, new_lines))
    
    # Analyze diff to find the changed index
    # ndiff produces lines starting with "- ", "+ ", "  "
    # If we see exactly one "-" and one "+" at the same relative position, it's a mod.
    
    changes = [d for d in diff if d.startswith("- ") or d.startswith("+ ")]
    if len(changes) != 2: return None # Too complex or no changes
    
    if not (changes[0].startswith("- ") and changes[1].startswith("+ ")):
        return None # Not a simple swap
        
    # Find the index
    changed_index = -1
    for i, (o, n) in enumerate(zip(old_lines, new_lines)):
        if o != n:
            changed_index = i
            break
            
    if changed_index == -1: return None
    
    new_line = new_lines[changed_index].strip()
    
    # IGNORE structural lines for patching (alt, loop, end, activate)
    # We only patch messages: "A -> B: msg"
    if any(x in new_line.lower() for x in ["alt ", "loop ", "opt ", "end", "activate ", "deactivate ", "participant ", "actor "]):
        return None 
        
    # Calculate which "Event" this corresponds to
    # We need to count how many "message" lines were above this line in the OLD DSL
    event_index = 0
    for i in range(changed_index):
        l = old_lines[i].strip()
        # Count if it looks like a message
        if "->" in l:
            event_index += 1
            
    # Check bounds
    if event_index >= len(current_json.get('events', [])):
        return None
        
    # Generate the new event object
    new_event = generate_event_from_line(new_line, api_key)
    if not new_event: return None
    
    # Update in place
    import copy
    new_data = copy.deepcopy(current_json)
    
    # Must preserve the 'step' of the old event to keep ordering valid
    old_event = new_data['events'][event_index]
    new_event['step'] = old_event.get('step', 1)
    
    new_data['events'][event_index] = new_event
    
    # Update title/metadata to show it was patched
    new_data['metadata']['summary'] = "Patched via Smart Edit"
    
    return new_data

def generate_json_from_dsl(dsl_text, api_key, graph_type="Sequence"):
    """Uses Gemini to convert DSL text into valid Graph JSON."""
    if not dsl_text.strip(): return None
    
    system_instruction = f"""
    You are an expert software architect. 
    Convert the following "{graph_type}" description (DSL) into a VALID JSON object adhering strictly to the schema.
    
    Input DSL:
    {dsl_text}
    """
    
    # OPTIMIZATION: Use a condensed schema instead of the full prompt file to save tokens.
    minified_schema = """
    {
      "participants": [
        {"id": "str", "label": "str", "type": "Actor|Participant", "description": "str"}
      ],
      "events": [
        {"step": 1, "type": "message", "source": "id", "target": "id", "label": "str", "arrowType": "solid|open_arrow", "lineType": "solid|dotted"}
      ],
      "activations": [{"participant": "id", "startStep": int, "endStep": int}],
      "fragments": [{"type": "alt", "condition": "str", "startStep": int, "endStep": int}]
    }
    """
        
    final_prompt = f"""
    {system_instruction}
    
    Convert to valid JSON matching this structure:
    {minified_schema}
    
    RULES:
    1. Output ONLY valid JSON.
    2. Auto-fill missing descriptions.
    3. Ensure 'step' numbers are sequential integers starting at 1.
    4. ALIGNMENT RULES:
        - If A sends a SYNC message ("->") to B, B MUST start an activation at that step.
        - If B sends a RETURN message ("-->") to A, B MUST end its activation at that step.
        - Nested calls (A->B, B->C) must stack activations correctly.
    5. 'activations' and 'fragments' must align with event steps.
    """

    genai.configure(api_key=api_key)
    # Switch to stable flash model which often has better rate limits than experimental
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    try:
        response = call_gemini_with_retry(model, final_prompt)
        cleaned_json = clean_json_output(response.text)
        return cleaned_json
    except Exception as e:
        raise Exception(f"Gemini Conversion Failed: {str(e)}")



# --- AI Assistant Logic ---

class GraphAssistant:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.api_key = api_key

    def _get_system_prompt(self, current_graph_context=None, graph_type="Graph", focused_context=None):
        base_prompt = """
        You are an advanced AI Data Visualization Assistant.
        Your goal is to help the user understand information by creating or modifying diagrams, OR by explaining existing ones.
        
        You have two modes of operation:
        1. **CONVERSATIONAL**: Answer questions, explain concepts, or discuss the current graph.
        2. **ACTION**: Generate a new graph or Modify the existing one.
        
        Tool Use Protocol:
        If the user wants to CREATE or CHANGE a visual, you must return a structured TOOL CALL.
        If the user just wants to talk, return normal text.
        
        TOOL CALL FORMAT:
        You must output a JSON block for actions:
        ```json
        {
          "action": "GENERATE" | "MODIFY",
          "target_type": "Graph" | "Mindmap" | "Sequence" | "Timeline" | null,
          "topic_or_instruction": "..."
        }
        ```
        
        - Use "GENERATE" if the user asks for a new topic or a completely new diagram.
        - Use "MODIFY" if the user asks to change, add, delete, or update the CURRENT diagram.
        - Use "target_type" ONLY if the user explicitly asks to convert or switch to a different view (e.g., "Show as Mindmap"). Otherwise null.
        """
        
        context_str = ""
        if focused_context:
             context_str = f"""
            FOCUSED NODE/CONTEXT:
            {json.dumps(focused_context, indent=2)}
            
            FULL GRAPH CONTEXT ({graph_type}):
            {json.dumps(current_graph_context, indent=2)}
            """
        elif current_graph_context:
            context_str = f"""
            CURRENT GRAPH CONTEXT ({graph_type}):
            {json.dumps(current_graph_context, indent=2)}
            """
        else:
            context_str = "CURRENT GRAPH CONTEXT: None (No graph generated yet)."
            
        return f"{base_prompt}\n\n{context_str}"

    def process_request(self, user_input, chat_history, current_graph_context=None, graph_type="Graph", focused_context=None):
        """
        Decides whether to chat or perform an action.
        Returns: (response_text, action_dict)
        """
        system_instruction = self._get_system_prompt(current_graph_context, graph_type, focused_context)
        
        # Construct chat history for the model
        messages = [
            {"role": "user", "parts": [system_instruction]} 
        ]
        
        # Add recent history (last 5 messages to avoid blowing context)
        # Assuming chat_history is list of {"role": "user"|"assistant", "content": "..."}
        if chat_history:
            for msg in chat_history[-5:]:
                 role = "model" if msg["role"] == "assistant" else "user"
                 messages.append({"role": role, "parts": [str(msg["content"])]})
                 
        messages.append({"role": "user", "parts": [user_input]})
        
        try:
            response = call_gemini_with_retry(self.model, messages[-1]['parts'][0]) 
            
            # Use a fresh chat session for the actual interaction to ensure clean state or use the direct prompt method
            combined_prompt = f"{system_instruction}\n\nUSER INPUT: {user_input}"
            response = call_gemini_with_retry(self.model, combined_prompt)
            
            text = response.text
            
            # Check for JSON tool call
            action_data = clean_json_output(text)
            
            if action_data and "action" in action_data:
                # It's an action!
                return None, action_data
            else:
                # It's just a conversation response
                return text, None
                
        except Exception as e:
            return f"Error processing request: {e}", None

def render_live_editor(graph_type, api_key):
    st.subheader("Live Editor")
    if st.session_state.current_json_data:
        
        # Tabs for different editing modes
        tab_simple, tab_json = st.tabs(["Simple Mode", "Advanced JSON"])
        
        with tab_simple:
            is_sequence = (graph_type == "Sequence")
            st.caption("Edit structure using Simple DSL.")
            if is_sequence:
                st.info(
                    """
                    **Sequence Syntax Guide:**
                    - **Sync Message:** `A -> B : Message` (Solid Line)
                    - **Async Message:** `A ->> B : Message` (Open Arrow)
                    - **Return Message:** `A --> B : Reply` (Dotted Line)
                    - **Self Message:** `A -> A : Internal process`
                    - **Alternative Flow:**
                      ```
                      alt Invalid Token
                        A --> B : 401 Error
                      end
                      ```
                    - **Activations:** `activate A` / `deactivate A`
                    """
                )
            else:
                st.info("💡 **Syntax:** `Node A -> Node B : Label`")
            
            # 1. Convert Current JSON -> DSL for initial view
            current_dsl = ""
            try:
                if is_sequence:
                    current_dsl = sequence_json_to_dsl(st.session_state.current_json_data)
                else:
                    current_dsl = json_to_dsl(st.session_state.current_json_data)
            except Exception:
                current_dsl = ""
            
            # Dynamic key to prevent state cross-talk between modes
            text_area_key = f"dsl_input_{graph_type}"
            
            st.caption("Simple Definition (Auto-updates)")
            dsl_input = st_ace(
                value=current_dsl,
                language='markdown',
                theme='monokai', # Nice dark theme
                height=750,
                key=text_area_key,
                auto_update=True, # Attempt instant updates
                show_gutter=True, # Line numbers!
                wrap=True
            )
            
            # Trigger update if input differs from the canonical DSL of the current JSON
            if dsl_input != current_dsl:
                # User Changed DSL -> Update JSON
                try:
                    new_json_from_dsl = None
                    if is_sequence:
                        new_json_from_dsl = sequence_dsl_to_json(dsl_input)
                    else:
                        new_json_from_dsl = dsl_to_json(dsl_input, st.session_state.current_json_data)
                    
                    if new_json_from_dsl and new_json_from_dsl != st.session_state.current_json_data:
                        st.session_state.current_json_data = new_json_from_dsl
                        # Update HTML
                        if graph_type == "Sequence":
                             html_loader = load_sequence_template
                        elif graph_type == "Timeline":
                             html_loader = load_timeline_template
                        else:
                             html_loader = load_html_template
                        
                        html_template = html_loader()
                        if html_template:
                            new_html = inject_data_into_html(html_template, new_json_from_dsl)
                            st.session_state.html_content = new_html
                            st.rerun()
                except Exception as e:
                    st.error(f"Error parsing DSL: {e}")

            # Add an AI "Magic Fix" button for complex requests or if user wants LLM help
            if is_sequence:
                if st.button("✨ AI Magic Fix / Regenerate", help="Use AI to rewrite the diagram based on your text (Slower but understands natural language)."):
                    if not api_key:
                        st.error("API Key required.")
                    else:
                        with st.spinner("🤖 Gemini is rewriting..."):
                            try:
                                # Use the robust generate function
                                ai_json = generate_json_from_dsl(dsl_input, api_key, "Sequence")
                                if ai_json and ai_json != st.session_state.current_json_data:
                                    st.session_state.current_json_data = ai_json
                                    # Update HTML
                                    html_template = load_sequence_template()
                                    if html_template:
                                        new_html = inject_data_into_html(html_template, ai_json)
                                        st.session_state.html_content = new_html
                                        st.rerun()
                            except Exception as e:
                                st.error(f"AI Generation Failed: {e}")

        with tab_json:
            # Convert current JSON to string for the text area
            json_str = json.dumps(st.session_state.current_json_data, indent=2)
            
            st.caption("Edit JSON Data (Press Ctrl+Enter to apply)")
            edited_json_str = st_ace(
                value=json_str, 
                language='json',
                theme='monokai',
                height=750, 
                key="live_editor_area",
                auto_update=False,
                show_gutter=True,
                wrap=True
            )

            if edited_json_str != json_str:
                try:
                    new_json_data = json.loads(edited_json_str)
                    
                    # Only update if semantically different
                    if new_json_data != st.session_state.current_json_data:
                        st.session_state.current_json_data = new_json_data
                        
                        html_loader = load_html_template
                        if "participants" in new_json_data and "events" in new_json_data:
                             html_loader = load_sequence_template
                        elif "mermaid_syntax" in new_json_data:
                             html_loader = load_timeline_template
                        
                        html_template = html_loader()
                        if html_template:
                            new_html = inject_data_into_html(html_template, new_json_data)
                            st.session_state.html_content = new_html
                            st.rerun() 
                        
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")
                except Exception as e:
                    st.error(f"Error updating graph: {e}")
    else:
        st.info("Generate a graph to see the editor.")

def save_prompt_callback(file_name, key):
    try:
        content = st.session_state[key]
        with open(file_name, "w") as f:
            f.write(content)
        st.toast(f"Saved {file_name}!", icon="💾")
    except Exception as e:
        st.error(f"Error saving file: {e}")

def reset_prompt_callback(file_name, key):
    try:
        default_file = file_name.replace(".txt", "_default.txt")
        if os.path.exists(default_file):
            with open(default_file, "r") as df:
                default_content = df.read()
            with open(file_name, "w") as f:
                f.write(default_content)
            
            # Update session state
            st.session_state[key] = default_content
            st.toast(f"Reset {file_name} to default!", icon="🔄")
        else:
            st.error(f"Default file {default_file} not found.")
    except Exception as e:
        st.error(f"Error checking reset: {e}")

def render_prompt_settings():
    st.markdown("---")
    with st.expander("Prompt Settings"):
        prompt_files = {
            "Graph Prompt": "json_only_prompt.txt",
            "Mindmap Prompt": "mindmap_prompt.txt",
            "Sequence Prompt": "sequence_prompt.txt",
            "Timeline Prompt": "timeline_prompt.txt",
            "Modification Prompt": "modification_prompt.txt", 
            "Mindmap Mod Prompt": "mindmap_modification_prompt.txt"
        }
        
        selected_prompt = st.selectbox("Select Prompt to Edit", list(prompt_files.keys()))
        file_name = prompt_files[selected_prompt]
        
        if file_name:
            # Load current content if widget not yet initialized
            # (Streamlit handles persistence via key, but we need initial value)
            current_content = ""
            if os.path.exists(file_name):
                 with open(file_name, "r") as f:
                    current_content = f.read()
            
            # Key for the text area
            ta_key = f"edit_{file_name}"
            
            # Initialize session state if not present to avoid "default value" warning
            if ta_key not in st.session_state:
                st.session_state[ta_key] = current_content
            
            st.text_area("Edit Prompt", height=300, key=ta_key)
            
            col1, col2 = st.columns(2)
            with col1:
                st.button("Save Changes", 
                          key=f"save_{file_name}", 
                          on_click=save_prompt_callback, 
                          args=(file_name, ta_key))
            
            with col2:
                st.button("Reset to Default", 
                          key=f"reset_{file_name}", 
                          on_click=reset_prompt_callback, 
                          args=(file_name, ta_key))

def main():
    # Initialize JSON data state to allow modifications
    if 'current_json_data' not in st.session_state:
        st.session_state.current_json_data = None
        
    # Initialize chat history (optional, but good for UX)
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    # Initialize HTML content
    if 'html_content' not in st.session_state:
        default_html = load_html_template()
        st.session_state.html_content = default_html if default_html else "<div>Error loading template.</div>"

    api_key = os.getenv("GOOGLE_API_KEY")

    # --- CSS Styles for Right Sidebar ---
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                left: unset !important;
                right: 0 !important;
                border-right: none !important;
                border-left: 1px solid #f0f2f6 !important;
            }
            
            [data-testid="stSidebarCollapsedControl"] {
                left: unset !important;
                right: 20px !important; 
            }
            
            .main .block-container {
                max-width: 100%;
                padding-right: 22rem; 
                padding-left: 5rem;
            }

            [data-testid="stSidebar"][aria-expanded="false"] + section .main .block-container {
                 padding-right: 5rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Sidebar (Now Right Fly-out) ---
    with st.sidebar:
        st.header("AI Assistant")
        
        if not api_key:
            st.warning("GOOGLE_API_KEY missing in .env")
        else:
            # Show Billing Info
            try:
                curr_cost, _ = billing.get_todays_cost()
                st.metric("Daily Spend", f"${curr_cost:.4f}", f"Limit: ${billing.DAILY_BUDGET:.2f}")
                if curr_cost >= billing.DAILY_BUDGET:
                    st.error("Budget Limit Reached!")
            except Exception:
                pass
            
        # Graph Type Selector - Bound to Session State for Programmatic Switching
        if "graph_type" not in st.session_state:
            st.session_state.graph_type = "Graph"
        
        # Callback to sync widget -> state
        def update_graph_type():
            st.session_state.graph_type = st.session_state.graph_type_input

        options = ["Graph", "Mindmap", "Sequence", "Timeline"]
        curr_type = st.session_state.graph_type
        try:
            curr_idx = options.index(curr_type)
        except ValueError:
            curr_idx = 0
            
        # Use a separate key for the widget to allow us to modify session_state.graph_type programmatically
        # without conflicting with the "widget instantiated" check.
        st.radio(
            "Structure", 
            options, 
            index=curr_idx,
            horizontal=True, 
            key="graph_type_input",
            on_change=update_graph_type,
            help="Choose 'Graph' for hierarchical flows, 'Mindmap' for radial brainstorming, 'Sequence' for interaction diagrams, or 'Timeline' for chronological events."
        )
        
        # Ensure local variable is set
        graph_type = st.session_state.graph_type
        
        # Chat Interface
        chat_container = st.container(height=700) 
        with chat_container:
            if not st.session_state.chat_messages:
                st.info("👋 Welcome! Type a topic (e.g., 'Solar System') to generate a graph, or ask for changes.")
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        
        # Chat Input
        if prompt := st.chat_input("Ask a question or request a diagram..."):
             # Add user message to state
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            # Show user message immediately
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            if not api_key:
                 st.session_state.chat_messages.append({"role": "assistant", "content": "Error: API Key missing."})
            else:
                # Initialize Assistant
                assistant = GraphAssistant(api_key)
                target_json = st.session_state.current_json_data
                
                with st.spinner("Thinking..."):
                    response_text, action_data = assistant.process_request(
                        prompt, 
                        st.session_state.chat_messages[:-1], 
                        target_json, 
                        graph_type
                    )
                
                if action_data:
                    action = action_data.get("action")
                    instruction = action_data.get("topic_or_instruction")
                    target_type = action_data.get("target_type")
                    
                    # Handle Type Switching
                    if target_type and target_type in ["Graph", "Mindmap", "Sequence", "Timeline"]:
                        if target_type != graph_type:
                            st.session_state.graph_type = target_type
                            graph_type = target_type # Update local var for immediate use
                            st.toast(f"Switching to {target_type} mode...", icon="🔄")
                    
                    if action == "GENERATE":
                        with st.status(f"Generating new {graph_type} for: {instruction}...", expanded=True) as status:
                            new_html, json_data, error = generate_graph(instruction, api_key, graph_type)
                            if new_html:
                                st.session_state.html_content = new_html
                                st.session_state.current_json_data = json_data
                                completion_msg = f"Here is the {graph_type} regarding {instruction}. I can also explain it if you like!"
                                st.session_state.chat_messages.append({"role": "assistant", "content": completion_msg})
                                status.update(label="Graph Generated!", state="complete", expanded=False)
                                st.rerun()
                            else:
                                error_msg = f"Generation Error: {error}"
                                st.error(error_msg)
                                st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
                                status.update(label="Generation Failed", state="error")

                    elif action == "MODIFY":
                        with st.status(f"Modifying graph: {instruction}...", expanded=True) as status:
                            new_html, json_data, error = modify_graph(target_json, instruction, api_key, graph_type)
                            if new_html:
                                st.session_state.html_content = new_html
                                st.session_state.current_json_data = json_data
                                completion_msg = f"Updated the graph: {instruction}."
                                st.session_state.chat_messages.append({"role": "assistant", "content": completion_msg})
                                status.update(label="Graph Updated!", state="complete", expanded=False)
                                st.rerun()
                            else:
                                error_msg = f"Modification Error: {error}"
                                st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
                                status.update(label="Modification Failed", state="error")
                
                elif response_text:
                    # Just conversational
                    st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
                    st.rerun()

            st.rerun()

        # Add Prompt Settings at the bottom of sidebar
        render_prompt_settings()

    # --- Main Area ---
    st.title("Interactive Graph Generator")
    
    if st.session_state.html_content:
        # Create layout based on user preference
        show_editor = st.toggle("Show Live Code Editor", value=True)
        
        if show_editor:
            col1, col2 = st.columns([1, 2]) 
        else:
            col1 = None # Editor hidden
            col2 = st.container() # Full width for preview

        if show_editor and col1:
            with col1:
                render_live_editor(graph_type, api_key)

        with col2:
            st.subheader("Preview")
            st.components.v1.html(st.session_state.html_content, height=1200, scrolling=True)
            st.download_button("Download HTML", st.session_state.html_content, "graph.html", "text/html")



if __name__ == "__main__":
    main()