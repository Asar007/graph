import streamlit as st
import google.generativeai as genai
import json
import os
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

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
            response = model.generate_content(final_prompt)
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

def modify_graph(current_json, prompt, api_key):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    
    template = load_modification_prompt()
    if not template:
        return None, None, "Modification prompt file missing."
        
    # Replace placeholders in the text file
    final_prompt = template.replace("[INSERT CURRENT JSON DATA HERE]", json.dumps(current_json))
    final_prompt = final_prompt.replace("[INSERT CHANGE REQUEST HERE]", prompt)
    
    try:
        messages = [
            HumanMessage(content=final_prompt)
        ]
        
        response = llm.invoke(messages)
        content = response.content
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*$', '', content)
        
        new_json_data = json.loads(content)
        
        html_template = load_html_template()
        if html_template:
            new_html = inject_data_into_html(html_template, new_json_data)
            return new_html, new_json_data, None
            
    except Exception as e:
        return None, None, str(e)
        
    return None, None, "Unknown error during modification."

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
            
            /* Fix the expand/collapse button position */
            [data-testid="stSidebarCollapsedControl"] {
                left: unset !important;
                right: 20px !important; 
            }
            
            /* Main Content adjustments */
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
            
        # Graph Type Selector
        graph_type = st.radio("Structure", ["Graph", "Mindmap", "Sequence", "Timeline"], horizontal=True, help="Choose 'Graph' for hierarchical flows, 'Mindmap' for radial brainstorming, 'Sequence' for interaction diagrams, or 'Timeline' for chronological events.")
        
        # Chat Interface
        chat_container = st.container(height=700) 
        with chat_container:
            if not st.session_state.chat_messages:
                st.info("👋 Welcome! Type a topic (e.g., 'Solar System') to generate a graph, or ask for changes.")
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        
        # Chat Input
        if prompt := st.chat_input("Type a topic or modification..."):
             # Add user message to state
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            # Show user message immediately
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            if not api_key:
                 st.session_state.chat_messages.append({"role": "assistant", "content": "Error: API Key missing."})
            else:
                target_json = st.session_state.current_json_data
                
                if target_json is None:
                    # Generate New
                    with st.spinner(f"Generating '{prompt}' ({graph_type})..."):
                        new_html, json_data, error = generate_graph(prompt, api_key, graph_type)
                        if new_html:
                            st.session_state.html_content = new_html
                            st.session_state.current_json_data = json_data
                            st.session_state.chat_messages.append({"role": "assistant", "content": f"Generated graph for: {prompt}"})
                        else:
                            error_msg = f"Error: {error}"
                            st.error(error_msg) # Show globally
                            st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
                else:
                    # Modify Existing
                    with st.spinner("Modifying..."):
                        new_html, json_data, error = modify_graph(target_json, prompt, api_key)
                        if new_html:
                            st.session_state.html_content = new_html
                            st.session_state.current_json_data = json_data # Update state
                            st.session_state.chat_messages.append({"role": "assistant", "content": "Graph updated successfully!"})
                        else:
                            st.session_state.chat_messages.append({"role": "assistant", "content": f"Error: {error}"})
            st.rerun()

    # --- Main Area ---
    st.title("Interactive Graph Generator")
    
    # Render Graph
    # Render Graph
    if st.session_state.html_content:
        # Increased height from 850 to 1200 for a bigger view
        st.components.v1.html(st.session_state.html_content, height=1200, scrolling=True)
        
        # Download button below graph
        st.download_button("Download HTML", st.session_state.html_content, "graph.html", "text/html")


if __name__ == "__main__":
    main()
