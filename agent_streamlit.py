# agent_streamlit.py
import time
start_time = time.time()
import streamlit as st
import os
import random
from datetime import datetime
from pyvis.network import Network
import streamlit.components.v1 as components

# Import your DesAgent from DesAgent.py
from DesAgent import DesAgent

import re
from typing import List, Dict, Any

def build_graph_html(processed_records: List[Dict[str, List[Dict[str, Any]]]]) -> str:
    """
    Builds an interactive graph using PyVis based on the processed Neo4j query results.
    This version accepts a list of records, where each record is a dictionary
    (e.g., [{'p1': [...], 'p2': [...]}, {...}, ... ]).

    Args:
        processed_records (List[Dict[str, List[Dict[str, Any]]]]):
            A list of dictionaries, each containing path variables (e.g., 'p1', 'p2') mapped
            to their respective sequences of nodes and relationships.

    Returns:
        str: HTML representation of the interactive graph that combines all records.
    """
    # Initialize a PyVis Network
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        filter_menu=False,
        select_menu=False,
        neighborhood_highlight=False
    )

    group_label_map = {
        "Mixture":   {"group": 1, "color": "#FFB3A7"},
        "Substance": {"group": 2, "color": "#B3FFB3"},
        "Article":   {"group": 3, "color": "#B3FFB3"},
        "Unknown":   {"group": 4, "color": "#E0E0E0"}
    }

    added_nodes = set()
    added_edges = set()
    node_id_map = {}

    # ------------------ PASS 1: ADD ALL NODES ------------------
    for record in processed_records:
        for path_var, elements in record.items():
            for element in elements:
                if 'n' in element:
                    node = element['n']
                    label = node.get('label', 'Unknown')
                    properties = node.get('properties', {})

                    # Determine a unique, stable signature for the node
                    # We'll combine the label + key property to form a key in node_id_map.
                    if label == 'Mixture':
                        key_property = properties.get('mixtureID', f"Mixture_{len(added_nodes)+1}")
                    elif label == 'Substance':
                        key_property = properties.get('pubChemName', f"Substance_{len(added_nodes)+1}")
                    else:
                        # fallback if no recognized label
                        key_property = f"Unknown_{len(added_nodes)+1}"

                    # This signature helps us consistently retrieve the same node ID
                    node_signature = (label, key_property)

                    if node_signature not in node_id_map:
                        # Create a tooltip with properties
                        tooltip = "<br>".join([f"{k}: {v}" for k, v in properties.items()])
                        # Choose a color
                        # color = label_color_map.get(label, label_color_map['Unknown'])

                        # Add the node to the PyVis network
                        net.add_node(
                            key_property,    # node ID
                            label=key_property,
                            title=tooltip,
                            # group=group_label_map.get(label, group_label_map['Unknown'])['group'],
                            color=group_label_map.get(label, group_label_map['Unknown'])['color']
                        )

                        # Mark it as added and remember the signature -> ID mapping
                        node_id_map[node_signature] = key_property
                        added_nodes.add(key_property)

    # ------------------ PASS 2: ADD ALL RELATIONSHIPS ------------------
    for record in processed_records:
        for path_var, elements in record.items():
            for idx, element in enumerate(elements):
                if 'r' in element:
                    relationship = element['r']
                    rel_type = relationship.get('type', 'UNKNOWN_RELATIONSHIP')

                    # We assume the order is: [node, relationship, node, relationship, node, ...]
                    # So if we're at index idx with a relationship, we look at the node before and the node after.
                    if 0 < idx < len(elements) - 1:
                        prev_element = elements[idx - 1]
                        next_element = elements[idx + 1]

                        if 'n' in prev_element and 'n' in next_element:
                            # Extract the label and key property for the source node
                            prev_label = prev_element['n'].get('label', 'Unknown')
                            prev_props = prev_element['n'].get('properties', {})
                            if prev_label == 'Mixture':
                                prev_key = prev_props.get('mixtureID', 'UnknownMixture')
                            elif prev_label == 'Substance':
                                prev_key = prev_props.get('pubChemName', 'UnknownSubstance')
                            else:
                                prev_key = f"Unknown_{idx-1}"

                            # Extract the label and key property for the target node
                            next_label = next_element['n'].get('label', 'Unknown')
                            next_props = next_element['n'].get('properties', {})
                            if next_label == 'Mixture':
                                next_key = next_props.get('mixtureID', 'UnknownMixture')
                            elif next_label == 'Substance':
                                next_key = next_props.get('pubChemName', 'UnknownSubstance')
                            else:
                                next_key = f"Unknown_{idx+1}"

                            edge_id = (prev_key, next_key, rel_type)

                            if edge_id not in added_edges:
                                net.add_edge(prev_key, next_key, title=rel_type) # dont show relationship type
                                # net.add_edge(prev_key, next_key, label=rel_type, title=rel_type) # show relationship type
                                added_edges.add(edge_id)

    # Enable physics for a better layout (optional)
    net.toggle_physics(True)
    net.show_buttons(filter_=['physics'])
    net.set_edge_smooth('dynamic')

    # Generate and return the HTML for embedding
    return net.generate_html()


def main():
    st.title("DESAgent Chat + PyVis Graph Demo")

    # Make sure logs folder exists
    os.makedirs("logs", exist_ok=True)

    # --- 1. Handle Session State for session_id, DESAgent, and Messages. ---
    if "session_id" not in st.session_state:
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        random_number = random.randint(1000, 9999)
        st.session_state.session_id = f"{current_time}_{random_number}"

    if "agent" not in st.session_state:
        st.session_state.agent = DesAgent(session_id=st.session_state.session_id)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize graph visibility state
    if "graph_visibility" not in st.session_state:
        st.session_state.graph_visibility = {}

    # 1. Initialize the user manual visibility in session_state if not set
    if "show_user_manual" not in st.session_state:
        st.session_state.show_user_manual = False

    # 2. Create two buttons for toggling user manual visibility
    col_show, col_hide = st.columns(2)
    with col_show:
        if st.button("Show Manual"):
            st.session_state.show_user_manual = True
    with col_hide:
        if st.button("Hide Manual"):
            st.session_state.show_user_manual = False

    # 3. Conditionally render the user manual
    if st.session_state.show_user_manual:
        # Load user manual content from an external Markdown file
        manual_path = r'D:\Users\YISHEN335\Documents\VScode\DES_Agent\docs\des_agent_doc.md'
        if os.path.exists(manual_path):
            with open(manual_path, "r", encoding="utf-8") as f:
                manual_content = f.read()
            st.markdown(manual_content)
        else:
            st.error("User manual file not found.")

    end_time = time.time()
    print(f"Time taken for initialization: {end_time - start_time:.2f}s")

    # --- 2. Render existing chat messages from session_state. ---
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "graph" in message and message["graph"]:
                # Create two columns for Show and Hide Graph buttons
                col1, col2 = st.columns(2)

                with col1:
                    # Show Graph Button
                    show_button_key = f"show_graph_{idx}"
                    if st.button("Show Graph", key=show_button_key):
                        st.session_state.graph_visibility[idx] = True

                with col2:
                    # Hide Graph Button
                    hide_button_key = f"hide_graph_{idx}"
                    if st.button("Hide Graph", key=hide_button_key):
                        st.session_state.graph_visibility[idx] = False

                # Display the graph if the visibility is set to True
                if st.session_state.graph_visibility.get(idx, False):
                    html_content = build_graph_html(message["graph"])
                    components.html(html_content, height=600, width=800, scrolling=True)
            
            else:
                # add a note saying that the graph is not available
                st.markdown("No graph available for this message.")

    # --- 3. Chat input box at the bottom for the user to ask a question. ---
    user_input = st.chat_input("Ask the DESAgent something...")
    if user_input:
        # (a) Log the user's message to the session
        st.session_state.messages.append({"role": "user", "content": user_input})

        # --- IMMEDIATE LOGGING: Save user input to file ---
        user_log_path = f"logs/{st.session_state.session_id}.log"
        os.makedirs("logs", exist_ok=True)
        with open(user_log_path, "a", encoding="utf-8") as f:
            f.write(f"[USER] {user_input}\n")

        # Show it in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # (b) Stream the assistant's response
        response_chunks = []
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            for chunk in st.session_state.agent.task_execution(user_input):
                response_chunks.append(chunk)
                combined_text = "".join(response_chunks)
                message_placeholder.markdown(combined_text)

        # (c) Finalize the assistant response
        final_response = "".join(response_chunks)
        processed_result = st.session_state.agent.get_latest_processed_result()

        # --- IMMEDIATE LOGGING: Save assistant response to file ---
        with open(user_log_path, "a", encoding="utf-8") as f:
            f.write(f"[ASSISTANT] \n{final_response}\n\n")

        # (d) Store assistant message + optional graph
        if processed_result:
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_response,
                "graph": processed_result
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_response,
            })

        # (e) Render “Show Graph / Hide Graph” if needed, for the newly generated message
        if st.session_state.messages[-1]["role"] == "assistant" and "graph" in st.session_state.messages[-1]:
            idx = len(st.session_state.messages) - 1
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Show Graph", key=f"show_graph_{idx}"):
                    st.session_state.graph_visibility[idx] = True
            with col2:
                if st.button("Hide Graph", key=f"hide_graph_{idx}"):
                    st.session_state.graph_visibility[idx] = False

            if st.session_state.graph_visibility.get(idx, False):
                html_content = build_graph_html(processed_result)
                components.html(html_content, height=600, width=800, scrolling=True)
        else:
            st.markdown("No graph available for this message.")

        # Optionally, you can reset the processed_results in the agent
        # to avoid reusing the same graph for multiple messages
        # Here, we keep all processed_results for potential future use

    # Note: The graph visualization is handled in the messages loop above.
    

if __name__ == "__main__":
    main()
