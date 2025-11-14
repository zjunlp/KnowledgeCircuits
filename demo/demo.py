import langid
import streamlit as st
from streamlit_react_flow import react_flow
import requests
import re
import os

# Set up language identification
langid.set_languages(['en', 'zh'])
lang_dic = {'zh': 'en', 'en': 'zh'}

color_dic = {
    'Hallucination': '#ff0000',
    'In-Context Learning': '#0000ff'
}

def set_page_config():
    """
    Configure the Streamlit page settings.
    """
    st.set_page_config(
        page_title='Knowledge Circuits in Pretrained Transformers',
        page_icon="🦈",
        layout='wide',
        initial_sidebar_state='expanded',
        menu_items={
            'Report a bug': 'https://github.com/zjunlp/KnowledgeCircuits/issues',
            'About': '## This work aims to build the circuits in the pretrained language models that are responsible for the specific knowledge and analyze the behavior of these components.'
        }
    )

def get_gv_files(circuit_dir):
    """
    Retrieve all .gv files from the specified directory.
    """
    gv_files = []
    for subdir in os.listdir(circuit_dir):
        subdir_path = os.path.join(circuit_dir, subdir)
        if os.path.isdir(subdir_path):
            gv_file = os.path.join(subdir_path, 'graph.gv')
            if os.path.exists(gv_file):
                gv_files.append((subdir, gv_file))
    return sorted(gv_files, key=lambda x: x[0])

def gv_to_edges(gv_file):
    """
    Parse a .gv file and extract edges.
    """
    edges = []
    edge_pattern = re.compile(r'\"<(.+?)>\" -> \"<(.+?)>\"')
    lines = open(gv_file, 'r').readlines() if isinstance(gv_file, str) else gv_file.getvalue().decode().split('\n')
    for line in lines:
        match = edge_pattern.search(line)
        if match:
            edges.append((match.group(1), match.group(2)))
    return edges

def get_layer(node):
    """
    Determine the layer of a node based on its name.
    """
    if node.startswith('m'):
        return int(node[1:].split('-')[0])
    elif node.startswith('a'):
        return int(node[1:].split('.')[0])
        
    return 100

def get_attention_head(node):
    return int(node.split('.')[-1]) if node.startswith('a') else -1

def find_reachable_nodes(edges, start_nodes, direction='out'):
    reachable, to_visit = set(start_nodes), list(start_nodes)
    while to_visit:
        current = to_visit.pop()
        for source, target in edges:
            if direction == 'out' and source == current and target not in reachable:
                reachable.add(target)
                to_visit.append(target)
            elif direction == 'in' and target == current and source not in reachable:
                reachable.add(source)
                to_visit.append(source)
    return reachable

def create_elements(edges, graph_width, graph_height, special_nodes, special_node_color):
    """
    Create elements for the React Flow graph based on the edges.
    """
    nodes = set()
    for source, target in edges:
        nodes.add(source)
        nodes.add(target)

    sorted_nodes = sorted(nodes, key=lambda x: (get_layer(x), get_attention_head(x) if x.startswith('a') else -1), reverse=True)

    min_layer = min(get_layer(node) for node in nodes if get_layer(node) != float('-inf'))
    max_layer = max(get_layer(node) for node in nodes if get_layer(node) != float('-inf'))

    layer_counts = [0] * (max_layer - min_layer + 1)
    layer_heights = [0] * (max_layer - min_layer + 1)

    for node in sorted_nodes:
        layer = get_layer(node)
        if layer != 100:
            layer_counts[layer - min_layer] += 1
            layer_heights[layer - min_layer] = max(layer_heights[layer - min_layer], 1)

    total_layers, layer_height = sum(layer_heights), graph_height / sum(layer_heights)
    elements = []
    node_positions = {}
    attention_counts = [0] * (max_layer - min_layer + 1)

    for node in sorted_nodes:
        layer = get_layer(node)
        if layer == 100:
            x = graph_width * 0.85
            y = 20
        else:
            if node.startswith('m'):
                x = graph_width * 0.05 
            else:
                attention_counts[layer - min_layer] += 1
                x = graph_width * 0.85 - attention_counts[layer - min_layer] * (graph_width * 0.8) / (layer_counts[layer - min_layer] + 1)  # Moved left from 0.9 to 0.85
            y = (total_layers - sum(layer_heights[:layer - min_layer]) - 0.5) * layer_height

        if node.startswith('m'):
            node_type = 'MLP'
        elif node.startswith('a'):
            node_type = 'Attention'
            # Ensure nodes in the different layer don't have same x position
            x =  (x + (layer - min_layer)*10) if ((layer - min_layer) % 2) == 1 else (x - (layer - min_layer)*10)
        elif node == 'resid_post':
            node_type = 'Output'
        else:
            node_type = 'default'
        
        node_positions[node] = {'x': x, 'y': y}
        node_style = node_types.get(node_type, {}).get('style', {})
        if node in special_nodes:
            special_node_style = node_style.copy()
            # 红色
            special_node_style['background'] = '#ff0000'
            elements.append({
                'id': node,
                'data': {'label': node},
                'type': node_type,
                'position': {'x': x, 'y': y},
                'style': special_node_style
            })
        else:
            elements.append({
                'id': node,
                'data': {'label': node},
                'type': node_type,
                'position': {'x': x, 'y': y},
                'style': node_style
            })

        if 'H' in node:
            elements[-1]['position']['y'] -= layer_height / 4

    reachable_from_special = find_reachable_nodes(edges, special_nodes, direction='out') if len(special_nodes) > 1 else set()
    reachable_to_special = find_reachable_nodes(edges, special_nodes, direction='in') if len(special_nodes) > 1 else set()

    for source, target in edges:
        edge_style = {'stroke': '#888', 'strokeWidth': 2, 'strokeDasharray': '5, 5'}
        if source in special_nodes or target in special_nodes or source in reachable_from_special or target in reachable_to_special:
            edge_style['stroke'] = special_node_color

        elements.append({
            'id': f'{source}-{target}',
            'source': source,
            'target': target,
            'type': 'smoothstep',
            'animated': True,
            'style': edge_style,
        })

    return elements

node_types = {
    'Input': {'style': {'background': '#f472b6', 'width': 50, 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}},
    'Embedding': {'style': {'background': '#4ea8de', 'width': 50, 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}},
    #浅的紫色
    'Attention': {'style': {'background': '#e6e6fa', 'width': 50, 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}},
    'MLP': {'style': {'background': '#6ede87', 'width': 50, 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}},
    'Output': {'style': {'background': '#ffcc50', 'width': 80, 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}},
}

def main():
    """
    Main function to run the our app.
    """
    # Read and update visit count
    visit_count_file = 'visit_count.txt'
    if os.path.exists(visit_count_file):
        with open(visit_count_file, 'r') as file:
            visit_count = int(file.read().strip())
        visit_count += 1
        with open(visit_count_file, 'w') as file:
            file.write(str(visit_count))
    else:
        visit_count = 4093

    set_page_config()
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {
                width: 450px !important; # Set the width to your desired value
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.title("Knowledge Circuits in Pretrained Transformers")
    # github & arXiv
    st.sidebar.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/zjunlp/KnowledgeCircuits) &nbsp; [![arXiv](https://img.shields.io/badge/arXiv-2405.17969-b31b1b.svg)](https://arxiv.org/abs/2405.17969)")

    # Choose analysis method
    analysis_method = st.sidebar.radio(
        "Choose analysis method",
        ("Select from existing cases", "Upload your own file")
    )

    if analysis_method == "Select from existing cases":
        selected_formatted_name = st.sidebar.radio('Select LLM Model', ['gpt2-medium'])
        circuit_dir, gv_files = selected_formatted_name, get_gv_files(selected_formatted_name)
        case = st.sidebar.selectbox('Select Case', [name for name, _ in gv_files])
        gv_file_path = next((file for name, file in gv_files if name == case), "")

        # input and output
        with open(f'{circuit_dir}/{case}/io.txt', 'r') as file:
            io_content = file.readlines()
        
        st.sidebar.write(f'**Input:** {io_content[0]}')
        st.sidebar.write(f'**Output:** {io_content[1]}') 

        # Read special nodes
        special_nodes_path = f'{circuit_dir}/{case}/specialNode.txt'
        special_nodes = [line.strip() for line in open(special_nodes_path, 'r').readlines()] if os.path.exists(special_nodes_path) else []
        special_node_color = color_dic['Hallucination']
    else:
        uploaded_file = st.sidebar.file_uploader("Upload your .gv file", type="gv")
        if uploaded_file:
            gv_file_path, selected_formatted_name, special_nodes, special_node_color = uploaded_file, "Custom Analysis", [], color_dic['Hallucination']
        else:
            st.sidebar.warning("Please upload a .gv file to proceed with the analysis.")
            return

    # attention analysis graph - 'attention_analysis.png'
    attention_analysis_path = f'{circuit_dir}/{case}/attention_analysis.png'
    if case == 'In-Context Learning' :
        att = 'a12.13'
    elif case == 'Hallucination' :
        att = 'a15.0'
    if os.path.exists(attention_analysis_path):
        st.sidebar.image(attention_analysis_path, caption=f'Attention Analysis: {att}')

    if not gv_file_path:
        st.error(f"No graph file found for {selected_formatted_name}")
        return
    
    # m12代表第12层，a12.13代表第12层的第13个注意力头
    st.sidebar.markdown('🟢 mx: the x-th MLP layer')
    st.sidebar.markdown('🟣 ax.y: the x-th layer and y-th attention head')
    st.sidebar.markdown('🔴 Special_node: related to Hallucination or ICL')  


    st.title(f'`{selected_formatted_name}` Knowledge Circuit')
    edges = gv_to_edges(gv_file_path)

    # Set graph dimensions
    # graph_width = 1500 - 100
    # graph_height = 1000 - 200
    graph_width = 1200  # Reduced from 1500 - 100
    graph_height = 800  # Reduced from 1000 - 200

    elements = create_elements(edges, graph_width, graph_height, special_nodes, special_node_color)

    flowStyles = {'height': f'{graph_height}px', 'width': f'{graph_width}px'}

    # Render the React Flow graph
    react_flow('transformer', elements=elements, flow_styles=flowStyles)
        
    # Add Footer
    footer = st.container()
    with footer:
        st.markdown("---")
        st.markdown(f"<div style='text-align: center;'>Visits: {visit_count}</div>", unsafe_allow_html=True)
        # zjukg 设置超链接 https://www.zjukg.org/
        st.markdown("<div style='text-align: center;'>Copyright ©2024 <a href='https://www.zjukg.org/'>ZJUKG</a> All Rights Reserved.</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
