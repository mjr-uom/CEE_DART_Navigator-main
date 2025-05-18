import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import gravis as gv
import altair as alt
import networkx as nx
import matplotlib as mt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import os
import hashlib
import importlib
import sys
import matplotlib.pyplot as plt
from len_gen import generate_legend_table, generate_legend_table_community

# Add source directory to sys.path if not already present
path_to_functions_directory = r'./source'
if path_to_functions_directory not in sys.path:
    sys.path.append(path_to_functions_directory)

import civic_data_code as civic
importlib.reload(civic)

import dataloaders as dtl
importlib.reload(dtl)

import functions_graphs as fg
importlib.reload(fg)

import LRPGraph_code as lrpgraph
importlib.reload(lrpgraph)

import LRP_comparison_code as lrpcomp

import pharmabk_code as pbk
importlib.reload(pbk)
import collections

import altair as alt



default_session_state = {
    
    'civic_data': pd.DataFrame([]),
    'page':'Home',
    'lrp_df': pd.DataFrame([]), # raw loaded data
    'tumor_tissue_site': list(),
    'ttss_selected': list(),
    'acronym': list(),    
    'filtered_tts_lrp_df': pd.DataFrame([]),
    'metadata_df': pd.DataFrame([]), # raw loaded data
    'filtered_lrp_df': pd.DataFrame([]),  # Initialize as an empty DataFrame
    'filtered_data_to_graphs': pd.DataFrame([]),
    'f_tumor_tissue_site': pd.DataFrame([]),
    'f_acronym': pd.DataFrame([]),
    'keywords': list(),
    'frequent_kws': pd.DataFrame([]), # raw loaded data
    'tts_filter_button': None,
    #'first_form_completed': False,
    'sample_names': list(),
    'sample1': None,
    'sample2': None,
    'sample2_options': list(),
    'top_n': 150,
    'stratify_by': None,
    'new_stratify_by': None,
    'group1': None,
    'group2': None,
    'sg_grouping_column': None,
    'sg_group_options': list(),
    'gg_grouping_column': None,
    'gg_group_options': list(),
    'group_for_agg': None,
    'gene_enrichment': None,
    'gene_list': list(),
    'gene_list_no_types': list(),
    'pharmGKB_details': {},
    'gene_enrichment': None,     
    'community_enrichment': None,
    'civic_evidence': None,      
    'pharmGKB_analysis': None,
    'ai_context':str(),
    'ai_prompt':str(),
    'community_analysis': dict(),

    'Filter_data_button': False, # 11
    'analysis_type': None,
    'Type_of_analysis_selected': False,
    'comparison_type': None,
    'Type_of_comparison_selected': False,
    'Generate_graphs_button': False, # plot 2 graphs
    'G1_G2_displayed': False,
    'G_dict': {},
    'diff_thres': 0.5,
    'p_value': 0.05,
    'Calculate_button': False, # plot DIFF graph
    'diff_Graphs_displayed': False,
    #'Analyse_evidence_button': False,
    'Evidence_analysis_done': False,
    'AI_assistance_button': False,
    
}

for key, default_value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

def reset_session_state_until(key_to_reset ):
    """
    Resets the session state of Streamlit variables until a specified key.
    This function is useful for resetting the state of the application to a specific point.
    """
    keys = list(default_session_state.keys())
    try:
        idx = keys.index(key_to_reset)
    except ValueError:
        idx = -1
    # Only reset keys from 'key_to_reset'
    for key in keys[idx:]:
        st.session_state[key] = default_session_state[key]
    print_session_state()




#sidebar_logo_DCR = "./images/CRUK_NBC_DCR.png"
#sidebar_logo = "./images/CCE_Dart_logo.png"
main_body_logo   = "./images/CCE_Dart_icon.png"

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"


st.set_page_config(page_title = "LRP Dashboard", page_icon = main_body_logo, layout = "wide")

def print_session_state():
    """
    Prints the current state of the Streamlit session variables in a structured format.
    This function is useful for debugging and tracking the state of the application.
    """
    session_state_keys = [
        'Filter_data_button',
        'analysis_type',
        'Type_of_analysis_selected',
        'comparison_type',
        'Type_of_comparison_selected',
        'Generate_graphs_button',
        'G1_G2_displayed',
        'diff_thres',
        'p_value',
        'Calculate_button',
        'diff_Graphs_displayed',
        'Analyse_evidence_button',
        'Evidence_analysis_done',
        'AI_assistance_button',
        'top_n',
        'group_for_agg',
        'sample1',
        'sample2',
        'group1',
        'group2',
        'sg_grouping_column',
        'sg_group_options',
        'gg_grouping_column',
        'gg_group_options'
    ]

    session_state = {key: st.session_state.get(key, None) for key in session_state_keys}
    print('\nSession state: \n', session_state)
    

def assign_colors(strings):
    color_names = sorted(list(mcolors.CSS4_COLORS.keys()))

    def get_color(string):
        hash_value = int(hashlib.sha256(string.encode()).hexdigest(), 16)
        return color_names[hash_value % len(color_names)]

    keys = {string.split("_")[-1] for string in strings}  # Set of unique keys
    sorted_keys = sorted(keys)  # Sort the keys for deterministic order

    color_dict = {key: get_color(key) for key in sorted_keys}
    return color_dict

def assign_colors_int(ids):
    # Get a sorted list of color names
    color_names = sorted(list(mcolors.CSS4_COLORS.keys()))

    def get_color(id):
        # Convert the integer id to a string before hashing
        hash_value = int(hashlib.sha256(str(id).encode()).hexdigest(), 16)
        # Map the hash value to a color
        return color_names[hash_value % len(color_names)]

    # Create a dictionary mapping each id to a color
    color_dict = {id: get_color(id) for id in ids}
    return color_dict


import tempfile

def save_my_uploaded_file(path, uploaded_file):
    # Użyj katalogu tymczasowego odpowiedniego dla systemu operacyjnego
    repository_folder = tempfile.gettempdir() if path == '/tmp' else path
    save_path = os.path.join(repository_folder, uploaded_file.name)
    with open(save_path, mode='wb') as w:
        w.write(uploaded_file.getvalue())
    return save_path

#def save_my_uploaded_file(path, uploaded_file):
#    repository_folder = path
#    save_path = os.path.join(repository_folder, uploaded_file.name)
#    with open(save_path, mode='wb') as w:
#        w.write(uploaded_file.getvalue())
#    return save_path

def find_my_keywords(lrp_df):
    option_tmp = []
    for opr_element in lrp_df.columns:
        tmp_ele = "_".join(opr_element.split("-")[-1].strip().split("_")[:-1])
        option_tmp.append(tmp_ele)

    keywords = list(dict.fromkeys(option_tmp))
    keywords.sort()
    return keywords

def get_adj_list(G):
    adj_list = {node: set() for node in G.nodes}  # Initialize adjacency list
    for u, v in G.edges:
        adj_list[u].add(v)
        adj_list[v].add(u)
    return adj_list

def plot_my_graph(container, graph, communities=None):
    # Helper function: Assign community-related colors

    
    def prepare_communities_data(communities):
        if communities is not None and not communities.empty:
            if "node" in communities.columns and "community" in communities.columns:
                node_community_mapping = dict(zip(communities["node"], communities["community"]))
                group_list = sorted(communities["community"].unique())
                group_color_mapper = assign_colors_int(group_list)
                return node_community_mapping, group_color_mapper
        return None, None

    # Helper function: Format edge attributes
    # def format_edge_data(newG):
    #     edge_dist = list(nx.get_edge_attributes(newG, 'LRP_norm').values())
    #     norm_colour = mt.colors.Normalize(vmin=0.0, vmax=max(edge_dist), clip=True)
    #     colour_mapper = cm.ScalarMappable(norm=norm_colour, cmap=cm.Greys)
    #     for u, v, data in newG.edges(data=True):
    #         lrp_norm = data.get("LRP_norm", 0)
    #         data.update({
    #             "hover": f'LRP_norm: {lrp_norm:.4f}',
    #             "color": mt.colors.rgb2hex(colour_mapper.to_rgba(lrp_norm)),
    #             "weight": lrp_norm,
    #             "value": lrp_norm,
    #             "click": f'LRP_norm: {lrp_norm:.4f}'
    #         })
    def format_edge_data(newG):
        # Step 1: Identify all unique edge types in the graph
        edge_types = set()
        for u, v in newG.edges():
            u_type = u.split('_')[-1]
            v_type = v.split('_')[-1]
            # Always store as sorted tuple for symmetry (rna, prot) == (prot, rna)
            edge_types.add(tuple(sorted([u_type, v_type])))

        # Step 2: Assign a unique color to each edge type using matplotlib colormap
        if len(edge_types) > 9:
            color_map = cm.get_cmap('tab20', len(edge_types))
        else:
            color_map = cm.get_cmap('Set1', len(edge_types))
        edge_type_to_color = {}
        for idx, edge_type in enumerate(sorted(edge_types)):
            rgb = color_map(idx)[:3]  # get RGB, ignore alpha
            edge_type_to_color[edge_type] = mt.colors.rgb2hex(rgb)

        # Step 3: Fallback color for edges with missing/unknown types
        default_color = '#888888'

        # Step 4: Normalize edge weights for other visual attributes
        edge_dist = list(nx.get_edge_attributes(newG, 'LRP_norm').values())
        if edge_dist:
            norm_colour = mt.colors.Normalize(vmin=0.0, vmax=max(edge_dist), clip=True)
            colour_mapper = cm.ScalarMappable(norm=norm_colour, cmap=cm.Greys)
        else:
            norm_colour = None
            colour_mapper = None

        # Step 5: Assign color and attributes to each edge
        for u, v, data in newG.edges(data=True):
            lrp_norm = data.get("LRP_norm", 0)
            u_type = u.split('_')[-1]
            v_type = v.split('_')[-1]
            edge_type = tuple(sorted([u_type, v_type]))
            color = edge_type_to_color.get(edge_type, default_color)
            # Optionally, fallback to grayscale if no edge types found
            if color == default_color and colour_mapper:
                color = mt.colors.rgb2hex(colour_mapper.to_rgba(lrp_norm))
            data.update({
                "hover": f'LRP_norm: {lrp_norm:.4f} | Type: {u_type}-{v_type}',
                "color": color,
                "weight": lrp_norm,
                "value": lrp_norm,
                "click": f'LRP_norm: {lrp_norm:.4f} | Type: {u_type}-{v_type}'
            })

    # Helper function: Format node attributes
    def format_node_data(newG, pos, neighbor_map, node_community_mapping, group_color_mapper, node_color_mapper):
        for node, data in newG.nodes(data=True):
            #x, y = pos[node]
            label = "_".join(node.split('_')[:-1])
            node_type = node.split('_')[-1]
            neighbors = neighbor_map[node]

            # Update node attributes
            data.update({
                #"x": x, "y": y,
                "label": label, "_type": node_type,
                "click": f"Node: {label} \nType: {node_type} \nNeighbors: \n" + "\n".join(["_".join(n.split('_')[:-1]) for n in neighbors]),
                "hover": f"{label} Neighbors: " + "\n".join(["_".join(n.split('_')[:-1]) for n in neighbors]),
                "value": len(neighbors),
                "color": (group_color_mapper[node_community_mapping[node]]
                          if node_community_mapping else node_color_mapper[node_type])
            })

    # Prepare community data
    node_community_mapping, group_color_mapper = prepare_communities_data(communities)

    # Generate graph and layout
    newG = graph.G
    node_color_mapper = assign_colors(sorted([node for node in newG.nodes]))
    neighbor_map = get_adj_list(newG)
    # if there are communities, the graph should be released so the communities are seprared if not connected
    if communities is not None:
        pos = nx.spring_layout(newG, scale = 500)
    else:
        pos = nx.spring_layout(newG, weight="LRP_norm", iterations=50,  scale = 500)
        #pos = nx.kamada_kawai_layout(newG, weight="LRP_norm", scale=500)

    

    # Format graph data
    format_edge_data(newG)
    format_node_data(newG, pos, neighbor_map, node_community_mapping, group_color_mapper, node_color_mapper)

    # Generate legend
    chart_legend_css = (generate_legend_table_community(group_color_mapper)
                        if communities is not None else generate_legend_table(node_color_mapper))

    # Styling
    css = """
    <style>
        body {
            margin: 0; padding: 10px; border: 1px solid white;
            font-family: Arial, sans-serif;
        }
    </style>
    """
    style_heading = 'text-align: center; font-size: 0.9em;'
    communities_label = " / Communities" if communities is not None else ""

    # Render in Streamlit container
    container.markdown(css, unsafe_allow_html=True)
    container.markdown(f"<h1 style='{style_heading}'> '{graph.sample_ID}' {communities_label}</h1>", unsafe_allow_html=True)
    #container.markdown(f"<h2 style='{style_heading}'>Top '{len(newG.edges)}' edges with the highest LRP values</h2>", unsafe_allow_html=True)
    
    # Create and display graph visualization
    fig = gv.d3(
        newG,
        use_node_size_normalization=True,
         node_size_normalization_min = 2,
          node_size_normalization_max = 10,
            node_size_data_source='value', node_hover_tooltip=True,
        node_hover_neighborhood=True, show_node_label=True, node_label_data_source='label',
        node_label_size_factor = .5,

        use_edge_size_normalization=True,
        edge_size_normalization_min =.5,
        edge_size_normalization_max = 5,
        edge_size_data_source='weight', 
        edge_hover_tooltip=True,# zoom_factor=0.55,
        edge_curvature=0.2,
        layout_algorithm_active=True,
        many_body_force_strength=100.0,
        use_many_body_force_max_distance=True,
        many_body_force_max_distance=50.0,
        use_collision_force=True,
        collision_force_radius=20.0,
        collision_force_strength=0.7,
        use_centering_force=True,
            

    )

    with container:
        subCol1, subCol2 = st.columns([5, 1])
        with subCol1:
            components.html(fig.to_html(), height=620, scrolling=True)
        with subCol2:
            components.html(chart_legend_css, height=620)


def create_multiselect(catalog_name: str, values: list, container: object):
    with container:
        selected_values = st.multiselect("Filter by groups from \"" + catalog_name + "\" : ",
            values,
            placeholder="Select one or more options (optional)"
        )

    return selected_values

def get_column_values(mdo):
    result = {}
    df = mdo.data

    for column in df.columns:
        if column != 'bcr_patient_barcode':
            unique_values = df[column].dropna().unique().tolist()
            unique_values = [value for value in unique_values if value != '' and not str(value).startswith('Unnamed')]
            unique_values.sort()
            if unique_values:  # Only include columns with non-empty values
                result[column] = unique_values

    return result

def map_index_to_unsorted(index_in_ordered: int, ordered_list: list, unordered_list: list) -> int:
    if not isinstance(index_in_ordered, int):
        raise ValueError("The input index must be an integer.")
    if index_in_ordered < 0 or index_in_ordered >= len(ordered_list):
        raise IndexError("The input index is out of range for the ordered list.")

    if sorted(ordered_list) != sorted(unordered_list):
        raise ValueError("The ordered_list and unordered_list must contain the same values.")

    unordered_index_map = {value: idx for idx, value in enumerate(unordered_list)}
    value_in_ordered = ordered_list[index_in_ordered]
    unordered_index = unordered_index_map[value_in_ordered]
    return unordered_index

# A helper function to force a page reload using JS
def rerun():
    st.markdown("<script>window.location.reload();</script>", unsafe_allow_html=True)


if __name__ == '__main__':
    # Logo in sidebar
    st.sidebar.image("./images/MIS_Portal_logo.png", width=250)

    # Added navigation buttons in the sidebar
    st.sidebar.markdown("""
        <style>
        .stButton>button {
            text-align: left;
            padding: 10px 20px;
            margin: 2px;
            width: 100%;
            display: flex;
            align-items: center;
        }
        .stButton>button>span {
            padding-left: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize the page state
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    # Navigation buttons
    if st.sidebar.button('Home'):
        st.session_state.page = "Home"
        reset_session_state_until('Filter_data_button')
        print_session_state()
    if st.sidebar.button('Analyse'):
        reset_session_state_until('Filter_data_button')
        print_session_state()
        st.session_state.page = "Analyse"
        print_session_state()
    if st.sidebar.button('AI Assistant'):  # New tab
        st.session_state.page = "AI Assistant"
        reset_session_state_until('Filter_data_button')
        print_session_state()
    if st.sidebar.button('FAQ'):
        st.session_state.page = "FAQ"
        reset_session_state_until('Filter_data_button')
        print_session_state()
    if st.sidebar.button('About'):
        st.session_state.page = "About"
        reset_session_state_until('Filter_data_button')
        print_session_state()
    if st.sidebar.button('News'):
        st.session_state.page = "News"
        reset_session_state_until('Filter_data_button')
        print_session_state()
    if st.sidebar.button('Results'):  # New tab
        st.session_state.page = "Results"
        reset_session_state_until('Filter_data_button')
        print_session_state()


    # Render content based on the current page state

    # Render Home Page
    if st.session_state.page == "Home":
        # Logo in the top left corner
        st.image("./images/MIS_Portal_logo.png", width=300)

        # Top frame with description
        st.markdown("""
        <div style="background-color:#f0f0f0; padding:20px; border-radius:10px; 
                    margin-top:20px; text-align:center; font-size:18px;">
            <strong>The Portal Analyzes Molecular Interaction Signatures</strong><br><br>
            The <strong>public version</strong> of the MIS portal provides a <strong>framework</strong> for comparing biological samples based on molecular interaction signatures. 
            Using deep learning metrics, statistical tests, and graph-based methods, it identifies key interaction patterns and assesses their biological relevance for <strong>biomarker discovery and hypothesis generation.</strong>
        </div>
        """, unsafe_allow_html=True)

        # Navigation buttons: Analyse + Examples
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Analyse Your Samples"):
                st.session_state.page = "Analyse"
                reset_session_state_until('Filter_data_button')
                print_session_state()
                #rerun()
        with col2:
            if st.button("See Examples"):
                st.session_state.page = "Examples"
                #rerun()

        # Spacer
        st.markdown("<br>", unsafe_allow_html=True)
        import base64

        def get_base64_image(image_path):
            with open(image_path, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode()

        # Base64-encoded image
        encoded_evidence = get_base64_image("./images/evidence.png")
        encoded_expert = get_base64_image("./images/expert.png")

        # Two content columns
        col1, col2 = st.columns([1, 1])

        # Left column: supporting evidence
        with col1:
            st.markdown(f"""
                <div style='text-align: center;'>
                    <img src='data:image/png;base64,{encoded_evidence}' width='50'/>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="background-color:#e0e0e0; padding:20px; border-radius:10px; 
                        text-align:center; font-size:16px;">
                <strong>The Portal uses distinct levels of supporting evidence</strong><br>
                Molecular Interaction Signatures (MIS) are derived from the analysis of molecular profiles and are annotated by a comprehensive set of knowledgebases and computational estimations.
                <br><br>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Read more"):
                st.session_state.page = "FAQ"
                rerun()

        # Right column: expert consensus
        with col2:
            st.markdown(f"""
                <div style='text-align: center;'>
                    <img src='data:image/png;base64,{encoded_expert}' width='50'/>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="background-color:#e0e0e0; padding:20px; border-radius:10px; 
                        text-align:center; font-size:16px;">
                <strong>The Portal follows clinical expert consensus</strong><br>
                The MIS portal supports the interpretation of molecular interaction signatures in the context of clinical expert consensus under the Cancer Core Europe umbrella and latest scientific evidence.
                <br><br>
            </div>
            """, unsafe_allow_html=True)


            # Button inside the frame
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("About us"):
                st.session_state.page = "About"
                rerun()

        # Informative text at the bottom
        st.markdown("""
        <div style="margin-top:30px; text-align:justify; font-size:16px;">
            <p><strong>This is an open-access version of the Molecular Interaction Signatures Portal, designed for the comparative study of biological samples based on molecular interaction signatures.</strong></p>
            <p>The portal utilizes computational methods, including deep learning-derived relevance metrics and statistical analyses, with references to the applied algorithms and data sources provided in the results. 
            Some resources integrated within the portal may require a license for commercial applications or clinical use; therefore, this version is strictly limited to academic research. 
            Users must accept these terms upon first login, which requires a valid email address. When using this portal, please cite:</p>
        </div>
        """, unsafe_allow_html=True)

        # Logos at the bottom
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image("./images/CRUK_NBC.png", width=220)
        with col2:
            st.image("./images/CRUK_NBC_DCR.png", width=220)
        with col3:
            st.image("./images/idiap_logo.png", width=170)
        with col4:
            st.image("./images/CCE.png", width=150)
        with col5:
            st.image("./images/CCE_DART.png", width=170)


    elif st.session_state.page == "Analyse":
        
        
        st.title("Analyse Molecular Interaction Signatures")
        st.markdown("""
            <div style="background-color:#f0f0f0; padding:20px; border-radius:10px; 
                        margin-top:20px; font-size:16px;">
            <strong>Instructions to Analyse Your Samples</strong>

            - 📁 **Upload your LRP results file** containing gene-gene interaction data.  
            _Format: CSV or TSV with columns: `SampleID`, `GeneA`, `GeneB`, `LRP_score`_

            - 🧾 **Upload your metadata file** with sample IDs and group labels.  
            _Format: CSV or TSV with columns: `SampleID`, `Group`_

            - 🔍 **Filter your data**:  
            &nbsp;&nbsp;&nbsp;&nbsp;• Choose a group of samples using metadata  
            &nbsp;&nbsp;&nbsp;&nbsp;• Optionally, specify a set of genes of interest

            - ⚙️ **Select analysis type**:  
            &nbsp;&nbsp;&nbsp;&nbsp;• Sample vs Sample  
            &nbsp;&nbsp;&nbsp;&nbsp;• Sample vs Group Median  
            &nbsp;&nbsp;&nbsp;&nbsp;• Group vs Group (Median vs Median)

            - 📊 **Choose statistical approach**:  
            &nbsp;&nbsp;&nbsp;&nbsp;• Univariable test (Mann–Whitney U test)  
            &nbsp;&nbsp;&nbsp;&nbsp;• Graph-based interaction comparison

            - 🧠 **For graph analysis**, define a **difference threshold**:  
            This controls which `LRP_diff` values (Graph1 - Graph2) are shown in the Difference Graph.

            - 🧬 **Automatic evidence extraction** will be performed after the analysis.

            - 🤖 To explore extracted evidence with the AI, go to the **'AI Assistant'** tab.
            </div>
            """, unsafe_allow_html=True)



        # Create a column layout with two columns for the upload buttons
        col1, col2 = st.columns(2)

        with col1:
            # Data upload button for LRP data
            path_to_LRP_data = st.file_uploader("Provide LRP-based data")

            if path_to_LRP_data is not None:
                #st.session_state['lrp_df'] = dtl.LRPData(file_path=save_my_uploaded_file('/tmp', path_to_LRP_data),
                #                                        delimiter=",").read_and_validate()
                temp_dir = tempfile.gettempdir()
                st.session_state['lrp_df'] = dtl.LRPData(file_path=save_my_uploaded_file(temp_dir, path_to_LRP_data),
                                                        delimiter=",").read_and_validate()
                st.markdown(f"""
                    <div style="background-color: #e6f7ff; color: black; padding: 4px 10px; border-radius: 4px; font-size: 14px;">
                        File {path_to_LRP_data.name} uploaded.
                    </div>
                """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        with col2:
            # Data upload button for metadata
            path_to_metadata = st.file_uploader("Provide metadata")

            if path_to_metadata is not None:
                temp_dir = tempfile.gettempdir()
                st.session_state['metadata_df'] = dtl.MetaData(file_path=save_my_uploaded_file(temp_dir, path_to_metadata),
                                                            delimiter=",")
                st.markdown(f"""
                    <div style="background-color: #e6f7ff; color: black; padding: 4px 10px; border-radius: 4px; font-size: 14px;">
                        File {path_to_metadata.name} uploaded.
                    </div>
                """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        civic_features_path = "./data/01-Feb-2025-FeatureSummaries.tsv"
        civic_mp_path = "./data/01-Feb-2025-MolecularProfileSummaries.tsv"

        if civic_features_path and civic_mp_path:
            st.session_state['civic_data'] = civic.CivicData(civic_features_path, civic_mp_path)
            st.session_state['civic_data'].load_data()

        # Check if all required data paths exist
        if path_to_LRP_data and path_to_metadata and civic_features_path and civic_mp_path:
            # Retrieve column values from metadata_df
            filter_catalog = get_column_values(st.session_state['metadata_df'])

            # Create a combined form for both data filtering and keyword selection
            combined_form = st.form('combined_filters')
            with combined_form:
                st.markdown("### Filter samples")
                # Create widgets for filtering based on metadata columns
                filters = {}
                for key, values in filter_catalog.items():
                    filters[key] = create_multiselect(key, values, combined_form)

                st.markdown("### Filter genes of interest")
                # File uploader widget for frequent keywords
                path_to_plkeywords = st.file_uploader("Upload file with genes of interest")
                if path_to_plkeywords is not None:
                    temp_dir = tempfile.gettempdir()
                    st.session_state['frequent_kws'] = pd.read_csv(
                        save_my_uploaded_file(temp_dir, path_to_plkeywords), header=None
                    )
                    st.info('File {0} has been analysed.'.format(path_to_plkeywords.name))

                st.markdown("or")
                # Set up the keywords list based on whether a file was uploaded
                if st.session_state.get('frequent_kws') is None or st.session_state['frequent_kws'].empty:
                    # No file uploaded – generate keywords from data
                    keywords = find_my_keywords(
                        st.session_state['lrp_df']
                        if st.session_state['filtered_tts_lrp_df'].empty
                        else st.session_state['filtered_tts_lrp_df']
                    )
                    default_keywords = []
                else:
                    # File uploaded – use its content as the default selection
                    keywords = find_my_keywords(
                        st.session_state['filtered_tts_lrp_df']
                        if not st.session_state['filtered_tts_lrp_df'].empty
                        else st.session_state['lrp_df']
                    )
                    default_keywords = st.session_state['frequent_kws'][0].tolist()

                # Multiselect widget for selecting keywords
                keywords_selected = combined_form.multiselect(
                    "Please select your genes of interest: ",
                    keywords,
                    default_keywords,
                    placeholder="Select one or more (optional)"
                )
                st.session_state['keywords'] = keywords_selected
                # A single "Run" button to execute both filtering steps
                filter_button = combined_form.form_submit_button(label='Filter data')

            # Execute filtering when the "Filter data" button is clicked
            if filter_button:
                reset_session_state_until(key_to_reset = 'Filter_data_button')
                # Filter data based on the selected metadata filters
                metadata = st.session_state['metadata_df'].data.copy()
                filters_applied = False

                for filter_name, values in filters.items():
                    if values:  # if any filter is selected
                        filters_applied = True
                        metadata = metadata[metadata[filter_name].isin(values)]

                if filters_applied:
                    valid_bc = metadata.index.tolist()
                    st.session_state['filtered_tts_lrp_df'] = st.session_state['lrp_df'].loc[valid_bc]
                else:
                    # No filters selected - use the entire dataset
                    st.session_state['filtered_tts_lrp_df'] = st.session_state['lrp_df'].copy()
                if st.session_state['filtered_tts_lrp_df'].empty:
                    st.warning("The selection criteria you have chosen are not yielding any results. Please select alternative values.")
                else:
                    # Custom success message with lower frame size (smaller padding and font size)
                    custom_success_html = """
                    <div style="background-color: #d4edda; border: 1px solid #c3e6cb; padding: 5px 10px; border-radius: 4px; font-size: 14px;">
                    The selection criteria you have chosen are filtered successfully! Proceed to the next step.
                    </div>
                    """
                    st.markdown(custom_success_html, unsafe_allow_html=True)
                    st.session_state['filters_form_completed'] = True
                # Filter data based on the selected keywords
                if not keywords_selected:
                    st.warning("No keywords selected. Using the entire dataset for analysis.")
                    st.session_state['filtered_lrp_df'] = st.session_state['filtered_tts_lrp_df']
                else:
                    st.session_state['filtered_lrp_df'] = fg.filter_columns_by_keywords(
                        st.session_state['filtered_tts_lrp_df'], st.session_state['keywords']
                    )

                # Check if the filtered DataFrame is empty
                if st.session_state['filtered_lrp_df'].empty:
                    st.error("The filtered dataset is empty. Please select different keywords or filters.")
                else:
                    # Prepare the data for graph generation
                    st.session_state['filtered_data_to_graphs'] = fg.prepare_lrp_to_graphs(st.session_state['filtered_lrp_df'])
                    #st.session_state['first_form_completed'] = True
                    st.session_state['Filter_data_button'] = True
                print_session_state()
                # Filter data based on the selected keywords
                #if not keywords_selected:
                #    keywords_selected = None
                
                #st.session_state['filtered_lrp_df'] = fg.filter_columns_by_keywords(st.session_state['filtered_tts_lrp_df'], st.session_state['keywords'])
                #st.session_state['filtered_data_to_graphs'] = fg.prepare_lrp_to_graphs(st.session_state['filtered_lrp_df'])
                #st.session_state['first_form_completed'] = True
                #st.success("Keywords filtered successfully! Proceed to the next step.")


        # After the filtering stage has completed
        if st.session_state.get('Filter_data_button', False):
            # In the "Analyse" branch, after filtering has completed and analysis type buttons are shown:
            st.markdown("### Type of analysis")
            col_a, col_b, col_c = st.columns(3)
            if col_a.button("sample-sample"):
                st.session_state["analysis_type"] = "sample-sample"
                reset_session_state_until('Type_of_analysis_selected')
                st.session_state["Type_of_analysis_selected"] = True
                print_session_state()
                #st.session_state["ready_for_comparison"] = False
                #st.session_state["G1_G2_displayed"] = False
                #print_session_state()
            if col_b.button("sample-group"):
                st.session_state["analysis_type"] = "sample-group"
                reset_session_state_until('Type_of_analysis_selected')
                st.session_state["Type_of_analysis_selected"] = True
                print_session_state()
                #st.session_state["ready_for_comparison"] = False
                #st.session_state["G1_G2_displayed"] = False
                #print_session_state()
            if col_c.button("group-group"):
                st.session_state["analysis_type"] = "group-group"
                #st.session_state["ready_for_comparison"] = False
                #st.session_state["G1_G2_displayed"] = False
                reset_session_state_until('Type_of_analysis_selected')
                st.session_state["Type_of_analysis_selected"] = True
                print_session_state()
                # Add visual feedback
            if "analysis_type" in st.session_state:
                if st.session_state["analysis_type"] == "sample-sample":
                    col_a.markdown("**Selected: sample-sample**")
                elif st.session_state["analysis_type"] == "sample-group":
                    col_b.markdown("**Selected: sample-group**")
                elif st.session_state["analysis_type"] == "group-group":
                    col_c.markdown("**Selected: group-group**")

            
            #####################################################################################################################
            # UNIVARIABLE SAMPLE - SAMPLE COMPARISON
            if st.session_state.get("analysis_type") == "sample-sample":
                st.markdown("### Sample-vs-Sample Comparison")
                #st.session_state['']
                # Display sub-buttons for comparison type:
                col_uni, col_graph = st.columns(2)
                if col_uni.button("Univariable Comparison"):
                    st.session_state["comparison_type"] = "univariable"
                    reset_session_state_until('Type_of_comparison_selected')
                    st.session_state["Type_of_comparison_selected"] = True                    
                    print_session_state()
                if col_graph.button("Graph Comparison"):
                    st.session_state["comparison_type"] = "graph"
                    reset_session_state_until('Type_of_comparison_selected')
                    st.session_state["Type_of_comparison_selected"] = True
                    print_session_state()
                    
                

                if st.session_state.get("comparison_type") == "univariable":
                    
                    with st.form("sample_vs_sample_form"):
                        if 'filtered_lrp_df' not in st.session_state:
                            st.session_state['filtered_lrp_df'] = pd.DataFrame([])
                        
                        if st.session_state["filtered_lrp_df"].empty:
                            st.error("Filtered dataset is empty. Please review your filtering criteria.")
                            st.session_state['sample_names'] = []
                        else:
                            st.session_state['sample_names'] = list(st.session_state["filtered_lrp_df"].index)
                            print(st.session_state['sample_names'])

                        if st.session_state['sample_names']:
                            # Remove the on_change callback from the selectbox: recalc options manually below.
                            sample1 = st.selectbox("Select Sample 1", st.session_state['sample_names'], key="sample1")
                            sample2 = st.selectbox("Select Sample 2", st.session_state['sample_names'], key="sample2")
                            print(sample1)
                            print(sample2)
                            # Recalculate sample2_options based on sample1
                            #st.session_state.
                            #sample2_options = [s for s in st.session_state['sample_names'] if s != sample1]
                            
                            #if len(st.session_state.sample2_options)==0:
                            #    st.error("Not enough distinct samples available for comparison.")
                            #    sample2 = None
                            #else:
                            
                            
                            #st.session_state['sample1'] = sample1
                            #st.session_state['sample2'] = sample2                          
                            
                            top_n_features = st.slider("Select Top N Features", min_value=1, max_value=100, value=10)
                            singles = sorted({col.split("_")[-1] for col in st.session_state['filtered_lrp_df'].columns if "_" in col})
                            pairs = [f"{t1}-{t2}" for i, t1 in enumerate(singles) for t2 in singles[i:]]
                            available_node_types = sorted(list(set(singles + pairs)))
                            selected_node_types = st.multiselect("Select Edge Type(s)", available_node_types,
                                                                default=[available_node_types[0]] if available_node_types else [])
                            compare_submit = st.form_submit_button("Compare Samples")
                        else:
                            compare_submit = False

                    if compare_submit:
                        if sample1 == sample2 or sample2 is None:
                            #print('Sample1:', sample1)
                            #print('Sample2:', sample2)
                            st.error("Selected samples must be distinct. Please choose two different samples for comparison.")
                        else:
                            print("Filtered DataFrame Index:", st.session_state['filtered_lrp_df'].index)
                            if sample1 not in st.session_state['filtered_lrp_df'].index or sample2 not in st.session_state['filtered_lrp_df'].index:
                                st.error(f"One or both selected samples ({sample1}, {sample2}) are not present in the filtered dataset. Please select valid samples.")
                            else:
                                save_plot = False  # Adjust as needed
                                comparison = lrpcomp.SampleVsSampleComparison(
                                    sample1_name=sample1,
                                    sample2_name=sample2,
                                    data_df=st.session_state['filtered_lrp_df'],
                                    clinical_features_df=st.session_state["metadata_df"].data
                                )
                                comparison.compute_boxplot_values()
                                comparison.select_top_n_by_column(column='median_abs_diff', n=top_n_features, ascending=False)
                                pattern = '|'.join(selected_node_types) if isinstance(selected_node_types, list) else selected_node_types
                                _ = comparison.filter_and_merge_data(selected_type=pattern)
                                fig = comparison.plot_scatter(selected_type=pattern, save_plot=save_plot)
                                # Center the figure in a column and set a defined width
                                centered_col = st.columns([1, 1, 1])[1]  # Middle column is wider
                                with centered_col:
                                    st.pyplot(fig, use_container_width=False)  # Removed width and height arguments

                elif st.session_state.get("comparison_type") == "graph":
                    #st.session_state['G1_G2_displayed'] = False
                    # Placeholder for Graph Comparison analysis.
                    st.markdown("### Graph Comparison for Sample vs Sample")

            #####################################################################################################################
            # UNIVARIABLE SAMPLE - GROUP COMPARISON
            if st.session_state.get("analysis_type") == "sample-group":
                st.markdown("### Sample-vs-Group Comparison")
                # Display sub-buttons for comparison type:
                col_uni2, col_graph2 = st.columns(2)
                if col_uni2.button("Univariable Comparison"):
                    st.session_state["comparison_type"] = "univariable"
                    reset_session_state_until('Type_of_comparison_selected')
                    st.session_state["Type_of_comparison_selected"] = True                    
                    print_session_state()
                if col_graph2.button("Graph Comparison"):
                    st.session_state["comparison_type"] = "graph"
                    reset_session_state_until('Type_of_comparison_selected')
                    st.session_state["Type_of_comparison_selected"] = True                    
                    print_session_state()
                
                if st.session_state.get("comparison_type") == "univariable":
                    # Use filtered metadata from the filtering stage
                    filtered_metadata = st.session_state["metadata_df"].data.loc[
                        st.session_state["filtered_tts_lrp_df"].index
                    ]
                    # ----
                    # Place the grouping column widget outside the form so its callback works.
                    metadata_cols = list(filtered_metadata.columns)
                    if "sg_grouping_column" not in st.session_state:
                        st.session_state["sg_grouping_column"] = metadata_cols[0]
                        st.session_state["sg_group_options"] = sorted(
                            filtered_metadata[metadata_cols[0]].dropna().unique().tolist()
                        )
                    # When the grouping column changes, update the group options.
                    chosen_column = st.selectbox("Select Grouping Column", metadata_cols,
                                                key="sg_grouping_column",
                                                on_change=lambda: st.session_state.update({
                                                    "sg_group_options": sorted(
                                                        filtered_metadata[st.session_state["sg_grouping_column"]]
                                                        .dropna().unique().tolist()
                                                    )
                                                }))
                    # ----
                    # Now inside the form, use the updated sg_group_options.
                    with st.form("sample_vs_group_form"):
                        # Create selection box for sample (from lrp_df index).
                        sample_names = list(st.session_state['filtered_lrp_df'].index)
                        sample1 = st.selectbox("Select Sample", sample_names, key="sample_vs_group_sample")
                        # Use the updated group options from filtered metadata.
                        chosen_group = st.selectbox("Select Group", st.session_state["sg_group_options"],
                                                    key="group_selected")
                        top_n_features = st.slider("Select Top N Features", min_value=1, max_value=100, value=10)
                        # Determine available node types (both single and paired).
                        singles = sorted({col.split("_")[-1] for col in st.session_state['filtered_lrp_df'].columns if "_" in col})
                        pairs = [f"{t1}-{t2}" for i, t1 in enumerate(singles) for t2 in singles[i:]]
                        available_node_types = sorted(list(set(singles + pairs)))
                        selected_node_types = st.multiselect("Select Edge Type(s)", available_node_types,
                                                            default=[available_node_types[0]] if available_node_types else [])
                        compare_submit = st.form_submit_button("Compare Sample vs Group")
                    
                    if compare_submit:
                        save_plot = False  # Adjust as needed
                        comparison = lrpcomp.SampleVsGroupComparison(
                            sample1_name=sample1,
                            column_name=chosen_column,
                            group1_name=chosen_group,
                            data_df=st.session_state['filtered_lrp_df'],
                            clinical_features_df=st.session_state["metadata_df"].data
                        )
                        comparison.compute_boxplot_values()
                        comparison.select_top_n_by_column(column='p-value', n=top_n_features, ascending=True)
                        pattern = '|'.join(selected_node_types) if isinstance(selected_node_types, list) else selected_node_types
                        merged_data = comparison.filter_and_merge_data(selected_type=pattern)
                        fig = comparison.plot_violin_sample_vs_group(
                            merged_data,
                            selected_type=pattern,
                            plot_title=f"Sample vs Group: {sample1} vs {chosen_group}",
                            save_plot=save_plot
                        )
                        # Upewnij się, że do st.pyplot() przekazujemy obiekt fig.
                        if fig is None:
                            import matplotlib.pyplot as plt
                            fig = plt.gcf()
                        centered_col = st.columns([1, 1, 1])[1]  # Middle column is wider
                        with centered_col:
                            st.pyplot(fig, use_container_width=False)  # Removed width and height arguments
                
                elif st.session_state.get("comparison_type") == "graph":
                    st.markdown("### Graph Comparison for Sample vs Group")


            #####################################################################################################################
            # UNIVARIABLE GROUP - GROUP COMPARISON
            if st.session_state.get("analysis_type") == "group-group":
                st.markdown("### Group-vs-Group Comparison")
                # Display sub-buttons for comparison type:
                col_uni2, col_graph2 = st.columns(2)
                if col_uni2.button("Univariable Comparison"):
                    st.session_state["comparison_type"] = "univariable"
                    reset_session_state_until('Type_of_comparison_selected')
                    st.session_state["Type_of_comparison_selected"] = True                    
                    print_session_state()
                if col_graph2.button("Graph Comparison"):
                    st.session_state["comparison_type"] = "graph"
                    reset_session_state_until('Type_of_comparison_selected')
                    st.session_state["Type_of_comparison_selected"] = True                    
                    print_session_state()

                if st.session_state.get("comparison_type") == "univariable":
                    # Use filtered metadata from the filtering stage
                    filtered_metadata = st.session_state["metadata_df"].data.loc[
                        st.session_state["filtered_tts_lrp_df"].index
                    ]
                    # OUTSIDE the form: select grouping column from filtered metadata and update group options via a callback.
                    metadata_cols = list(filtered_metadata.columns)
                    def update_gg_groups():
                        st.session_state["gg_group_options"] = sorted(
                            filtered_metadata[st.session_state["gg_grouping_column"]]
                            .dropna().unique().tolist()
                        )
                    if "gg_grouping_column" not in st.session_state:
                        st.session_state["gg_grouping_column"] = metadata_cols[0]
                    chosen_column = st.selectbox("Select Grouping Column", metadata_cols,
                                                key="gg_grouping_column", on_change=update_gg_groups)
                    if "gg_group_options" not in st.session_state:
                        st.session_state["gg_group_options"] = sorted(
                            filtered_metadata[chosen_column].dropna().unique().tolist()
                        )

                    # INSIDE the form: select the two groups and other parameters.
                    with st.form("group_vs_group_form"):
                        group1 = st.selectbox("Select Group 1", st.session_state["gg_group_options"],
                                            key="gg_group1")
                        group2 = st.selectbox("Select Group 2", st.session_state["gg_group_options"],
                                            key="gg_group2")
                        
                        top_n_features = st.slider("Select Top N Features", min_value=1, max_value=100, value=10)
                        # Determine available node types (both single and paired)
                        singles = sorted({col.split("_")[-1] for col in st.session_state['filtered_lrp_df'].columns if "_" in col})
                        pairs = []
                        for i, t1 in enumerate(singles):
                            for t2 in singles[i:]:
                                pairs.append(f"{t1}-{t2}")
                        available_node_types = sorted(list(set(singles + pairs)))
                        selected_node_types = st.multiselect("Select Edge Type(s)", available_node_types,
                                                            default=[available_node_types[0]] if available_node_types else [])
                        compare_submit = st.form_submit_button("Compare Group vs Group")

                    # On form submission:
                    if compare_submit:
                        if group1 == group2:
                            st.error("Selected groups must be distinct. Please choose two different groups for comparison.")
                        else:
                            save_plot = False  # Adjust as needed
                            # Filter the filtered metadata to include only the selected groups.
                            filtered_sel_metadata = filtered_metadata[
                                filtered_metadata[chosen_column].isin([group1, group2])
                            ]
                            # Ensure that the filtered_sel_metadata and lrp_df share the same index.
                            common_index = st.session_state['filtered_lrp_df'].index.intersection(filtered_sel_metadata.index)
                            filtered_sel_metadata = filtered_sel_metadata.loc[common_index]
                            data_df = st.session_state['filtered_lrp_df'].loc[common_index]
                            
                            # Create the comparison object using the index-matched data.
                            comparison = lrpcomp.GroupVsGroupComparison(
                                column_name=chosen_column,
                                group1_name=group1,
                                group2_name=group2,
                                data_df=data_df,
                                clinical_features_df=filtered_sel_metadata
                            )
                            comparison.compute_boxplot_values()
                            comparison.select_top_n_by_column(column='p-value', n=top_n_features, ascending=True)
                            pattern = '|'.join(selected_node_types) if isinstance(selected_node_types, list) else selected_node_types
                            merged_data = comparison.filter_and_merge_data(selected_type=pattern)
                            fig = comparison.plot_violin(merged_data, selected_type=pattern, save_plot=save_plot)
                            # Poprawka: upewniamy się, że do st.pyplot() przekazujemy obiekt fig.
                            if fig is None:
                                fig = plt.gcf()
                            centered_col = st.columns([1, 1, 1])[1]  # Middle column is wider
                            with centered_col:
                                st.pyplot(fig, use_container_width=False)  # Removed width and height arguments

                elif st.session_state.get("comparison_type") == "graph":
                    st.markdown("### Graph Comparison for Group vs Group")
    
    elif st.session_state.page == "AI Assistant":
        st.title("AI Assistant")
        st.markdown("""
            <div style="background-color:#f0f0f0; padding:20px; border-radius:10px; 
                        margin-top:20px; text-align:center; font-size:18px;">
                <strong>AI Assistant</strong><br>
                This section presents examples of interactions with an AI assistant used to support reasoning analysis, 
                hypothesis generation, and interpretation of results. It showcases how AI tools can complement human 
                expertise in scientific research.
            </div>
        """, unsafe_allow_html=True)
                        
    elif st.session_state.page == "Examples":
        st.title("Examples")
        st.markdown("""
            <div style="background-color:#f0f0f0; padding:20px; border-radius:10px; 
                        margin-top:20px; text-align:center; font-size:18px;">
                <strong>Examples of Analyses</strong><br>
                <br>
                Here you can find examples of analyses performed using the Molecular Interaction Signatures Portal.
                <br><br>
                Explore the examples to understand how to use the portal for your own analyses.
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        # Logos at the bottom
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image("./images/CRUK_NBC.png", width=220)
        with col2:
            st.image("./images/CRUK_NBC_DCR.png", width=220)
        with col3:
            st.image("./images/idiap_logo.png", width=170)
        with col4:
            st.image("./images/CCE.png", width=150)
        with col5:
            st.image("./images/CCE_DART.png", width=170)

    elif st.session_state.page == "FAQ":
        st.title("Frequently Asked Questions")
        st.markdown("""
            <div style="background-color:#f0f0f0; padding:20px; border-radius:10px; 
                        margin-top:20px; text-align:center; font-size:18px;">
                <strong>Frequently Asked Questions</strong><br>
                <br>
                Here you can find answers to the most commonly asked questions about the Molecular Interaction Signatures Portal.
                <br><br>
                If you have any other questions, please contact us at support@misportal.org.
            </div>
        """, unsafe_allow_html=True)
        # Add a line break between the frame and the first expandable topic
        st.markdown("<br>", unsafe_allow_html=True)
        # First expandable topic: "What is the MIS Portal?"
        with st.expander("What is the MIS Portal?"):
            st.markdown("""
                **What is the public MISP website?**  
                The public version of the portal provides an open lightweight version of the MISP analytical pipeline. Please note that the analyses provided in this public version are limited and do not incorporate certain explanatory layers that are relevant for clinical decision-making. This resource is intended for research purposes only.
                
                **Are there any technical requirements for the public MISP?**  
                The public MISP should work well with the following web browsers: Chrome Version v71, Microsoft Edge, FireFox Quantum, Opera V57, Safari.  Please note that Internet Explorer is not supported, and you may experience visualization problems with it.
            """, unsafe_allow_html=True)

        # Second expandable topic: "What does the public MISP analyses provide?"
        with st.expander("What does the public MISP analyses provide?"):
            st.markdown("""
                **What does the public MISP support as input?**
                The public version of the MISP supports the analysis of multiomics data—including gene expression levels, mutations, copy number alterations (such as amplifications and deletions), gene fusions, and protein expression data. These data can be uploaded in CSV format. For additional details on input formats and requirements, please refer to the tooltips available in the ‘Analyse’ interface.
                
                **How are the public MISP reports structured?** 
                The public MISP generates an HTML report that classifies the uploaded samples into three distinct tables based on the type of analysis and the relevance of evidence. Each table is accompanied by multiple sources of evidence that support the classification. The HTML report also features interactive elements that allow users to: 1) Open pop-up windows providing further information and detailed gene annotations. 2) Access external resources with the original evidence.
            """, unsafe_allow_html=True)

        # Third expandable topic: "Which resources are employed to interpret the results?"
        with st.expander("Which resources are employed to interpret the results?"):
            st.markdown("""
                **Which resources are used for interpretation?**
                The MISP harnesses a diverse array of resources to annotate the genes under analysis. These resources include sequencing data from previous cohorts, a variety of bioinformatics tools, and public databases. While some resources are developed internally, many are community-based—in these cases, the specific version used in an analysis is displayed in the upper right corner of the report. Notably, the public portal integrates several knowledge bases created by international initiatives, which are open for academic research. Data models and gene nomenclature are first harmonized to ensure accurate aggregation, and additional filtering can be applied (for instance, to differentiate weaker supporting evidence as indicated by proprietary knowledgebase metadata) as needed. Currently, the knowledge bases harmonized in the public portal include ClinVar, BRCA-Exchange, OncoKB, and CIViC.
                
                **Are the knowledgebase contents updated?** 
                Because the harmonization of the knowledgebase data cannot be fully automated, the content is downloaded and then manually processed to reformat it. This updating process is performed periodically. For further details, please note that every MISP report lists the version, reference, and access information for all resources used to annotate the genes, along with the original evidence assertions. Additionally, the “News” section on the public MISP website provides updates and relevant information for users.
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        # Logos at the bottom
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image("./images/CRUK_NBC.png", width=220)
        with col2:
            st.image("./images/CRUK_NBC_DCR.png", width=220)
        with col3:
            st.image("./images/idiap_logo.png", width=170)
        with col4:
            st.image("./images/CCE.png", width=150)
        with col5:
            st.image("./images/CCE_DART.png", width=170)


    elif st.session_state.page == "About":
        st.title("About Us")
        st.markdown("""
            <div style="background-color:#f0f0f0; padding:20px; border-radius:10px; 
                        margin-top:20px; text-align:center; font-size:18px;">
                <strong>About the Molecular Interaction Signatures Portal</strong><br>
                <br>
                <strong>The Molecular Interaction Signatures (MIS) Portal </strong>is a comprehensive platform designed to analyze and compare biological samples based on their molecular interaction signatures. It leverages advanced computational methods—including deep learning metrics, statistical testing, and graph-based approaches—to uncover key interaction patterns and assess their biological relevance.
                <br><br>
                <strong>Our mission is to accelerate biomarker discovery and hypothesis generation</strong> by providing researchers with a robust, flexible framework for analyzing molecular profiles. The portal integrates curated knowledgebase data with computational estimations to deliver a detailed and insightful exploration of molecular interactions.
                <br><br>
                The MIS Portal is developed by members of <strong>the Digital Cancer Research AI Team</strong> at the CRUK National Biomarker Centre and members of <strong>the Neuro-symbolic AI Group</strong> at the Idiap Research Institute. Our group conducts research at the interface of neural and symbolic methods, aiming to build the next generation of explainable, data-efficient, and safe AI systems. We focus on integrating latent and explicit data representations to enhance learning and reasoning capabilities in complex domains.
                <br>
                The development of the MIS Portal reflects our group’s broader mission: enabling transparent and robust AI applications in critical domains such as healthcare and life sciences.
                <br><br>
                <strong>Contact Us:</strong><br>
                For more information, please visit our website or contact us at info@misportal.org.
                <br><br>
                <strong>Design & Development:</strong><br>
                <a href="https://www.linkedin.com/in/oskar-wysocki/" target="_blank" style="color: #0077b5; text-decoration: underline;">Oskar Wysocki, PhD</a> 
                [<a href="https://scholar.google.com/citations?user=3r-xFXsAAAAJ&hl=en" target="_blank" style="color: #4285F4; text-decoration: underline;">Google Scholar</a>]<br>
                <a href="https://www.linkedin.com/in/magdalena-wysocka-052905141/" target="_blank" style="color: #0077b5; text-decoration: underline;">Magdalena Wysocka, PhD</a> 
                [<a href="https://scholar.google.com/citations?user=lJQO-lEAAAAJ&hl=en" target="_blank" style="color: #4285F4; text-decoration: underline;">Google Scholar</a>]<br>
                <a href="https://www.linkedin.com/in/mauricio-jacobo/" target="_blank" style="color: #0077b5; text-decoration: underline;">Mauricio Jacobo-Romero, PhD</a>
                <br>
                <a href="https://www.linkedin.com/in/andrefreitas/" target="_blank" style="color: #0077b5; text-decoration: underline;">Andre Freitas, PhD</a> 
                [<a href="https://scholar.google.com/citations?user=ExmHmMoAAAAJ&hl=en" target="_blank" style="color: #4285F4; text-decoration: underline;">Google Scholar</a>]<br>
                <br><br>
                <strong>Collaborators:</strong><br>
                <a href="https://www.cancercoreeurope.eu/" target="_blank" style="color: black; text-decoration: underline;">Cancer Core Europe</a>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        # Logos at the bottom
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image("./images/CRUK_NBC.png", width=220)
        with col2:
            st.image("./images/CRUK_NBC_DCR.png", width=220)
        with col3:
            st.image("./images/idiap_logo.png", width=170)
        with col4:
            st.image("./images/CCE.png", width=150)
        with col5:
            st.image("./images/CCE_DART.png", width=170)

    elif st.session_state.page == "News":
        st.title("News")
        
        st.markdown("""
            <div style="background-color:#f0f0f0; padding:20px; border-radius:10px; 
                        margin-top:20px; text-align:center; font-size:18px;">
                <strong>Latest News</strong><br><br>
                Stay updated with the latest news and updates about the Molecular Interaction Signatures Portal.<br>
                Check back regularly for new features, updates, and announcements.
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Example LinkedIn posts section
        posts = [
            {
                "image_url": "https://imgur.com/9xdhThG.jpg",
                "post_url": "https://www.linkedin.com/posts/andrefreitas_can-llms-perform-logical-inference-in-highly-activity-7323468431167479809-Pg0r?utm_source=share&utm_medium=member_desktop&rcm=ACoAACJ73f4Bel0LYwdOIia675_aXTbVbFr6hBo",
                "caption": "How to evaluate highly specialized reasoning in LLMs without expensive manual annotations?"
            },
            {
                "image_url": "https://imgur.com/MwGZIim.jpg",
                "post_url": "https://www.linkedin.com/posts/andrefreitas_acl-2024-is-happening-this-week-in-bangkok-activity-7229406012217139201-WHSh?utm_source=share&utm_medium=member_desktop&rcm=ACoAACJ73f4Bel0LYwdOIia675_aXTbVbFr6hBo",
                "caption": "An LLM-based Knowledge Synthesis and Scientific Reasoning Framework for Biomedical Discovery"
            },
            {
                "image_url": "https://imgur.com/HqhZ680.jpg",
                "post_url": "https://www.linkedin.com/feed/update/urn:li:activity:7076861591119360000?utm_source=share&utm_medium=member_desktop&rcm=ACoAACJ73f4Bel0LYwdOIia675_aXTbVbFr6hBo",
                "caption": "Do you ask ChatGPT or GTP-4 scientific questions? Be aware of hallucinations and low factuality!"
            }
        ]
        
        # Display posts
        for post in posts:
            st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 30px;">
                    <div style="flex: 0 0 150px; margin-right: 20px;">
                        <a href="{post['post_url']}" target="_blank">
                            <img src="{post['image_url']}" alt="Post thumbnail" style="width: 100%; border-radius: 10px;">
                        </a>
                    </div>
                    <div style="flex: 1; font-size: 16px;">
                        <a href="{post['post_url']}" target="_blank" style="text-decoration: none; color: black;">
                            <strong>{post['caption']}</strong>
                        </a>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Logos at the bottom
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image("./images/CRUK_NBC.png", width=220)
        with col2:
            st.image("./images/CRUK_NBC_DCR.png", width=220)
        with col3:
            st.image("./images/idiap_logo.png", width=170)
        with col4:
            st.image("./images/CCE.png", width=150)
        with col5:
            st.image("./images/CCE_DART.png", width=170)
        
    elif st.session_state.page == "Results":
        st.title("Results")
        st.markdown("""
        <div style="background-color:#f0f0f0; padding:20px; border-radius:10px; 
                    margin-top:20px; text-align:justify; font-size:18px;">
            <strong>Related Papers</strong><br>
            <br>
            The following publications are closely related to the research presented in this work. 
                They provide additional context, complementary methodologies, or foundational insights 
                that support and extend the findings discussed here.
            <br><br>
            <strong>SylloBio-NLI: Evaluating Large Language Models on Biomedical Syllogistic Reasoning.</strong> Wysocka M, Carvalho D, Wysocki O, Valentino M, Freitas A. NAACL, 2025 –  
            <a href="https://aclanthology.org/2025.naacl-long.371/" target="_blank" style="color: black; text-decoration: underline;">Read More</a>
            <br><br>
            <strong>An LLM-based Knowledge Synthesis and Scientific Reasoning Framework for Biomedical Discovery.</strong> Wysocki O, Wysocka M, Carvalho D, Bogatu A, Miranda D, Delmas M, Unsworth H, Freitas A. ACL Anthology, 2024 – 
            <a href="https://aclanthology.org/2024.acl-demos.34/" target="_blank" style="color: black; text-decoration: underline;">Read More</a>
            <br><br>
            <strong>Relation Extraction in underexplored biomedical domains: A diversity-optimised sampling and synthetic data generation approach.</strong> Delmas M, Wysocka M, Freitas A. ACL Anthology, 2024 –  
            <a href="https://aclanthology.org/2024.cl-3.4/" target="_blank" style="color: black; text-decoration: underline;">Read More</a>
            <br><br>
            <strong>Large Language Models, scientific knowledge and factuality: A systematic analysis in antibiotic discovery.</strong> Wysocka M, Wysocki O, Delmas M, Mutel V, Freitas A. Journal of Biomedical Informatics, 2024 –  
            <a href="https://www.sciencedirect.com/science/article/pii/S1532046424001424?via%3Dihub" target="_blank" style="color: black; text-decoration: underline;">Read More</a>
            <br><br>
            <strong>A systematic review of biologically-informed deep learning models for cancer: fundamental trends for encoding and interpreting oncology data.</strong> Wysocka M, Wysocki O, Zufferey M, Landers D, Freitas A. BMC Bioinformatics, 2023 –  
            <a href="https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-023-05262-8" target="_blank" style="color: black; text-decoration: underline;">Read More</a>
            <br><br>
            <strong>Transformers and the Representation of Biomedical Background Knowledge.</strong> Wysocki O, Zhou Z, O’Regan P, Ferreira D, Wysocka M, Landers D, Freitas A. Computational Linguistics, 2022 –  
            <a href="https://direct.mit.edu/coli/article/49/1/73/113017/Transformers-and-the-Representation-of-Biomedical" target="_blank" style="color: black; text-decoration: underline;">Read More</a>
            <br><br>
            <strong>Assessing the communication gap between AI models and healthcare professionals: Explainability, utility and trust in AI-driven clinical decision-making.</strong> Wysocki O, Davies JK, Vigo M, Armstrong AC, Landers D, Lee R, Freitas A. Artificial Intelligence, 2022 –   
            <a href="https://www.sciencedirect.com/science/article/pii/S0004370222001795" target="_blank" style="color: black; text-decoration: underline;">Read More</a>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        # Logos at the bottom
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image("./images/CRUK_NBC.png", width=220)
        with col2:
            st.image("./images/CRUK_NBC_DCR.png", width=220)
        with col3:
            st.image("./images/idiap_logo.png", width=170)
        with col4:
            st.image("./images/CCE.png", width=150)
        with col5:
            st.image("./images/CCE_DART.png", width=170)
################
#
#  GRAPH COMPARISON
#
###############
###############
#
# Second part - s-s (Sample-Sample Graph Comparison)
#
###############

if (st.session_state.get('Type_of_comparison_selected', False) and 
    st.session_state.get('analysis_type') == 'sample-sample' and 
    st.session_state.get("comparison_type") == "graph"):

    sample_container = st.container()

    # Get available sample IDs from the LRP data
    sample_options = [col for col in st.session_state['filtered_data_to_graphs'].columns 
                      if col not in ["index", "source_node", "target_node"]]

    # Select Sample 1
    selected_sample1 = sample_container.selectbox(
        "Please select Sample 1:",
        sample_options,
        key="sample1"
    )

    # Exclude Sample 1 from the options for Sample 2
    sample2_options = [s for s in sample_options if s != selected_sample1]

    # Select Sample 2
    selected_sample2 = sample_container.selectbox(
        "Please select Sample 2:",
        sample2_options,
        key="sample2"
    )
    
    # Form to select the number of top edges to use for generating graphs
    compare_form = sample_container.form('Compare_sample_vs_sample')
    with compare_form:
        st.session_state['top_n'] = st.slider(
            "Please select $n$ edges with highest interaction (LRP values) to display.",
            min_value=1,
            max_value=len(st.session_state['filtered_data_to_graphs'].index),
            value=150 if len(st.session_state['filtered_data_to_graphs'].index) >= 150 else len(st.session_state['filtered_data_to_graphs'].index)
        )
        submit_button = st.form_submit_button(label='Generate graphs')
        if submit_button:
            st.session_state["Generate_graphs_button"] = True
            reset_session_state_until('G1_G2_displayed')
            print_session_state()
            #st.session_state['Calculate_button'] = False

    
    if st.session_state.get("Generate_graphs_button"):
        if selected_sample1 == selected_sample2:
            st.error("Selected samples must be distinct. Please choose two different samples.")
        else:
            # Ensure the filtered metadata index is of type string
            st.session_state["metadata_df"].data.index = st.session_state["metadata_df"].data.index.astype(str)

            # Use the already filtered & prepared LRP data
            st.session_state["LRP_to_graphs"] = st.session_state['filtered_data_to_graphs']
            st.session_state["LRP_to_graphs"].columns = st.session_state["LRP_to_graphs"].columns.astype(str)

            # Call split_and_aggregate_lrp for sample_vs_sample comparison
            st.session_state["LRP_to_graphs_stratified"] = fg.split_and_aggregate_lrp(
                st.session_state["LRP_to_graphs"],
                st.session_state["metadata_df"].data,  # Pass metadata
                comparison_type="sample_vs_sample",
                agg_func="median",
                sample1_name=selected_sample1,
                sample2_name=selected_sample2
            )

            # Subset the stratified DataFrame to retain the edge columns plus the two comparison columns
            LRP_to_graphs_stratified_sel_samples = st.session_state["LRP_to_graphs_stratified"][
                ['index', 'source_node', 'target_node'] + [selected_sample1, selected_sample2]
            ]

            # Generate graphs from the stratified data using the given top_diff_n value
            G_dict12 = fg.get_all_graphs_from_lrp(
                LRP_to_graphs_stratified_sel_samples,
                st.session_state['top_n']
            )

            
            # Display the generated graphs
            graphs = list(G_dict12.values())
            if len([selected_sample1, selected_sample2]) > 1:
                col1, col2 = sample_container.columns(2)
                for i, G in enumerate(graphs):
                    if i % 2 == 0:
                        container_topn = col1.container(border=False)
                    else:
                        container_topn = col2.container(border=False)
                    plot_my_graph(container_topn, G)
            else:
                for G in graphs:
                    container_topn = sample_container.container(border=False)
                    plot_my_graph(container_topn, G)

            # Mark the comparison as complete
            st.session_state["G1_G2_displayed"] = True
            st.session_state["G_dict12"] = G_dict12
            print_session_state()

   
###############
#
#  Second part - s-g
#
###############

if (st.session_state.get('Type_of_comparison_selected', False) and 
    st.session_state.get('analysis_type') == 'sample-group' and 
    st.session_state.get("comparison_type") == "graph"):

    group_container = st.container()

    # Use the filtered metadata for samples that passed filtering.
    filtered_metadata = st.session_state["metadata_df"].data.loc[
        st.session_state["filtered_tts_lrp_df"].index
    ]
    # Use the columns from filtered metadata for stratification.
    stratify_by_values = filtered_metadata.columns.tolist()
    if "stratify_by" not in st.session_state:
        st.session_state['stratify_by'] = stratify_by_values[0]

    def new_stratify_by_callback():
        st.session_state["stratify_by"] = st.session_state.new_stratify_by

    st.session_state["stratify_by"] = group_container.selectbox(
        "Stratify by:",
        stratify_by_values,
        index=0,
        help="Select the metadata column to stratify samples (e.g. tissue type).",
        key="new_stratify_by",
        placeholder="Select a value...",
        on_change=new_stratify_by_callback,
    )

    # Only after a stratification column is chosen, display the sample and group selection UI.
    if st.session_state["stratify_by"]:
        # Get unique group values from the filtered metadata for the chosen stratification column.
        unique_groups = sorted(list(filtered_metadata[st.session_state["stratify_by"]].unique()))

        # ---- Group Selection for Aggregation ----
        group_for_agg = group_container.selectbox(
            "Please select the group to aggregate:",
            unique_groups,
            key="group_for_agg"
        )

        # ---- Sample Selection ----
        # Get available sample IDs from the LRP data.
        # Here we assume st.session_state['filtered_data_to_graphs'] (created earlier) already contains:
        #    'index', 'source_node', 'target_node', and sample columns.
        sample_options = [col for col in st.session_state['filtered_data_to_graphs'].columns 
                          if col not in ["index", "source_node", "target_node"]]
        selected_sample = group_container.selectbox(
            "Please select a sample:",
            sample_options,
            key="sample1"
        )

        # Form to select the number of top edges to use for generating graphs.
        compare_form = group_container.form('Compare_sample_vs_group')
        with compare_form:
            st.session_state['top_n'] = st.slider(
                "Please select $n$ edges with highest interaction (LRP values) to display.",
                min_value=1,
                max_value=len(st.session_state['filtered_data_to_graphs'].index),
                value=150 if len(st.session_state['filtered_data_to_graphs'].index) >= 150 else len(st.session_state['filtered_data_to_graphs'].index)
            )
            submit_button = st.form_submit_button(label='Generate graphs')
            if submit_button:
                st.session_state["Generate_graphs_button"] = True
                reset_session_state_until('G1_G2_displayed')
                print_session_state()
                
        
        if st.session_state.get("Generate_graphs_button"):
            if not selected_sample:
                st.error("Please select a sample for comparison.")
            else:
                # Ensure the filtered metadata index is type string.
                filtered_metadata.index = filtered_metadata.index.astype(str)

                # Use the already filtered & prepared LRP data.
                # NOTE: st.session_state['filtered_data_to_graphs'] is set from:
                #       fg.prepare_lrp_to_graphs(filtered_df)
                # which has 'index', 'source_node', 'target_node' and sample IDs as columns.
                st.session_state["LRP_to_graphs"] = st.session_state['filtered_data_to_graphs']
                st.session_state["LRP_to_graphs"].columns = st.session_state["LRP_to_graphs"].columns.astype(str)

                # Call split_and_aggregate_lrp for sample_vs_group comparison.
                # This function is expected to use sample1_name and group_name parameters,
                # and it aggregates the group data, naming the column as f"{group_name}_aggregated".
                st.session_state["LRP_to_graphs_stratified"] = fg.split_and_aggregate_lrp(
                    st.session_state["LRP_to_graphs"],
                    filtered_metadata,   # use filtered metadata for stratification
                    comparison_type="sample_vs_group",
                    stratify_by=st.session_state["stratify_by"],
                    agg_func="mean",
                    sample1_name=selected_sample,
                    group_name=group_for_agg
                )

                # Subset the stratified DataFrame to retain the edge columns plus the two comparison columns:
                # one for the selected sample and one for the aggregated group.
                aggregated_col = f"{group_for_agg}_aggregated"
                LRP_to_graphs_stratified_sel_grps = st.session_state["LRP_to_graphs_stratified"][
                    ['index', 'source_node', 'target_node'] + [selected_sample, aggregated_col]
                ]

                # Generate graphs from the stratified data using the given top_diff_n value.
                G_dict12 = fg.get_all_graphs_from_lrp(
                    LRP_to_graphs_stratified_sel_grps,
                    st.session_state['top_n']
                )
                # Key modification: Save these graphs to st.session_state["G_dict12"]
                
                # Display the generated graphs.
                graphs = list(G_dict12.values())
                if len([selected_sample, aggregated_col]) > 1:
                    col1, col2 = group_container.columns(2)
                    for i, G in enumerate(graphs):
                        if i % 2 == 0:
                            container_topn = col1.container(border=False)
                        else:
                            container_topn = col2.container(border=False)
                        plot_my_graph(container_topn, G)
                else:
                    for G in graphs:
                        container_topn = group_container.container(border=False)
                        plot_my_graph(container_topn, G)
                
                st.session_state["G1_G2_displayed"] = True
                st.session_state["G_dict12"] = G_dict12
                
                print_session_state()

###############
#
#  Second part - g-g
#
###############
if (st.session_state.get('Type_of_comparison_selected', False) and 
    st.session_state.get('analysis_type') == 'group-group' and 
    st.session_state.get("comparison_type") == "graph"):

    group_container = st.container()

    # Use the filtered metadata (i.e. only for samples that passed filtering)
    filtered_metadata = st.session_state["metadata_df"].data.loc[
        st.session_state["filtered_tts_lrp_df"].index
    ]
    # Use the columns of the filtered metadata for stratification.
    stratify_by_values = filtered_metadata.columns.tolist()
    if "stratify_by" not in st.session_state:
        st.session_state['stratify_by'] = stratify_by_values[0]

    def new_stratify_by_callback():
        st.session_state["stratify_by"] = st.session_state.new_stratify_by

    st.session_state["stratify_by"] = group_container.selectbox(
        "Stratify by:",
        stratify_by_values,
        index=0,
        help="Select the metadata column to stratify samples (e.g. tissue type).",
        key="new_stratify_by",
        placeholder="Select a value...",
        on_change=new_stratify_by_callback,
    )

    # Only after a stratification column is chosen, display the group selection UI.
    if st.session_state["stratify_by"]:
        # Get the unique group values from the filtered metadata (for the chosen stratification column)
        unique_groups = sorted(list(filtered_metadata[st.session_state["stratify_by"]].unique()))

        # ---- Group Selection for Group-Group Comparison ----

        def on_group_selection_change():
            st.session_state["Generate_graphs_button"] = False
            st.session_state['Calculate_button'] = False

        
        group1 = group_container.selectbox(
            "Please select Group 1:",
            unique_groups,
            key="group1",
            on_change=on_group_selection_change
        )
        group2 = group_container.selectbox(
            "Please select Group 2:",
            unique_groups,
            key="group2",
            on_change=on_group_selection_change
        )

        # Form to select the number of top edges to use for generating graphs.
        compare_form = group_container.form('Compare')
        with compare_form:
            st.session_state['top_n'] = st.slider(
            "Please select $n$ edges with highest interaction (LRP values) to display ",
            min_value=1,
            max_value=len(st.session_state['filtered_data_to_graphs'].index),
            value=150 if len(st.session_state['filtered_data_to_graphs'].index) >= 150 else len(st.session_state['filtered_data_to_graphs'].index)
            )
            st.session_state["ready_for_comparison"] = True
            submit_button = st.form_submit_button(label='Generate graphs')
            if submit_button:
                st.session_state["Generate_graphs_button"] = True
                reset_session_state_until('G1_G2_displayed')
                print_session_state()

        # Inside the submit block
        #if submit_button:
        if st.session_state.get("Generate_graphs_button"):
            if group1 == group2:
                st.error("Selected groups must be distinct. Please choose two different groups.")
            else:
                # Ensure the filtered metadata index is of type string.
                filtered_metadata.index = filtered_metadata.index.astype(str)
                
                # Use the already filtered & prepared LRP data.
                # NOTE: st.session_state['filtered_data_to_graphs'] was set via:
                #       st.session_state['filtered_data_to_graphs'] = fg.prepare_lrp_to_graphs(filtered_df)
                # and thus has sample IDs as columns.
                st.session_state["LRP_to_graphs"] = st.session_state['filtered_data_to_graphs']
                st.session_state["LRP_to_graphs"].columns = st.session_state["LRP_to_graphs"].columns.astype(str)
                
                # Call split_and_aggregate_lrp using user-selected groups.
                st.session_state["LRP_to_graphs_stratified"] = fg.split_and_aggregate_lrp(
                    st.session_state["LRP_to_graphs"],
                    filtered_metadata,  # pass the filtered metadata instead of the full metadata
                    comparison_type="group_vs_group",
                    stratify_by=st.session_state["stratify_by"],
                    agg_func="mean",
                    group1_name=group1,
                    group2_name=group2
                )
                
                # Subset the stratified DataFrame to retain the edge columns plus the columns for the selected groups.
                LRP_to_graphs_stratified_sel_grps = st.session_state["LRP_to_graphs_stratified"][
                    ['index', 'source_node', 'target_node'] + [group1, group2]
                ]
                
                # Generate graphs from the stratified data using the given top_diff_n value.
                G_dict12 = fg.get_all_graphs_from_lrp(
                    LRP_to_graphs_stratified_sel_grps,
                    st.session_state['top_n']
                )
                
                # Display the generated graphs.
                graphs = list(G_dict12.values())
                if len([group1, group2]) > 1:
                    col1, col2 = group_container.columns(2)
                    for i, G in enumerate(graphs):
                        if i % 2 == 0:
                            container_topn = col1.container(border=False)
                        else:
                            container_topn = col2.container(border=False)
                        plot_my_graph(container_topn, G)
                else:
                    for G in graphs:
                        container_topn = group_container.container(border=False)
                        plot_my_graph(container_topn, G)
                
                st.session_state["G1_G2_displayed"] = True
                st.session_state["G_dict12"] = G_dict12
                print_session_state()

###############
#
#  graph comparisons analysis 
#
###############


import LRPgraphdiff_code as LRPgraphdiff  # module for calculating differences between graphs
import gprofiler_code as ge  # module performing gene enrichment analysis
import civic_evidence_code  
import requests
import json
import io


###############
#
#  Graph Difference Analysis
#
###############


# Activate the block if analysis type is 'group-group', 'sample-group', or 'sample-sample' and comparison subtype is "graph"
if st.session_state.get('G1_G2_displayed'):
    #and st.session_state.get('analysis_type') in ['sample-sample', 'group-group', 'sample-group'] and \
   #st.session_state.get("comparison_type") == "graph":
    print_session_state()
    st.markdown("### Graph Difference Analysis")
    st.markdown("Display a graph which edges represent the differences between two graphs. The higher the difference, the more significant (thicker) the edge.")

    # Threshold selection form
    threshold_selection_form = st.form('ThresSelection')
    with threshold_selection_form:
        col_slider, col_slider2, col_button = st.columns([3, 3, 1])
        with col_slider:
            diff_thres = st.slider(
                "Minimum difference in LRP interactions between two graphs:",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                help="Select a threshold for significant differences. The higher the threshold, the fewer edges are included in the diff.graph. If the threshold is too high, the diff.graph may be empty.",
            )
            st.session_state['diff_thres'] = diff_thres
                            
        with col_slider2:
            p_value = st.slider(
                "Gene Enrichment p-value threshold:",
                min_value=0.0,
                max_value=0.2,
                value=0.05,
                step=0.001,
                help="Select a p-value threshold for significance for Gene Enrichment analysis. The lower the p-value, the more significant the enrichment and less likely to be a false positive. The pathways and biological processes are filtered ",
            )
            st.session_state['p_value'] = p_value
                            
        with col_button:
            calculate_button = st.form_submit_button(label='Calculate')
                
        if calculate_button:
            print("Calculate button pressed.")
            print_session_state()
            st.session_state['Calculate_button'] = True
            reset_session_state_until('diff_Graphs_displayed')

        if st.session_state.get('Calculate_button'):
            # Perform graph difference analysis for the selected pair
            #if len(G_dict12) == 2:  # Ensure exactly two graphs are selected
            G_dict12 = st.session_state.get("G_dict12")
            fg.get_all_fixed_size_adjacency_matrices(G_dict12)
            #fg.get_all_fixed_size_embeddings(G_dict12)

            try:
                diff_graph_obj = LRPgraphdiff.LRPGraphDiff(G_dict12[0], G_dict12[1], diff_thres=st.session_state['diff_thres'])
            
                label1 = "Graph 1"
                label2 = "Graph 2"

                # Display the difference graph
                col_diff1, col_diff2 = st.columns(2)
                with col_diff1:
                    print(f"Displaying difference graph for {label1} vs {label2} in the first column.")
                    plot_my_graph(col_diff1, diff_graph_obj.diff_graph)
                with col_diff2:
                    print(f"Displaying communities for {label1} vs {label2} in the second column.")
                    #diff_graph.get_communitites()
                    if diff_graph_obj.diff_graph.communitites is not None and not diff_graph_obj.diff_graph.communitites.empty and "node" in diff_graph_obj.diff_graph.communitites.columns and "community" in diff_graph_obj.diff_graph.communitites.columns:
                        #st.session_state['community_analysis'] = dict(zip(diff_graph_obj.diff_graph.communitites["node"], diff_graph_obj.diff_graph.communitites["community"]))
                        # Invert the mapping: community number -> list of gene names
                        comm_df = diff_graph_obj.diff_graph.communitites
                        community_to_genes = collections.defaultdict(list)
                        for node, comm in zip(comm_df["node"], comm_df["community"]):
                            community_to_genes[comm].append(node)
                        st.session_state['community_analysis'] = dict(community_to_genes)
                    else:
                        st.session_state['community_analysis'] = {} # Ensure it's an empty dict if no communities
                    plot_my_graph(col_diff2, diff_graph_obj.diff_graph, diff_graph_obj.diff_graph.communitites)
                    print("Node communities:", diff_graph_obj.diff_graph.communitites)
                    print("Communities: ", st.session_state['community_analysis'])

                    st.session_state['diff_Graphs_displayed'] = True

                diff_graph_obj.get_edges_and_nodes_vs_threshold()


                col_chart0, col_chart1, col_chart2 = st.columns(3)
                with col_chart1:
                    print(f"Plotting 'Number of Edges vs Difference Threshold' for {label1} vs {label2}.")
                    st.subheader("Number of Edges vs Difference Threshold")
                    st.markdown(
                        "This plot shows the number of edges and nodes in the graph as a function of the difference threshold. "
                        "The higher the threshold, the fewer edges and nodes are included. Guided by this graph, you can adjust the size of the diff.graph."
                    )
                    # Use Altair for multi-line plot
                    df = diff_graph_obj.edge_node_df_sizes
                    threshold = st.session_state['diff_thres']
                    # Main line chart with custom legend labels
                    chart = alt.Chart(df).transform_fold(
                        ['num_edges', 'num_nodes'],
                        as_=['Metric', 'Count']
                    ).mark_line(point=True).encode(
                        x=alt.X('threshold:Q', title='Difference in LRP interactions between two graphs (threshold)'),
                        y=alt.Y('Count:Q', title=''),
                        color=alt.Color(
                            'Metric:N',
                            title='',
                            scale=alt.Scale(
                                domain=['num_edges', 'num_nodes'],
                                range=['#1f77b4', '#ff7f0e']
                            ),
                            legend=alt.Legend(
                                title="Metric",
                                labelExpr="{'num_edges': '# Edges', 'num_nodes': '# Nodes'}[datum.label]"
                            )
                        )
                    )

                    # Vertical dashed line for threshold
                    vline = alt.Chart(pd.DataFrame({'threshold': [threshold], 'legend': ['Selected Threshold']})).mark_rule(
                        color='black', strokeDash=[4,4]
                    ).encode(
                        x='threshold:Q',
                        detail='legend:N',
                        color=alt.value('black')
                    )

                    # Combine the main chart and the vertical line
                    final_chart = chart + vline

                    # Add a manual legend for the dashed line
                    st.altair_chart(final_chart, use_container_width=True)
            except:
                print("Defined G_dict12 but no graph difference object.")
                st.error("Error: Unable to calculate graph differences. Try lowering the difference threshold to include more edges in the difference graph.")


    ###############################################################################################################################################
    ###############################################################################################################################################

# Add "Analyse evidence" button


#col_button, col_slider, col_slider2  = st.columns([1, 1, 1])

#with col_button:
    #analyse_evidence = st.form_submit_button(label='Analyse evidence')
# If the button is clicked, set a persistent flag in session state.
#if analyse_evidence:
#    st.session_state['Analyse_evidence_button'] = True
#    reset_session_state_until('Evidence_analysis_done')
#    print_session_state()

#if st.session_state.get('diff_Graphs_displayed'):
    #if st.button("Analyse evidence"):#, key="Analyse_evidence_button"):
    #    print("Analyse_evidence_button clicked.")
    #st.session_state["Analyse_evidence_button"] = True
    #reset_session_state_until('Evidence_analysis_done')
    #print_session_state()



# Używamy właściwego atrybutu node_names_no_type dla wybranej pary
#diff_graph = LRPgraphdiff.LRPGraphDiff(G_dict12[0], G_dict12[1], diff_thres=st.session_state['diff_thres'])
###############################################################################################################################################
if st.session_state['diff_Graphs_displayed']:
    st.divider()
    st.subheader("Evidence Analysis:")
    st.markdown("""
            This section provides comprehensive information about the evidence related to the genes identified in the difference graph.

            We integrate multiple sources of biological and clinical knowledge to support the interpretation of molecular interaction signatures:
            - **Gene Enrichment Analysis** is performed using [g:Profiler](https://biit.cs.ut.ee/gprofiler/gost), which identifies statistically significant biological pathways, Gene Ontology terms, and functional categories enriched in your gene set.
            - **CIViC (Clinical Interpretation of Variants in Cancer)** provides curated clinical evidence for cancer-related genes and variants, including summaries, descriptions, molecular profiles, and supporting literature ([link](https://civicdb.org/evidence/home)).
            - **PharmaGKB** offers pharmacogenomics knowledge, linking genes to drug responses, clinical annotations, and relevant literature ([link](https://www.pharmgkb.org/)).

            These integrated resources help you understand the biological and clinical relevance of the genes and interactions highlighted by the difference graph, supporting biomarker discovery and hypothesis generation.
            """)
    
    # Gene list 

    st.session_state['gene_list'] = diff_graph_obj.diff_graph.G.nodes
    st.session_state['gene_list_no_types'] = list(set(diff_graph_obj.diff_graph.node_names_no_type))
    print("Gene List:", st.session_state['gene_list'])
    print("Gene List no types:", st.session_state['gene_list_no_types'])
    # show the gene list
    st.subheader("Gene List:")
    with st.expander("See Gene List"):
        st.write(st.session_state['gene_list'] )

    ###############################################################################################################################################
    # Communities
    
    st.subheader("Groups of nodes interacting (communities):")
    with st.expander("See groups (communities)"):
        st.write(st.session_state['community_analysis'] )
    ###############################################################################################################################################
    # Analiza gene enrichment for all nodes
    # Analiza gene enrichment for all nodes
    ge_analyser = ge.GE_Analyser(st.session_state['gene_list_no_types'])
    ge_results = ge_analyser.run_GE_on_nodes(user_threshold=st.session_state['p_value'])  # dostosuj próg, jeśli potrzeba
    st.session_state['gene_enrichment'] = ge_analyser.ge_results_dict

    st.subheader("Gene Enrichment for All Genes")
    with st.expander("See details of Gene Enrichment for All Genes"):
        if ge_results is not None and not ge_results.empty:
            # Select columns to display (adjust as needed)
            columns_to_show = ["name", "description", "intersections", "p_value","native"]
            columns_to_show = [col for col in columns_to_show if col in ge_results.columns]
            st.dataframe(ge_results[columns_to_show], use_container_width=True)
        else:
            st.write("No Gene Enrichment Results available.")
            st.session_state['gene_enrichment'] = None
    
    # Gene enrichment analysis for each community
    community_enrichment_dfs = {}
    community_enrichment_results = {}

    for community_id, gene_list in st.session_state['community_analysis'].items():
        if gene_list:  # Only run if the community has genes
            gene_list = set([gene.split('_')[0] for gene in gene_list])  # Remove type suffix)
            print(f"Running gene enrichment for community {community_id} with genes: {gene_list}")
            ge_analyser = ge.GE_Analyser(gene_list)
            ge_results = ge_analyser.run_GE_on_nodes(user_threshold=st.session_state['p_value'])
            community_enrichment_results[community_id] = ge_analyser.ge_results_dict
            community_enrichment_results[community_id]['gene_list'] = ", ".join(gene_list)
            community_enrichment_dfs[community_id] = ge_results

    st.session_state['community_enrichment'] = community_enrichment_results

    # Display results in the dashboard
    st.subheader("Gene Enrichment for Each Community")
    with st.expander("See details of Community Gene Enrichment"):
        for community_id, ge_results in community_enrichment_dfs.items():
            genes_in_community = ", ".join(st.session_state['community_analysis'][community_id])
            st.markdown(f"### Community {community_id}")
            st.markdown(f"Nodes: {genes_in_community}")
            if ge_results is not None and not ge_results.empty:
                # Select columns to display (adjust as needed)
                columns_to_show = ["name", "description", "intersections", "p_value","native"]
                # Only show columns that exist in the DataFrame
                columns_to_show = [col for col in columns_to_show if col in ge_results.columns]
                st.dataframe(ge_results[columns_to_show], use_container_width=True)
            else:
                st.write("No enrichment results for this community.")
        
    #############################################################################################################################################  
    # CIVIC Evidence Analysis
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        civicdb_path = os.path.join(base_path, 'resources', 'civicdb')

        if not os.path.exists(civicdb_path):
            st.error(f"CIVIC database path does not exist: {civicdb_path}")
            st.session_state['civic_evidence'] = None
        else:
            analyzer = civic_evidence_code.CivicEvidenceAnalyzer(civicdb_path, st.session_state['gene_list_no_types'])
            analyzer.create_feature_details_dict()
            details_dict = analyzer.add_evidence_to_dict()
            st.session_state['civic_evidence'] = details_dict

            st.subheader("CIVICdb Knowledge:")
            with st.expander("See details of CIVICdb Knowledge"):
                if details_dict:
                    for feature, feature_dict in details_dict.items():
                        st.markdown(f"### {feature}")
                        # Use tabs for displaying details
                        tab_sum, tab_desc, tab_mp, tab_ev = st.tabs([
                            "See Summary",
                            "See Description",
                            "See Molecular Profiles",
                            "See Evidence"
                        ])
                        with tab_sum:
                            st.write(feature_dict.get("Summary", "No Summary available."))
                        with tab_desc:
                            st.write(feature_dict.get("Description", "No Description available."))
                        with tab_mp:
                            st.write(feature_dict.get("Molecular_profiles", "No Molecular Profiles available."))
                        with tab_ev:
                            # Try "Evidence", if missing then try lower-case "pmid"
                            evidence = feature_dict.get("Evidence")
                            if evidence is None:
                                evidence = feature_dict.get("pmid", "No Evidences available.")
                            # If evidence is a list of dicts, display as table
                            if isinstance(evidence, list) and len(evidence) > 0 and isinstance(evidence[0], dict):
                                evidence_df = pd.DataFrame(evidence)
                                # Make pmid column clickable if it exists
                                if "pmid" in evidence_df.columns:
                                    def make_link(pmid):
                                        if pd.notnull(pmid) and str(pmid).strip() != "":
                                            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                                            return f"[{pmid}]({url})"
                                        return ""
                                    evidence_df["pmid"] = evidence_df["pmid"].apply(make_link)
                                # Display as markdown table to preserve links
                                st.write(evidence_df.to_markdown(index=False), unsafe_allow_html=True)
                            else:
                                st.write(evidence)
                else:
                    st.write("No CIVIC evidence details available.")
                    st.session_state['civic_evidence'] = None
    except Exception as e:
        st.error(f"An error occurred during CIVIC Evidence Analysis: {e}")
        st.session_state['civic_evidence'] = None

    #############################################################################################################################################
    # pharmGKB Analysis
    try:
        pharmagkb_files_path = os.path.join(base_path, 'resources', 'pharmgkb')

        if not os.path.exists(pharmagkb_files_path):
            st.error(f"PharmaGKB files path does not exist: {pharmagkb_files_path}")
            st.session_state['pharmGKB_analysis'] = None
        else:
            pharmGKB_analyzer = pbk.pharmGKB_Analyzer(st.session_state['gene_list_no_types'])
            pharmGKB_analyzer.get_pharmGKB_knowledge(files_path=pharmagkb_files_path)

            # Combine all filtered data into one DataFrame
            pharmGKB_df = pd.concat([
                pharmGKB_analyzer.pharmGKB_var_pheno_ann_filtered,
                pharmGKB_analyzer.pharmGKB_var_drug_ann_filtered,
                pharmGKB_analyzer.pharmGKB_var_fa_ann_filtered
            ], ignore_index=True)

            if pharmGKB_df.empty:
                st.write("No pharmGKB evidence details available.")
                st.session_state['pharmGKB_analysis'] = None
            else:
                # Ensure required columns exist
                required_columns = ['Gene', 'Drug(s)', 'Sentence', 'Notes', 'PMID']
                missing_columns = [col for col in required_columns if col not in pharmGKB_df.columns]
                if missing_columns:
                    st.error(f"Missing required columns in pharmGKB data: {', '.join(missing_columns)}")
                else:
                    # Group results by gene in the requested format (list of dicts per gene)
                    pharmGKB_details = {}
                    grouped = pharmGKB_df.groupby('Gene')
                    for gene, group in grouped:
                        # Convert each row to a dictionary, keeping column names as keys
                        gene_entries = []
                        for _, row in group.iterrows():
                            entry = {
                                "Drug(s)": row.get("Drug(s)", None),
                                "Sentence": row.get("Sentence", None),
                                "Notes": row.get("Notes", None) if pd.notnull(row.get("Notes", None)) else None,
                                "PMID": int(row["PMID"]) if pd.notnull(row.get("PMID", None)) and str(row["PMID"]).isdigit() else row.get("PMID", None)
                            }
                            gene_entries.append(entry)
                        pharmGKB_details[gene] = gene_entries
                    st.session_state['pharmGKB_analysis'] = pharmGKB_details
                    st.subheader("pharmGKB Knowledge:")
                    # Convert PMID values to PubMed URLs
                    # Convert PMID values to PubMed URLs and display as clickable links
                    pharmGKB_df['PMID'] = pharmGKB_df['PMID'].astype(str).apply(
                        lambda pmid: f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid and pmid != 'nan' else ""
                    )
                    with st.expander("See details of pharmGKB Knowledge"):
                        for gene, group in pharmGKB_df.groupby('Gene'):
                            st.markdown(f"### {gene}")
                            # Prepare table with clickable PMID links
                            gene_table = group[["Drug(s)", "Sentence", "Notes", "PMID"]].reset_index(drop=True)
                            # Create a copy for display with clickable links
                            display_table = gene_table.copy()
                            def make_link(url):
                                if url:
                                    pmid = url.rstrip('/').split('/')[-1]
                                    return f"[{pmid}]({url})"
                                return ""
                            display_table['PMID'] = display_table['PMID'].apply(make_link)
                            st.write(display_table.to_markdown(index=False), unsafe_allow_html=True)
                            # Optionally, keep the tabs for more detailed view
                            
                    
    except Exception as e:
        st.error(f"An error occurred during pharmGKB Analysis: {e}")
        st.session_state['pharmGKB_analysis'] = None

    if (st.session_state.get('gene_enrichment') is not None or st.session_state.get('civic_evidence') is not None or st.session_state.get('pharmGKB_analysis') is not None):
        st.session_state['Evidence_analysis_done'] = True
    print_session_state()
                
#########################################################################################################
#### AI Assistant button
######################################################################################################### 
# Check if at least one of the analyses has produced results, then display the AI Assistant button.
#st.session_state["ai_assistant_shown"] = False

#if st.session_state['Evidence_analysis_done']:
    
    # If the button is clicked, set a persistent flag in session state.
#    if st.button("Interpret results with AI Assistant", key="ai_assistant_button_graphdiff"):
#        print("AI Assistant button clicked.")
#        st.session_state["AI_assistance_button"] = True

# If the flag is set (or already set), always display the AI Assistant UI.
#
if st.session_state.page == "AI Assistant":
    context_input = st.text_area(
            "Context",
            placeholder="Enter context here...",
            key="ai_context"
        )
    prompt_input = st.text_area(
        "Question/Prompt",
        placeholder="Enter your question or prompt here...",
        key="ai_prompt"
    )
    print("Retrieved text areas; Context:", st.session_state.get("ai_context"),
            "Question/Prompt:", st.session_state.get("ai_prompt"))


    gene_enrichment = st.session_state.get('gene_enrichment')
    community_enrichment = st.session_state.get('community_enrichment')
    civic_evidence = st.session_state.get('civic_evidence')
    pharmGKB_analysis = st.session_state.get('pharmGKB_analysis')

    print("Gene Enrichment:", gene_enrichment)
    print("Community Enrichment:", community_enrichment)
    print("civic_evidence Analysis:", civic_evidence)
    print("pharmGKB_analysis:", pharmGKB_analysis)

    output_dict = {'Context': st.session_state.get("ai_context", ""),
                    'Prompt': st.session_state.get("ai_prompt", ""),
                    'Gene Enrichment': gene_enrichment,
                    'Community Enrichment': community_enrichment,
                    'CIVIC Evidence': civic_evidence,
                    'pharmGKB Analysis': pharmGKB_analysis}

    print("Output dictionary:", output_dict)
    # Display the output dictionary in the dashboard in a readable format
    #st.markdown("### AI Assistant Input (Dictionary View):")
    def convert_np(obj):
        if isinstance(obj, dict):
            return {k: convert_np(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_np(i) for i in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    
    #########################################################################################################
    # Send the output_dict to n8n webhook

    # Add a button to send the JSON payload to the n8n webhook endpoint
    send_to_n8n = st.button("Analyse with AI", key="send_to_n8n_button")

    payload = convert_np(output_dict)
    with st.expander("See json sent to AI", expanded=False):
        st.json(payload)

    #
    # Add a button to download the payload as a JSON file

    payload_json = json.dumps(payload, indent=2)
    st.download_button(
        label="Download JSON",
        data=io.BytesIO(payload_json.encode("utf-8")),
        file_name="analysis_output.json",
        mime="application/json"
    )


    if send_to_n8n:
        print('Sending JSON to AI')
        

        # Define the n8n webhook endpoint URL (replace with your actual endpoint)
        n8n_webhook_url = "http://localhost:5678/webhook-test/ac8a4b13-ca15-4511-9663-39ab520132ca"


        # Send the JSON payload to the n8n webhook endpoint
        try:
            response = requests.post(
                n8n_webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                st.success("Data sent successfully to n8n webhook.")
            else:
                st.error(f"Failed to send data to n8n webhook. Status code: {response.status_code}")
        except Exception as e:
            st.error(f"Error sending data to n8n webhook: {e}")

    # Display a text area with mockup AI output (placeholder)
    st.markdown("### AI Assistant Output")
    ai_output = st.text_area(     
        label = ' ',   
        value="This is a mockup output from the AI analyser. The results of your analysis will appear here.",
        height=200,
        key="ai_output_mockup"
    )






            
 
# ##########################################
# #### AI Assistant Sidebar 
# ##########################################               
# if st.session_state.page == "AI Assistant":
    
#     # Text area for additional context
#     context_input = st.text_area("Context", 
#                                  placeholder="Enter context here...", 
#                                  key="ai_context",
#                                  value=st.session_state.get("ai_context", ""))
    
#     # Text area for Question/Prompt
#     prompt_input = st.text_area("Question/Prompt", 
#                                 placeholder="Enter your question or prompt here...", 
#                                 key="ai_prompt",
#                                 value=st.session_state.get("ai_prompt", ""))
    
#     print("Displayed AI Assistant page with context:", st.session_state.get("ai_context"),
#           "and prompt:", st.session_state.get("ai_prompt"))
    
#     ##########################################################
#     # Keyword Options: Either upload frequent keywords or select keywords
#     ##########################################################
#     st.markdown("### Keyword Options:")
#     keyword_option = st.radio("Choose keyword option:",
#                               ("Upload Frequent Keywords", "Keyword Selection"),
#                               key="keyword_option")
#     print("Keyword option chosen:", st.session_state.get("keyword_option"))
    
#     if keyword_option == "Upload Frequent Keywords":
#         # File uploader for frequent keywords (CSV or TXT)
#         keyword_file = st.file_uploader("Upload Frequent Keywords", type=["csv", "txt"], key="keyword_file")
#         if keyword_file:
#             try:
#                 # Read keywords as CSV (assuming one keyword per row)
#                 keywords_df = pd.read_csv(keyword_file, header=None)
#                 uploaded_keywords = keywords_df[0].tolist()
#                 st.session_state["uploaded_keywords"] = uploaded_keywords
#                 st.success("Frequent keywords uploaded successfully (CSV)!")
#                 st.write("Uploaded frequent keywords:", uploaded_keywords)
#                 print("Frequent keywords loaded from CSV:", uploaded_keywords)
#             except Exception as e_csv:
#                 # If CSV reading fails, try reading as plain text
#                 keyword_file_str = keyword_file.getvalue().decode("utf-8")
#                 uploaded_keywords = [line.strip() for line in keyword_file_str.splitlines() if line.strip()]
#                 st.session_state["uploaded_keywords"] = uploaded_keywords
#                 st.success("Frequent keywords uploaded successfully (TEXT)!")
#                 st.write("Uploaded frequent keywords:", uploaded_keywords)
#                 print("Frequent keywords loaded from text:", uploaded_keywords)
#         else:
#             print("No frequent keywords file uploaded.")
#     else:  # Keyword Selection branch using the frequent keywords list from Analyse page
#         if st.session_state.get("frequent_kws") is not None and not st.session_state["frequent_kws"].empty:
#             # Use the first column of the uploaded frequent keywords CSV
#             uploaded_list = st.session_state["frequent_kws"][0].tolist()
#             selected_keywords = st.multiselect("Please select your keyword:", 
#                                                  uploaded_list, 
#                                                  key="selected_keywords")
#             st.session_state["selected_keywords"] = selected_keywords
#             st.write("Selected keywords:", selected_keywords)
#             print("Selected keywords from uploaded list:", selected_keywords)
#         else:
#             st.warning("No frequent keywords available. Please upload frequent keywords in the Analyse page.")
#             print("No frequent keywords available.")
    
#     # Ensure that one of the options is provided before proceeding.
#     provided_keywords = (st.session_state.get("uploaded_keywords")
#                            if st.session_state.get("keyword_option") == "Upload Frequent Keywords"
#                            else st.session_state.get("selected_keywords"))
    
#     if not provided_keywords:
#         st.warning("Please provide frequent keywords either by uploading a file or by selecting from the uploaded list to proceed.")