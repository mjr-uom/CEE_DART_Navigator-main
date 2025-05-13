import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import seaborn as sns
import gravis as gv
import altair as alt
import networkx as nx
import matplotlib as mt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from pyvis.network import Network
import os
import hashlib
import itertools
import importlib, sys
import matplotlib.pyplot as plt
from gptAgent import OpenAIAgent

from len_gen import generate_legend_table, generate_legend_table_community
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

import pingouin as pg

import pharmabk_code as pbk
importlib.reload(pbk)

default_session_state = {
    'first_form_completed': False,
    'second_form_completed': False,
    'enable_comparison': False,
    'G_dict': {},
    'keywords': list(),
    'tumor_tissue_site': list(),
    'ttss_selected': list(),
    'acronym': list(),
    'lrp_df': pd.DataFrame([]), # raw loaded data
    'filtered_tts_lrp_df': pd.DataFrame([]),
    'metadata_df': pd.DataFrame([]), # raw loaded data
    'civic_data': pd.DataFrame([]),
    'filtered_df': pd.DataFrame([]),  # Initialize as an empty DataFrame
    'f_tumor_tissue_site': pd.DataFrame([]),
    'f_acronym': pd.DataFrame([]),
    'filters_form_completed': None,
    'tts_filter_button': None,
    'frequent_kws': pd.DataFrame([]), # raw loaded data
    'LRP_to_graphs_stratified': pd.DataFrame([]),
    'calculate_button': False,
    'comparison_grp_button': False,
    'compare_form_complete': False,
    'top_n': 150,
    'top_diff_n': 0,
    'top_n_similar': None,
    'compare_grp_selected': list(),
    'enable_chat_bot': False,
    'messages': list(),
    'context_input': None,
    'awaiting_context': True,
    'user_input': "",
    'disabled': False,
    'openai_model': "gpt-3.5-turbo",
    'edge_df': pd.DataFrame([]),
    'all_facts': "",
    'page': "Home",
    'analysis_type': None,
    'ready_for_comparison': False,
    'sample_comparison_type': None,
    'group_comparison_type': None,
    'filtered_data': pd.DataFrame([]),
    'stratify_by': None,
    'new_stratify_by': None,
    'group_for_agg': None,
    'sample1': None,
    'sample2': None,
    'group1': None,
    'group2': None,
    'sg_grouping_column': None,
    'sg_group_options': list(),
    'gg_grouping_column': None,
    'gg_group_options': list(),
    'LRP_to_graphs': pd.DataFrame([]),
    'G_dict12': {},
    'stratify_by_values': list(),
    'diff_thres': 0.5,
    'p_value': 0.05,
    'pharmakb_details': {},
}

for key, default_value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

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
        'first_form_completed', 'analysis_type', 'stratify_by', 'second_form_completed',
        'enable_comparison', 'filters_form_completed', 'tts_filter_button', 'calculate_button',
        'comparison_grp_button', 'compare_form_complete', 'top_n', 'top_diff_n',
        'top_n_similar', 'compare_grp_selected','ready_for_comparison',
        'sample_comparison_type', 'group_comparison_type', 'group_for_agg',
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
    def format_edge_data(newG):
        edge_dist = list(nx.get_edge_attributes(newG, 'LRP_norm').values())
        norm_colour = mt.colors.Normalize(vmin=0.0, vmax=max(edge_dist), clip=True)
        colour_mapper = cm.ScalarMappable(norm=norm_colour, cmap=cm.Greys)
        for u, v, data in newG.edges(data=True):
            lrp_norm = data.get("LRP_norm", 0)
            data.update({
                "hover": f'LRP_norm: {lrp_norm:.4f}',
                "color": mt.colors.rgb2hex(colour_mapper.to_rgba(lrp_norm)),
                "weight": lrp_norm,
                "value": lrp_norm,
                "click": f'LRP_norm: {lrp_norm:.4f}'
            })

    # Helper function: Format node attributes
    def format_node_data(newG, pos, neighbor_map, node_community_mapping, group_color_mapper, node_color_mapper):
        for node, data in newG.nodes(data=True):
            x, y = pos[node]
            label = "_".join(node.split('_')[:-1])
            node_type = node.split('_')[-1]
            neighbors = neighbor_map[node]

            # Update node attributes
            data.update({
                "x": x, "y": y, "label": label, "_type": node_type,
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
    pos = nx.kamada_kawai_layout(newG, weight="LRP_norm", scale=500)

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
    container.markdown(f"<h2 style='{style_heading}'>Top '{graph.top_n_edges}' edges with the highest LRP values</h2>", unsafe_allow_html=True)

    # Create and display graph visualization
    fig = gv.d3(
        newG,
        use_node_size_normalization=True, node_size_data_source='value', node_hover_tooltip=True,
        node_hover_neighborhood=True, show_node_label=True, node_label_data_source='label',
        use_edge_size_normalization=True, edge_size_data_source='weight', edge_curvature=0.0,
        edge_hover_tooltip=True, zoom_factor=0.55
    )

    with container:
        subCol1, subCol2 = st.columns([5, 1])
        with subCol1:
            components.html(fig.to_html(), height=620, scrolling=True)
        with subCol2:
            components.html(chart_legend_css, height=620)


def create_multiselect(catalog_name: str, values: list, container: object):
    with container:
        selected_values = st.multiselect("Please select \"" + catalog_name + "\" : ",
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
    if st.sidebar.button('Analyse'):
        st.session_state.page = "Analyse"
    if st.sidebar.button('AI Assistant'):  # New tab
        st.session_state.page = "AI Assistant"
    if st.sidebar.button('FAQ'):
        st.session_state.page = "FAQ"
    if st.sidebar.button('About'):
        st.session_state.page = "About"
    if st.sidebar.button('News'):
        st.session_state.page = "News"
    if st.sidebar.button('Related Papers'):  # New tab
        st.session_state.page = "Related Papers"


    # Render content based on the current page state
    if st.session_state.page == "Home":
        #st.title("Home")
        # Logo in the top left corner (on the home page)
        st.image("./images/MIS_Portal_logo.png", width=300)  # Logo w lewym górnym rogu
        # Main frame with text + buttons inside
        st.markdown("""
    <div style="background-color:#f0f0f0; padding:20px; border-radius:10px; 
                margin-top:20px; text-align:center; font-size:18px;">
        <strong>The Portal Analyzes Molecular Interaction Signatures</strong><br>
        <br>
        The <strong>public version</strong> of the MIS portal provides a <strong>framework</strong> for comparing biological samples based on molecular interaction signatures. 
        Using deep learning metrics, statistical tests, and graph-based methods, it identifies key interaction patterns and assesses their biological relevance for <strong>biomarker discovery and hypothesis generation.</strong>
        <br><br>
        <button onclick="window.location.href = window.location.href.split('?')[0] + '?page=Analyse'; window.location.reload();" style="display:inline-block; background:white; color:#0078D4; 
            border: 2px solid #0078D4; padding:10px 20px; text-decoration:none; border-radius:5px; margin-right:10px;">Analyse Your Samples</button>
        <button onclick="window.location.href='?page=Examples'" style="display:inline-block; background:white; color:#0078D4; 
            border: 2px solid #0078D4; padding:10px 20px; text-decoration:none; border-radius:5px;">See Examples</button>
    </div>
""", unsafe_allow_html=True)
        # Break before next frames
        st.markdown("<br>", unsafe_allow_html=True)
        # Two columns next to each other
        col1, col2 = st.columns([1, 1])
        # First column (left)
        with col1:
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)  # Center the logo
            st.image("./images/evidence.png", width=50)  # Logo above the frame
            st.markdown("</div>", unsafe_allow_html=True)
            # Frame with text and button
            st.markdown(
                """
                <div style="background-color:#e0e0e0; padding:20px; border-radius:10px; 
                            text-align:center; font-size:16px; padding:20px;">
                    <strong>The Portal uses distinct levels of supporting evidence</strong><br>
                    Molecular Interaction Signatures (MIS) are derived from the analysis of molecular profiles and are annotated by a comprehensive set of knowledgebases and computational estimations.
                    <br><br>
                    <button onclick="window.location.href='?page=FAQ'" style="display:inline-block; background:white; color:#0078D4; 
                        border: 2px solid #0078D4; padding:10px 20px; text-decoration:none; border-radius:5px;">Read more</button>
                </div>
                """,
                unsafe_allow_html=True
            )
        # Second column (right)
        with col2:
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)  # Center the logo
            st.image("./images/expert.png", width=50)  # Logo above the frame
            st.markdown("</div>", unsafe_allow_html=True)
            # Frame with text and button
            st.markdown(
                """
                <div style="background-color:#e0e0e0; padding:20px; border-radius:10px; 
                            text-align:center; font-size:16px; padding:20px;">
                    <strong>The Portal follows clinical expert consensus</strong><br>
                    The MIS portal is designed to support the interpretation of molecular interaction signatures in the context of clinical expert consensus developed under the Cancer Core Europe umbrella and the latest scientific evidence.
                    <br><br>
                    <button onclick="window.location.href='?page=About'" style="display:inline-block; background:white; color:#0078D4; 
                        border: 2px solid #0078D4; padding:10px 20px; text-decoration:none; border-radius:5px;">About us</button>
                </div>
                """,
                unsafe_allow_html=True
            )
        # Text below the columns
        st.markdown("""
            <div style="margin-top:30px; text-align:justify; font-size:16px;">
                <p><strong>This is an open-access version of the Molecular Interaction Signatures Portal, designed for the comparative study of biological samples based on molecular interaction signatures.</strong></p>
                <p>The portal utilizes computational methods, including deep learning-derived relevance metrics and statistical analyses, with references to the applied algorithms and data sources provided in the results. 
                Some resources integrated within the portal may require a license for commercial applications or clinical use; therefore, this version is strictly limited to academic research. 
                Users must accept these terms upon first login, which requires a valid email address. When using this portal, please cite:</p>
            </div>
        """, unsafe_allow_html=True)
        # Four logos in one row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.image("./images/CRUK_NBC.png", width=220)  

        with col2:
            st.image("./images/CRUK_NBC_DCR.png", width=220)  

        with col3:
            st.image("./images/CCE.png", width=150)  

        with col4:
            st.image("./images/CCE_DART.png", width=170)   

    elif st.session_state.page == "Analyse":
        st.title("Analyse Molecular Interaction Signatures")
        st.markdown("""
            <div style="background-color:#f0f0f0; padding:20px; border-radius:10px; 
                        margin-top:20px; text-align:center; font-size:18px;">
                <strong>Analyse Your Samples</strong><br>
                <br>
                Compare and interpret biological samples based on molecular interaction signatures according to multiple evidence sources.
                Upload your LRP-based Interaction Metrics data and metadata to start analyzing your samples based on molecular interaction signatures.
                <br><br>
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
                        File {path_to_LRP_data.name} has been analyzed.
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
                        File {path_to_metadata.name} has been analyzed.
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
                st.markdown("### Data Filtering")
                # Create widgets for filtering based on metadata columns
                filters = {}
                for key, values in filter_catalog.items():
                    filters[key] = create_multiselect(key, values, combined_form)

                st.markdown("### Upload Frequent Keywords")
                # File uploader widget for frequent keywords
                path_to_plkeywords = st.file_uploader("Upload frequent keywords")
                if path_to_plkeywords is not None:
                    temp_dir = tempfile.gettempdir()
                    st.session_state['frequent_kws'] = pd.read_csv(
                        save_my_uploaded_file(temp_dir, path_to_plkeywords), header=None
                    )
                    st.info('File {0} has been analysed.'.format(path_to_plkeywords.name))

                st.markdown("### Keyword Selection")
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
                    "Please select your keyword: ",
                    keywords,
                    default_keywords,
                    placeholder="Select one or more options (optional)"
                )
                st.session_state['keywords'] = keywords_selected
                # A single "Run" button to execute both filtering steps
                run_button = combined_form.form_submit_button(label='Run')

            # Execute filtering when the "Run" button is clicked
            if run_button:
                # Filter data based on the selected metadata filters
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
                    st.session_state['filtered_df'] = st.session_state['filtered_tts_lrp_df']
                else:
                    st.session_state['filtered_df'] = fg.filter_columns_by_keywords(
                        st.session_state['filtered_tts_lrp_df'], st.session_state['keywords']
                    )

                # Check if the filtered DataFrame is empty
                if st.session_state['filtered_df'].empty:
                    st.error("The filtered dataset is empty. Please select different keywords or filters.")
                else:
                    # Prepare the data for graph generation
                    st.session_state['filtered_data'] = fg.prepare_lrp_to_graphs(st.session_state['filtered_df'])
                    st.session_state['first_form_completed'] = True
        
                # Filter data based on the selected keywords
                #if not keywords_selected:
                #    keywords_selected = None
                
                #st.session_state['filtered_df'] = fg.filter_columns_by_keywords(st.session_state['filtered_tts_lrp_df'], st.session_state['keywords'])
                #st.session_state['filtered_data'] = fg.prepare_lrp_to_graphs(st.session_state['filtered_df'])
                #st.session_state['first_form_completed'] = True
                #st.success("Keywords filtered successfully! Proceed to the next step.")
        # After the filtering stage has completed
        if st.session_state.get('first_form_completed', False):
            # In the "Analyse" branch, after filtering has completed and analysis type buttons are shown:
            st.markdown("### Type of analysis")
            col_a, col_b, col_c = st.columns(3)
            if col_a.button("sample-sample"):
                st.session_state["analysis_type"] = "sample-sample"
                st.session_state["ready_for_comparison"] = False
            if col_b.button("sample-group"):
                st.session_state["analysis_type"] = "sample-group"
                st.session_state["ready_for_comparison"] = False
            if col_c.button("group-group"):
                st.session_state["analysis_type"] = "group-group"
                st.session_state["ready_for_comparison"] = False
                # Add visual feedback
            if "analysis_type" in st.session_state:
                if st.session_state["analysis_type"] == "sample-sample":
                    col_a.markdown("**Selected: sample-sample**")
                elif st.session_state["analysis_type"] == "sample-group":
                    col_b.markdown("**Selected: sample-group**")
                elif st.session_state["analysis_type"] == "group-group":
                    col_c.markdown("**Selected: group-group**")

            
            # If the selected analysis type is sample-sample, display the two sub-buttons always.
            if st.session_state.get("analysis_type") == "sample-sample":
                st.markdown("### Sample-vs-Sample Comparison")
                # Display sub-buttons for comparison type:
                col_uni, col_graph = st.columns(2)
                if col_uni.button("Univariable Comparison"):
                    st.session_state["sample_comparison_type"] = "univariable"
                if col_graph.button("Graph Comparison"):
                    st.session_state["sample_comparison_type"] = "graph"

                if st.session_state.get("sample_comparison_type") == "univariable":
                    with st.form("sample_vs_sample_form"):
                        if 'filtered_df' not in st.session_state:
                            st.session_state['filtered_df'] = pd.DataFrame([])
                        
                        if st.session_state["filtered_df"].empty:
                            st.error("Filtered dataset is empty. Please review your filtering criteria.")
                            st.session_state.sample_names = []
                        else:
                            st.session_state.sample_names = list(st.session_state["filtered_df"].index)
                        
                        if st.session_state.sample_names:
                            # Remove the on_change callback from the selectbox: recalc options manually below.
                            sample1 = st.selectbox("Select Sample 1", st.session_state.sample_names, key="sample1")
                            # Recalculate sample2_options based on sample1
                            st.session_state.sample2_options = [s for s in st.session_state.sample_names if s != sample1]
                            if not st.session_state.sample2_options:
                                st.error("Not enough distinct samples available for comparison.")
                                sample2 = None
                            else:
                                sample2 = st.selectbox("Select Sample 2", st.session_state.sample2_options, key="sample2", index=0)
                            top_n_features = st.slider("Select Top N Features", min_value=1, max_value=100, value=10)
                            singles = sorted({col.split("_")[-1] for col in st.session_state["filtered_df"].columns if "_" in col})
                            pairs = [f"{t1}-{t2}" for i, t1 in enumerate(singles) for t2 in singles[i:]]
                            available_node_types = sorted(list(set(singles + pairs)))
                            selected_node_types = st.multiselect("Select Edge Type(s)", available_node_types,
                                                                default=[available_node_types[0]] if available_node_types else [])
                            compare_submit = st.form_submit_button("Compare Samples")
                        else:
                            compare_submit = False

                    if compare_submit:
                        if sample1 == sample2 or sample2 is None:
                            st.error("Selected samples must be distinct. Please choose two different samples for comparison.")
                        else:
                            print("Filtered DataFrame Index:", st.session_state["filtered_df"].index)
                            if sample1 not in st.session_state["filtered_df"].index or sample2 not in st.session_state["filtered_df"].index:
                                st.error(f"One or both selected samples ({sample1}, {sample2}) are not present in the filtered dataset. Please select valid samples.")
                            else:
                                save_plot = False  # Adjust as needed
                                comparison = lrpcomp.SampleVsSampleComparison(
                                    sample1_name=sample1,
                                    sample2_name=sample2,
                                    data_df=st.session_state["filtered_df"],
                                    clinical_features_df=st.session_state["metadata_df"].data
                                )
                                comparison.compute_boxplot_values()
                                comparison.select_top_n_by_column(column='median_abs_diff', n=top_n_features, ascending=False)
                                pattern = '|'.join(selected_node_types) if isinstance(selected_node_types, list) else selected_node_types
                                _ = comparison.filter_and_merge_data(selected_type=pattern)
                                fig = comparison.plot_scatter(selected_type=pattern, save_plot=save_plot)
                                st.pyplot(fig)

                elif st.session_state.get("sample_comparison_type") == "graph":
                    # Placeholder for Graph Comparison analysis.
                    st.markdown("### Graph Comparison for Sample vs Sample")
            # If the selected analysis type is sample-group, display the two sub-buttons always.
            if st.session_state.get("analysis_type") == "sample-group":
                st.markdown("### Sample-vs-Group Comparison")
                # Display sub-buttons for comparison type:
                col_uni2, col_graph2 = st.columns(2)
                if col_uni2.button("Univariable Comparison"):
                    st.session_state["group_comparison_type"] = "univariable"
                if col_graph2.button("Graph Comparison"):
                    st.session_state["group_comparison_type"] = "graph"
                
                if st.session_state.get("group_comparison_type") == "univariable":
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
                        sample_names = list(st.session_state["filtered_df"].index)
                        sample1 = st.selectbox("Select Sample", sample_names, key="sample_vs_group_sample")
                        # Use the updated group options from filtered metadata.
                        chosen_group = st.selectbox("Select Group", st.session_state["sg_group_options"],
                                                    key="group_selected")
                        top_n_features = st.slider("Select Top N Features", min_value=1, max_value=100, value=10)
                        # Determine available node types (both single and paired).
                        singles = sorted({col.split("_")[-1] for col in st.session_state["filtered_df"].columns if "_" in col})
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
                            data_df=st.session_state["filtered_df"],
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
                        st.pyplot(fig)
                
                elif st.session_state.get("group_comparison_type") == "graph":
                    st.markdown("### Graph Comparison for Sample vs Group")
            # If the selected analysis type is group-group, display the two sub-buttons
            if st.session_state.get("analysis_type") == "group-group":
                st.markdown("### Group-vs-Group Comparison")
                # Display sub-buttons for comparison type:
                col_uni2, col_graph2 = st.columns(2)
                if col_uni2.button("Univariable Comparison"):
                    st.session_state["group_comparison_type"] = "univariable"
                if col_graph2.button("Graph Comparison"):
                    st.session_state["group_comparison_type"] = "graph"

                if st.session_state.get("group_comparison_type") == "univariable":
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
                        singles = sorted({col.split("_")[-1] for col in st.session_state["filtered_df"].columns if "_" in col})
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
                            common_index = st.session_state["filtered_df"].index.intersection(filtered_sel_metadata.index)
                            filtered_sel_metadata = filtered_sel_metadata.loc[common_index]
                            data_df = st.session_state["filtered_df"].loc[common_index]
                            
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
                            st.pyplot(fig)

                elif st.session_state.get("group_comparison_type") == "graph":
                    st.markdown("### Graph Comparison for Group vs Group")
    
    elif st.session_state.page == "AI Assistant":
        st.title("AI Assistant")
        st.markdown("""
            <div style="background-color:#f0f0f0; padding:20px; border-radius:10px; 
                        margin-top:20px; text-align:justify; font-size:16px;">
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


    elif st.session_state.page == "About":
        st.title("About Us")
        st.markdown("""
            <div style="background-color:#f0f0f0; padding:20px; border-radius:10px; 
                        margin-top:20px; text-align:center; font-size:18px;">
                <strong>About the Molecular Interaction Signatures Portal</strong><br>
                <br>
                The Molecular Interaction Signatures (MIS) Portal is a comprehensive platform designed to analyze and compare biological samples based on molecular interaction signatures. 
                Our portal leverages advanced computational methods, including deep learning metrics, statistical tests, and graph-based techniques, to identify key interaction patterns and assess their biological relevance.
                <br><br>
                Our mission is to facilitate biomarker discovery and hypothesis generation by providing researchers with a robust framework for analyzing molecular profiles. 
                The portal integrates data from various knowledgebases and computational estimations to offer a detailed and insightful analysis of molecular interactions.
                <br><br>
                <strong>Contact Us:</strong><br>
                For more information, please visit our website or contact us at info@misportal.org.
            </div>
        """, unsafe_allow_html=True)

    elif st.session_state.page == "News":
        st.title("News")
        st.markdown("""
            <div style="background-color:#f0f0f0; padding:20px; border-radius:10px; 
                        margin-top:20px; text-align:center; font-size:18px;">
                <strong>Latest News</strong><br>
                <br>
                Stay updated with the latest news and updates about the Molecular Interaction Signatures Portal.
                <br><br>
                Check back regularly for new features, updates, and announcements.
            </div>
        """, unsafe_allow_html=True)
        
    elif st.session_state.page == "Related Papers":
        st.title("Related Papers")
        st.markdown("""
            <div style="background-color:#f0f0f0; padding:20px; border-radius:10px; 
                        margin-top:20px; text-align:justify; font-size:16px;">
                <strong>Related Papers</strong><br>
                The following publications are closely related to the research presented in this work. 
                They provide additional context, complementary methodologies, or foundational insights 
                that support and extend the findings discussed here.
            </div>
            <br>
            <ul>
                <li><a href="https://aclanthology.org/2024.acl-demos.34/" target="_blank">Wysocki et al., ACL 2024</a></li>
                <li><a href="https://aclanthology.org/2025.naacl-long.371/" target="_blank">Wysocka et al., NAACL 2025</a></li>
            </ul>
        """, unsafe_allow_html=True)
################
#
#  Second part - s-s
#
###############
###############
#
# Second part - s-s (Sample-Sample Graph Comparison)
#
###############

if (st.session_state.get('first_form_completed', False) and 
    st.session_state.get('analysis_type') == 'sample-sample' and 
    st.session_state.get("sample_comparison_type") == "graph"):

    sample_container = st.container()

    # Get available sample IDs from the LRP data
    sample_options = [col for col in st.session_state['filtered_data'].columns 
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
        st.session_state['top_diff_n'] = st.slider(
            "Please select the number of top n edges",
            min_value=1,
            max_value=len(st.session_state['filtered_data'].index),
            value=len(st.session_state['filtered_data'].index) // 2
        )
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        if selected_sample1 == selected_sample2:
            st.error("Selected samples must be distinct. Please choose two different samples.")
        else:
            # Ensure the filtered metadata index is of type string
            st.session_state["metadata_df"].data.index = st.session_state["metadata_df"].data.index.astype(str)

            # Use the already filtered & prepared LRP data
            st.session_state["LRP_to_graphs"] = st.session_state["filtered_data"]
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
                st.session_state['top_diff_n']
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
            st.session_state["compare_form_complete"] = True
            st.session_state["G_dict12"] = G_dict12
            st.session_state["ready_for_comparison"] = True
            print_session_state()

   
###############
#
#  Second part - s-g
#
###############

if (st.session_state.get('first_form_completed', False) and 
    st.session_state.get('analysis_type') == 'sample-group' and 
    st.session_state.get("group_comparison_type") == "graph"):

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
        # Here we assume st.session_state['filtered_data'] (created earlier) already contains:
        #    'index', 'source_node', 'target_node', and sample columns.
        sample_options = [col for col in st.session_state['filtered_data'].columns 
                          if col not in ["index", "source_node", "target_node"]]
        selected_sample = group_container.selectbox(
            "Please select a sample:",
            sample_options,
            key="sample1"
        )

        # Form to select the number of top edges to use for generating graphs.
        compare_form = group_container.form('Compare_sample_vs_group')
        with compare_form:
            st.session_state['top_diff_n'] = st.slider(
                "Please select the number of top n edges",
                min_value=1,
                max_value=len(st.session_state['filtered_data'].index),
                value=len(st.session_state['filtered_data'].index) // 2
            )
            submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            if not selected_sample:
                st.error("Please select a sample for comparison.")
            else:
                # Ensure the filtered metadata index is type string.
                filtered_metadata.index = filtered_metadata.index.astype(str)

                # Use the already filtered & prepared LRP data.
                # NOTE: st.session_state['filtered_data'] is set from:
                #       fg.prepare_lrp_to_graphs(filtered_df)
                # which has 'index', 'source_node', 'target_node' and sample IDs as columns.
                st.session_state["LRP_to_graphs"] = st.session_state["filtered_data"]
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
                    st.session_state['top_diff_n']
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
                
                st.session_state["compare_form_complete"] = True
                st.session_state["G_dict12"] = G_dict12
                st.session_state["ready_for_comparison"] = True
                print_session_state()

###############
#
#  Second part - g-g
#
###############
if (st.session_state.get('first_form_completed', False) and 
    st.session_state.get('analysis_type') == 'group-group' and 
    st.session_state.get("group_comparison_type") == "graph"):

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
        group1 = group_container.selectbox(
            "Please select Group 1:",
            unique_groups,
            key="group1"
        )
        group2 = group_container.selectbox(
            "Please select Group 2:",
            unique_groups,
            key="group2"
        )

        # Form to select the number of top edges to use for generating graphs.
        compare_form = group_container.form('Compare')
        with compare_form:
            st.session_state['top_diff_n'] = st.slider(
                "Please select the number of top n edges",
                min_value=1,
                max_value=len(st.session_state['filtered_data'].index),
                value=len(st.session_state['filtered_data'].index) // 2
            )
            submit_button = st.form_submit_button(label='Submit')

        # Inside the submit block
        if submit_button:
            if group1 == group2:
                st.error("Selected groups must be distinct. Please choose two different groups.")
            else:
                # Ensure the filtered metadata index is of type string.
                filtered_metadata.index = filtered_metadata.index.astype(str)
                
                # Use the already filtered & prepared LRP data.
                # NOTE: st.session_state['filtered_data'] was set via:
                #       st.session_state['filtered_data'] = fg.prepare_lrp_to_graphs(filtered_df)
                # and thus has sample IDs as columns.
                st.session_state["LRP_to_graphs"] = st.session_state["filtered_data"]
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
                    st.session_state['top_diff_n']
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
                st.session_state["compare_form_complete"] = True
                st.session_state["G_dict12"] = G_dict12
                st.session_state["ready_for_comparison"] = True
                print_session_state()

###############
#
#  graph comparisons analysis 
#
###############


import LRPgraphdiff_code as LRPgraphdiff  # moduł do obliczania różnic między grafami
import gprofiler_code as ge  # moduł realizujący analizę gene enrichment
import civic_evidence_code  


###############
#
#  Graph Difference Analysis
#
###############

# Ensure G_dict12 is available in session state
#if "G_dict12" not in st.session_state:
#    st.session_state["G_dict12"] = {}
#G_dict12 = st.session_state["G_dict12"]

# Activate the block if analysis type is 'group-group', 'sample-group', or 'sample-sample' and comparison subtype is "graph"
if st.session_state.get('ready_for_comparison'):
    #and st.session_state.get('analysis_type') in ['sample-sample', 'group-group', 'sample-group'] and \
   #st.session_state.get("group_comparison_type") == "graph":
    print_session_state()
    st.markdown("### Graph Difference Analysis")

    # Retrieve generated graphs from session state
    G_dict12 = st.session_state.get("G_dict12")
    if not G_dict12:
        st.error("No graphs available for difference analysis. Please run the graph generation step first.")
    else:
        # Perform graph difference analysis
        fg.get_all_fixed_size_adjacency_matrices(G_dict12)
        fg.get_all_fixed_size_embeddings(G_dict12)

        # Calculate adjacency differences for each pair of graphs
        pairs = list(itertools.combinations(range(len(G_dict12)), 2))
        adj_diff_list = []
        for i, j in pairs:
            diff = fg.calculate_adjacency_difference(G_dict12[i], G_dict12[j])
            adj_diff_list.append(diff)

        # Threshold selection form
        threshold_selection_form = st.form('ThresSelection')
        with threshold_selection_form:
            col_slider, col_slider2, col_button = st.columns([3, 3, 1])
            with col_slider:
                diff_thres = st.slider(
                    "LRP difference threshold value:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    help="Select a threshold for significant differences."
                )
                                
            with col_slider2:
                p_value = st.slider(
                    "p-value threshold:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.05,
                    step=0.01,
                    help="Select a p-value threshold for significance."
                )
                                
            with col_button:
                calculate_button = st.form_submit_button(label='Calculate')
                
        if calculate_button:
            print("Calculate button pressed.")
            print_session_state()

            # Perform graph difference analysis for the selected pair
            if len(G_dict12) == 2:  # Ensure exactly two graphs are selected
                fg.get_all_fixed_size_adjacency_matrices(G_dict12)
                fg.get_all_fixed_size_embeddings(G_dict12)

                # Calculate adjacency difference for the selected pair
                diff = fg.calculate_adjacency_difference(G_dict12[0], G_dict12[1])

                edge_df = fg.create_edge_dataframe_from_adj_diff(diff, diff_thres)
                print("Edge DataFrame Columns:", edge_df.columns)
                if edge_df.empty:
                    st.subheader("The edges representing graph differences are not above the threshold to plot.")
                    print("No edges above the threshold to plot.")
                else:
                    label1 = "Graph 1"
                    label2 = "Graph 2"

                    diff_graph = lrpgraph.LRPGraph(
                        edges_sample_i=edge_df,
                        source_column="source_node",
                        target_column="target_node",
                        edge_attrs=["LRP", "LRP_norm"],
                        top_n_edges=st.session_state.get('top_diff_n', 10),
                        sample_ID=f"DIFFERENCE {label1} vs {label2}"
                    )

                    # Display the difference graph
                    col_diff1, col_diff2 = st.columns(2)
                    with col_diff1:
                        print(f"Displaying difference graph for {label1} vs {label2} in the first column.")
                        plot_my_graph(col_diff1, diff_graph)
                    with col_diff2:
                        print(f"Displaying communities for {label1} vs {label2} in the second column.")
                        diff_graph.get_communitites()
                        plot_my_graph(col_diff2, diff_graph, diff_graph.communitites)

                    edge_df_sizes = []
                    for th_val in np.arange(0.0, 1.0, 0.02):
                        tmp_edge_df = fg.create_edge_dataframe_from_adj_diff(diff, th_val)
                        edge_df_sizes.append((th_val, len(tmp_edge_df)))

                    col_chart1, col_chart2 = st.columns(2)
                    with col_chart1:
                        print(f"Plotting 'Number of Edges vs Difference Threshold' for {label1} vs {label2}.")
                        st.subheader("Number of Edges vs Difference Threshold")
                        tmp_df = pd.DataFrame(edge_df_sizes, columns=["x", "y"])
                        st.vega_lite_chart(
                            tmp_df,
                            {
                                "mark": {"type": "line", "point": True},
                                "encoding": {
                                    "x": {"field": "x", "type": "quantitative", "title": "Difference Threshold"},
                                    "y": {"field": "y", "type": "quantitative", "title": "Number of Edges"}
                                },
                                "selection": {"brush": {"type": "interval", "bind": "scales"}}
                            },
                            use_container_width=True
                        )
                    with col_chart2:
                        print(f"Plotting 'LRP Values for Edges Sorted by LRP' for {label1} vs {label2}.")
                        st.subheader("LRP Values for Edges Sorted by LRP")
                        if 'LRP' not in edge_df.columns and 'LRP_norm' in edge_df.columns:
                            edge_df = edge_df.rename(columns={'LRP_norm': 'LRP'})
                        st.line_chart(edge_df['LRP'], x_label='Edge #', y_label='LRP')
            else:
                st.error("Graph difference analysis requires exactly two graphs. Please select a valid pair.")        
            st.session_state["enable_chat_bot"] = True
            st.divider()

            # Po wyświetleniu wszystkich grafów różnic i wykresów, wykonaj analizę gene enrichment dla wybranej pary:
            if G_dict12 and len(G_dict12) == 2:  # Upewnij się, że mamy dokładnie dwa grafy
                # Używamy właściwego atrybutu node_names_no_type dla wybranej pary
                diff_graph = LRPgraphdiff.LRPGraphDiff(G_dict12[0], G_dict12[1], diff_thres=st.session_state['diff_thres'])
                gene_list = diff_graph.diff_graph.node_names_no_type

                # Analiza gene enrichment
                ge_analyser = ge.GE_Analyser(gene_list)
                ge_results = ge_analyser.run_GE_on_nodes(user_threshold=st.session_state['p_value'])  # dostosuj próg, jeśli potrzeba

                if ge_results is not None and not ge_results.empty:
                    st.subheader("Gene Enrichment Results:")
                    with st.expander("See Gene Enrichment Results"):
                        # Dla każdego wyniku z analizy gene enrichment
                        for idx, row in ge_results.iterrows():
                            st.markdown(f"### Enriched Term: {row['name']}")
                            # Tworzymy zakładki odpowiadające informacjom ze słownika wyniku
                            tab_term, tab_desc, tab_inter, tab_pval = st.tabs([
                                "Term", "Description", "Intersection Size", "p-value"
                            ])
                            with tab_term:
                                st.write(row["name"])
                            with tab_desc:
                                st.write(row["description"])
                            with tab_inter:
                                st.write(row.get("intersection_size", "N/A"))
                            with tab_pval:
                                st.write(row.get("p_value", "N/A"))
                else:
                    st.write("No Gene Enrichment Results available.")
            else:
                st.error("Gene enrichment analysis requires exactly two graphs. Please ensure a valid pair is selected.")
                            # Now run CIVIC evidence analysis using functions from civic_evidence_code
                                # CIVIC evidence analysis using functions from civic_evidence_code
                # Now run CIVIC evidence analysis using functions from civic_evidence_code
                # Now run CIVIC evidence analysis using functions from civic_evidence_code
                import os
            # CIVIC Evidence Analysis
            try:
                base_path = os.path.dirname(os.path.abspath(__file__))
                civicdb_path = os.path.join(base_path, 'resources', 'civicdb')

                if not os.path.exists(civicdb_path):
                    st.error(f"CIVIC database path does not exist: {civicdb_path}")
                else:
                    analyzer = civic_evidence_code.CivicEvidenceAnalyzer(civicdb_path, gene_list)
                    analyzer.create_feature_details_dict()
                    details_dict = analyzer.add_evidence_to_dict()

                    st.subheader("CIVIC Evidence Knowledge:")
                    with st.expander("See CIVIC Evidence Knowledge"):
                        if details_dict:
                            for feature, feature_dict in details_dict.items():
                                st.markdown(f"### {feature}")
                                # Use tabs for displaying details
                                tab_desc, tab_sum, tab_mp, tab_ev = st.tabs([
                                    "See Description",
                                    "See Summary",
                                    "See Molecular Profiles",
                                    "See Evidences"
                                ])
                                with tab_desc:
                                    st.write(feature_dict.get("Description", "No Description available."))
                                with tab_sum:
                                    st.write(feature_dict.get("Summary", "No Summary available."))
                                with tab_mp:
                                    st.write(feature_dict.get("Molecular_profiles", "No Molecular Profiles available."))
                                with tab_ev:
                                    st.write(feature_dict.get("Evidence", "No Evidences available."))
                        else:
                            st.write("No CIVIC evidence details available.")
            except Exception as e:
                st.error(f"An error occurred during CIVIC Evidence Analysis: {e}")

            # PharmaKB Analysis
            try:
                pharmagkb_files_path = os.path.join(base_path, 'resources', 'pharmgkb')

                if not os.path.exists(pharmagkb_files_path):
                    st.error(f"PharmaGKB files path does not exist: {pharmagkb_files_path}")
                else:
                    pharmakb_analyzer = pbk.Pharmakb_Analyzer(gene_list)
                    pharmakb_analyzer.get_pharmakb_knowledge(files_path=pharmagkb_files_path)

                    # Combine all filtered data into one DataFrame
                    pharmakb_df = pd.concat([
                        pharmakb_analyzer.pharmakb_var_pheno_ann_filtered,
                        pharmakb_analyzer.pharmakb_var_drug_ann_filtered,
                        pharmakb_analyzer.pharmakb_var_fa_ann_filtered
                    ], ignore_index=True)

                    if pharmakb_df.empty:
                        st.write("No PharmaKB evidence details available.")
                    else:
                        # Ensure required columns exist
                        required_columns = ['Gene', 'Drug(s)', 'Sentence', 'Notes', 'PMID']
                        missing_columns = [col for col in required_columns if col not in pharmakb_df.columns]
                        if missing_columns:
                            st.error(f"Missing required columns in PharmaKB data: {', '.join(missing_columns)}")
                        else:
                            # Group results by gene
                            pharmakb_details = {}
                            grouped = pharmakb_df.groupby('Gene')
                            for gene, group in grouped:
                                drugs = list(group['Drug(s)'].unique())
                                statement = "; ".join(group['Sentence'].astype(str).unique())
                                notes = "; ".join(group['Notes'].astype(str).unique())
                                pmids = list(group['PMID'].unique())
                                pharmakb_details[gene] = {
                                    "Drug(s)": drugs if drugs else "No Drugs available.",
                                    "Statement": statement if statement else "No Statement available.",
                                    "Notes": notes if notes else "No Notes available.",
                                    "PMID": pmids if pmids else "No PMID available."
                                }

                            st.subheader("PharmaKB Knowledge:")
                            with st.expander("See PharmaKB Knowledge"):
                                for gene, details in pharmakb_details.items():
                                    st.markdown(f"### {gene}")
                                    # Use tabs for displaying details
                                    tab_drug, tab_statement, tab_notes, tab_pmids = st.tabs([
                                        "Drug(s)",
                                        "Statement",
                                        "Notes",
                                        "PMID"
                                    ])
                                    with tab_drug:
                                        st.write(details["Drug(s)"])
                                    with tab_statement:
                                        st.write(details["Statement"])
                                    with tab_notes:
                                        st.write(details["Notes"])
                                    with tab_pmids:
                                        st.write(details["PMID"])
            except Exception as e:
                st.error(f"An error occurred during PharmaKB Analysis: {e}")