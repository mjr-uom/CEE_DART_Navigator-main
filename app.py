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

default_session_state = {
    'first_form_completed': False,
    'second_form_completed': False,
    'enable_comparison': False,
    'G_dict': {},
    'keywords': list(),
    'tumor_tissue_site': list(),
    'ttss_selected': list(),
    'acronym': list(),
    'lrp_df': pd.DataFrame([]),
    'filtered_tts_lrp_df': pd.DataFrame([]),
    'metadata_df': pd.DataFrame([]),
    'civic_data': pd.DataFrame([]),
    'f_tumor_tissue_site': pd.DataFrame([]),
    'f_acronym': pd.DataFrame([]),
    'filters_form_completed': None,
    'tts_filter_button': None,
    'frequent_kws': pd.DataFrame([]),
    'LRP_to_graphs_stratified': pd.DataFrame([]),
    'calculate_button': False,
    'comparison_grp_button': False,
    'compare_form_complete': False,
    'top_n': 0,
    'top_diff_n': 0,
    'top_n_similar': None,
    'compare_grp_selected': list(),
    'enable_chat_bot' : False,
    'messages' : list(),
    'context_input' : None,
    'awaiting_context' : True,
    'user_input' : "",
    'disabled' : False,
    'openai_model' : "gpt-3.5-turbo",
    'edge_df' : pd.DataFrame([]),
    'all_facts' : "" ,
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


def save_my_uploaded_file(path, uploaded_file):
    repository_folder = path
    save_path = os.path.join(repository_folder, uploaded_file.name)
    with open(save_path, mode='wb') as w:
        w.write(uploaded_file.getvalue())
    return save_path

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
    if st.sidebar.button('FAQ'):
        st.session_state.page = "FAQ"
    if st.sidebar.button('About'):
        st.session_state.page = "About"
    if st.sidebar.button('News'):
        st.session_state.page = "News"


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
                st.session_state['lrp_df'] = dtl.LRPData(file_path=save_my_uploaded_file('/tmp', path_to_LRP_data),
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
                st.session_state['metadata_df'] = dtl.MetaData(file_path=save_my_uploaded_file('/tmp', path_to_metadata),
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
                    st.session_state['frequent_kws'] = pd.read_csv(
                        save_my_uploaded_file('/tmp', path_to_plkeywords), header=None
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

                # A single "Run" button to execute both filtering steps
                run_button = combined_form.form_submit_button(label='Run')

            # Execute filtering when the "Run" button is clicked
            if run_button:
                # Filter data based on the selected metadata filters
                desired_barcodes = st.session_state['metadata_df'].data
                for filter_name, values in filters.items():
                    if len(values) > 0:
                        filter_query = filter_name + ' == [ ' + ', '.join(f'"{i}"' for i in values) + ' ]'
                        desired_barcodes = desired_barcodes.query(filter_query)
                valid_bc = list(set(desired_barcodes.index.tolist()))
                st.session_state['filtered_tts_lrp_df'] = st.session_state['lrp_df'].filter(items=valid_bc, axis=0)
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
                    keywords_selected = None
                filtered_df = fg.filter_columns_by_keywords(st.session_state['filtered_tts_lrp_df'], keywords_selected)
                st.session_state['filtered_data'] = fg.prepare_lrp_to_graphs(filtered_df)
                st.session_state['first_form_completed'] = True
                #st.success("Keywords filtered successfully! Proceed to the next step.")
        # After the filtering stage has completed
        if st.session_state.get('first_form_completed', False):
            # In the "Analyse" branch, after filtering has completed and analysis type buttons are shown:
            st.markdown("### Type of analysis")
            col_a, col_b, col_c = st.columns(3)
            if col_a.button("sample-sample"):
                st.session_state["analysis_type"] = "sample-sample"
            if col_b.button("sample-group"):
                st.session_state["analysis_type"] = "sample-group"
            if col_c.button("group-group"):
                st.session_state["analysis_type"] = "group-group"

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
                        # Use filtered samples if available; otherwise fall back to full data.
                        if not st.session_state["filtered_tts_lrp_df"].empty:
                            sample_names = list(st.session_state["filtered_tts_lrp_df"].index)
                        else:
                            sample_names = list(st.session_state["lrp_df"].index)
                        sample1 = st.selectbox("Select Sample 1", sample_names, key="sample1")
                        # Exclude the selected sample1 for sample2 selection.
                        available_sample2 = [s for s in sample_names if s != sample1]
                        if not available_sample2:
                            st.error("Not enough distinct samples available for comparison.")
                        else:
                            sample2 = st.selectbox("Select Sample 2", available_sample2, key="sample2")
                        top_n_features = st.slider("Select Top N Features", min_value=1, max_value=100, value=10)
                        # Determine available node types (both single and paired).
                        singles = sorted({col.split("_")[-1] for col in st.session_state["lrp_df"].columns if "_" in col})
                        pairs = [f"{t1}-{t2}" for i, t1 in enumerate(singles) for t2 in singles[i:]]
                        available_node_types = sorted(list(set(singles + pairs)))
                        selected_node_types = st.multiselect("Select Edge Type(s)", available_node_types,
                                                            default=[available_node_types[0]] if available_node_types else [])
                        compare_submit = st.form_submit_button("Compare Samples")
                    
                    if compare_submit:
                        if sample1 == sample2:
                            st.error("Selected samples must be distinct. Please choose two different samples for comparison.")
                        else:
                            save_plot = False  # Adjust as needed
                            comparison = lrpcomp.SampleVsSampleComparison(
                                sample1_name=sample1,
                                sample2_name=sample2,
                                data_df=st.session_state["lrp_df"],
                                clinical_features_df=st.session_state["metadata_df"].data
                            )
                            # Compute boxplot values and select top features.
                            comparison.compute_boxplot_values()
                            comparison.select_top_n_by_column(column='median_abs_diff', n=top_n_features, ascending=False)
                            pattern = '|'.join(selected_node_types) if isinstance(selected_node_types, list) else selected_node_types
                            _ = comparison.filter_and_merge_data(selected_type=pattern)
                            fig = comparison.plot_scatter(selected_type=pattern, save_plot=save_plot)
                            st.pyplot(fig)
                elif st.session_state.get("sample_comparison_type") == "graph":
                    # Placeholder for Graph Comparison analysis.
                    st.markdown("### Graph Comparison not yet implemented.")
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
                        sample_names = list(st.session_state["lrp_df"].index)
                        sample1 = st.selectbox("Select Sample", sample_names, key="sample_vs_group_sample")
                        # Use the updated group options from filtered metadata.
                        chosen_group = st.selectbox("Select Group", st.session_state["sg_group_options"],
                                                    key="group_selected")
                        top_n_features = st.slider("Select Top N Features", min_value=1, max_value=100, value=10)
                        # Determine available node types (both single and paired).
                        singles = sorted({col.split("_")[-1] for col in st.session_state["lrp_df"].columns if "_" in col})
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
                            data_df=st.session_state["lrp_df"],
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
                    st.markdown("### Graph Comparison not yet implemented for Sample vs Group")
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
                        singles = sorted({col.split("_")[-1] for col in st.session_state["lrp_df"].columns if "_" in col})
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
                            common_index = st.session_state["lrp_df"].index.intersection(filtered_sel_metadata.index)
                            filtered_sel_metadata = filtered_sel_metadata.loc[common_index]
                            data_df = st.session_state["lrp_df"].loc[common_index]
                            
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
                    with st.form("group_vs_group_form_graph"):
                        st.markdown("### Graph Comparison for Group vs Group")
                        st.info("Graph Comparison is not yet implemented for Group vs Group.")
                        dummy_submit = st.form_submit_button("Submit (dummy)")
                    
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

################
if st.session_state.get('first_form_completed', False):
    Col1, Col2, Col3, Col4, Col5 = st.tabs(
        ["፨ selected sample", "⩬ top n similar (under dev.)", "🔍 compare groups", "⚖️ show differences", "🤖 AI interpreter"])
    st.session_state['enable_comparison'] = True
    node_selection_form = Col1.form('TopNSelection')
    with node_selection_form:
        st.session_state['top_n'] = node_selection_form.slider(
            "Please select the number of top n edges",
            min_value=1,
            max_value=len(st.session_state['filtered_data'].index),
            value=len(st.session_state['filtered_data'].index) // 2
        )
        submit_button = node_selection_form.form_submit_button(label='Submit')
        if submit_button:
            G_dict = fg.get_all_graphs_from_lrp(st.session_state['filtered_data'], st.session_state['top_n'])
            # Validation
            assert len(G_dict[1].G.edges) == st.session_state['top_n'], "Edge count mismatch."
            fg.get_all_fixed_size_adjacency_matrices(G_dict)
            assert len(G_dict[2].all_nodes) == np.shape(G_dict[1].fixed_size_adjacency_matrix)[
                1], "Node count mismatch."
            fg.get_all_fixed_size_embeddings(G_dict)

            st.session_state['second_form_completed'] = True
            st.session_state['G_dict'] = G_dict
            node_selection_form.success('{0} graphs have been generated.'.format(len(G_dict)))

if st.session_state['second_form_completed']:

    sampleIDs = []
    for i in range(len(st.session_state['G_dict'])):
        sampleIDs.append(st.session_state['G_dict'][i].sample_ID)

    disp_list = sampleIDs.copy()
    disp_list.sort()

    if "selected_gId" not in st.session_state:
        st.session_state['selected_gId'] = sampleIDs[0]


    def new_gid_callback():
        st.session_state["selected_gId"] = st.session_state.new_gId


    st.session_state["selected_gId"] = Col1.selectbox("Please select the sample you want to see:",
                                                      disp_list,
                                                      index=0,
                                                      help="Choose from the available graphs listed below.",
                                                      key="new_gId",
                                                      placeholder="Select a graph...",
                                                      on_change=new_gid_callback,
                                                      )

    if st.session_state["selected_gId"]:
        print("You selected index: {0}".format(disp_list.index(st.session_state["selected_gId"])))
        G = st.session_state['G_dict'][map_index_to_unsorted(disp_list.index(st.session_state["selected_gId"]), disp_list, sampleIDs)]
        container_main = Col1.container(border=False)
        plot_my_graph(container_main, G)

    # Get the top n most similar samples
    G = st.session_state['G_dict'][map_index_to_unsorted(disp_list.index(st.session_state["selected_gId"]), disp_list, sampleIDs)]
    embeddings_df = fg.extract_raveled_fixed_size_embedding_all_graphs(st.session_state['G_dict'])
    sorted_distance_df = fg.compute_sorted_distances(embeddings_df, G.sample_ID)


    def new_top_n_similar_callback():
        st.session_state["top_n_similar"] = st.session_state.new_top_n_similar


    st.session_state["top_n_similar"] = Col2.number_input(
        "Please provide the number of similar graphs to display:",
        min_value=1,
        max_value=6,
        step=1,
        key="new_top_n_similar",
        placeholder="Select a value...",
        on_change=new_top_n_similar_callback
    )
    if st.session_state["top_n_similar"] > 0:
        top_n_samples = sorted_distance_df.head(st.session_state["top_n_similar"] + 1)
        Col2_subC_1, Col2_subC_2 = Col2.columns(2)
        for i in range(st.session_state["top_n_similar"] + 1):
            sample_ID = top_n_samples.iloc[i, 0]
            G = next(G for G in st.session_state['G_dict'].values() if G.sample_ID == sample_ID)
            if i % 2:
                container_topn = Col2_subC_2.container(border=False)
            else:
                container_topn = Col2_subC_1.container(border=False)
            plot_my_graph(container_topn, G)

###############
#
#  Second part
#
###############
if st.session_state.get('first_form_completed', False):
    stratify_by_values = st.session_state['metadata_df'].data.columns.tolist()
    if "stratify_by" not in st.session_state:
        st.session_state['stratify_by'] = stratify_by_values[0]


    def new_stratify_by_callback():
        st.session_state["stratify_by"] = st.session_state.new_stratify_by


    st.session_state["stratify_by"] = Col3.selectbox("Stratify by:",
                                                     stratify_by_values,
                                                     index=None,
                                                     help="\"Stratify by criteria.\"",
                                                     key="new_stratify_by",
                                                     placeholder="Select a value...",
                                                     on_change=new_stratify_by_callback,
                                                     )

    if st.session_state["stratify_by"]:
        st.session_state["LRP_to_graphs_stratified"] = fg.split_and_aggregate_lrp(
            st.session_state['filtered_data'],
            st.session_state['metadata_df'].data,
            st.session_state["stratify_by"],
            agg_func="mean")

######<-----here
    filtered_columns = [col for col in st.session_state["LRP_to_graphs_stratified"].columns
                        if (isinstance(col, str) and not any(
            substr in col for substr in ["index", "source_node", "target_node"])) or isinstance(col, int)]

    compare_form = Col3.form('Compare')
    with (compare_form):
        if len(filtered_columns):
            st.session_state["compare_grp_selected"] = compare_form.multiselect("Please select your comparison groups: ",
                                                                                filtered_columns,
                                                                                [],
                                                                                placeholder="Choose a comparison group.")

            st.session_state['top_diff_n'] = compare_form.slider("Please select the number of top n edges",
                                                                 min_value=1,
                                                                 max_value=len(st.session_state['filtered_data'].index),
                                                                 value=len(st.session_state['filtered_data'].index) // 2
                                                                 )
        else:
            compare_form.warning("Unfortunately, your selection criteria are not generating any comparison group.")

        st.session_state["comparison_grp_button"]  = compare_form.form_submit_button(label='Submit')

        if st.session_state["comparison_grp_button"] and st.session_state["compare_grp_selected"]:
            LRP_to_graphs_stratified_sel_grps = st.session_state["LRP_to_graphs_stratified"][
                ['index', 'source_node', 'target_node'] + st.session_state["compare_grp_selected"]]

            G_dict12 = fg.get_all_graphs_from_lrp(LRP_to_graphs_stratified_sel_grps,
                                                  st.session_state['top_diff_n'])
            if len(st.session_state["compare_grp_selected"]) > 1:
                Col3_subC_1, Col3_subC_2 = Col3.columns(2)
                for i in range(len(G_dict12)):
                    if not i % 2:
                        container_topn = Col3_subC_1.container(border=False)
                    else:
                        container_topn = Col3_subC_2.container(border=False)
                    G = G_dict12[i]
                    plot_my_graph(container_topn, G)
            else:
                for i in range(len(G_dict12)):
                    container_topn = Col3.container(border=False)
                    G = G_dict12[i]
                    plot_my_graph(container_topn, G)
            st.session_state["compare_form_complete"] = True
###############
#  comparisons
#
###############
if st.session_state.get('enable_comparison', False):
    if len(st.session_state["compare_grp_selected"]) < 2:
        Col4.subheader(
            "Regrettably, your selection criteria are not generating a sufficient number of comparison groups.",
            divider=True)
    else:
        LRP_to_graphs_stratified_sel_grps = st.session_state["LRP_to_graphs_stratified"][
            ['index', 'source_node', 'target_node'] + st.session_state["compare_grp_selected"]]

        G_dict12 = fg.get_all_graphs_from_lrp(LRP_to_graphs_stratified_sel_grps,
                                              st.session_state['top_diff_n'])

        fg.get_all_fixed_size_adjacency_matrices(G_dict12)
        fg.get_all_fixed_size_embeddings(G_dict12)

        adj = []
        for i in range(len(G_dict12)):
            adj.append(G_dict12[i].fixed_size_adjacency_matrix)

        pairs = list(itertools.combinations(range(len(G_dict12)), 2))
        adj_diff_list = []
        for i,j in pairs:
            adj_diff_list.append(fg.calculate_adjacency_difference(G_dict12[i], G_dict12[j]))

        threshold_selection_form = Col4.form('ThresSelection')
        with threshold_selection_form:
            diff_thres = st.slider("Threshold value:",
                                   min_value=0.0,
                                   max_value=1.0,
                                   value=0.5,
                                   help="\"Threshold value.\""
                                   )

            calculate_button = st.form_submit_button(label='Calculate')
            if calculate_button:
                pair_counter = 0
                for adj_diff in adj_diff_list:
                    edge_df = fg.create_edge_dataframe_from_adj_diff(adj_diff, diff_thres)
                    if edge_df.empty:
                        Col4.subheader(
                            "The edges representing graph differences are not above the threshold to plot.",
                            divider=True)
                    else:
                        i = pairs[pair_counter][0]
                        j = pairs[pair_counter][1]
                        diff_graph = lrpgraph.LRPGraph(
                            edges_sample_i=edge_df,
                            source_column="source_node",
                            target_column="target_node",
                            edge_attrs=["LRP", "LRP_norm"],
                            top_n_edges=st.session_state['top_diff_n'],
                            sample_ID='DIFFERENCE ' + st.session_state["compare_grp_selected"][i] + ' vs ' +
                                      st.session_state["compare_grp_selected"][j],
                        )
                        st.session_state['civic_data'].get_molecular_profiles_matching_nodes(diff_graph)
                        diff_plots_container = Col4.container(border=False)
                        sb_t_col1, sb_t_col2 = Col4.columns(2)
                        container_difn_x = sb_t_col1.container(border=False)
                        container_difn_y = sb_t_col2.container(border=False)

                        plot_my_graph(container_difn_x, diff_graph)
                        diff_graph.get_communitites()
                        plot_my_graph(container_difn_y, diff_graph, diff_graph.communitites)

                        edge_df_sizes = []
                        for diff_thres in np.arange(0.0, 1., 0.02):
                            edge_df_tmp = fg.create_edge_dataframe_from_adj_diff(adj_diff, diff_thres)
                            edge_df_sizes.append((diff_thres, len(edge_df_tmp)))


                        sb_t_col1, sb_t_col2 = Col4.columns(2)
                        container_difn_x = sb_t_col1.container(border=False)
                        container_difn_y = sb_t_col2.container(border=False)
                        plot_my_graph(container_difn_x, G_dict12[i])
                        plot_my_graph(container_difn_y, G_dict12[j])

                        with sb_t_col1:
                            st.subheader("Number of Edges vs Difference Threshold")
                            tmp_df = pd.DataFrame(edge_df_sizes, columns=["x", "y"])
                            st.vega_lite_chart(
                                tmp_df,
                                {
                                    "mark": {
                                        "type": "line",
                                        "point": True
                                    },
                                    "encoding": {
                                        "x": { "field": "x", "type": "quantitative", "title": "Difference Threshold" },
                                        "y": { "field": "y", "type": "quantitative", "title": "Number of Edges"}
                                    },
                                    "selection": {
                                        "brush": {
                                            "type": "interval",
                                            "bind": "scales"
                                        }
                                    }
                                },
                                use_container_width=True
                            )
                        with sb_t_col2:
                            st.subheader("LRP values for edges sorted by LRP value")
                            st.line_chart(edge_df['LRP'], x_label='Edge #', y_label='LRP')

                        st.session_state['civic_data'].get_mps_summaries()
                        Col4.subheader("Molecular Profiles summaries")
                        mps_summaries = Col4.expander("See Molecular Profiles summaries")
                        mps_summaries.write(st.session_state['civic_data'].mps_summaries)

                        st.session_state['civic_data'].get_features_matching_nodes(diff_graph)
                        st.session_state['civic_data'].get_features_descriptions()
                        Col4.subheader("Genes descriptions")
                        mps_summaries = Col4.expander("See genes descriptions")
                        mps_summaries.write(st.session_state['civic_data'].features_descriptions)

                        st.session_state['civic_data'].get_evidence_ids_df()
                        st.session_state['civic_data'].get_evidence_desctiptions()
                        st.session_state['civic_data'].agragate_all_facts()
                        Col4.subheader("All facts")
                        mps_summaries = Col4.expander("See all facts")
                        st.session_state["edge_df"] = edge_df
                        st.session_state["all_facts"] = st.session_state['civic_data'].all_facts
                        print(st.session_state['civic_data'].all_facts)
                        mps_summaries.write(st.session_state['civic_data'].all_facts)
                        st.session_state["enable_chat_bot"] = True
                        Col4.divider()
                    pair_counter += 1
###########
#
# Chatbot
#
#
###########
    import random
    import time


    def return_gen_prot(datafm):
        proteins = set()
        genes = set()

        # Process both 'source_node' and 'target_node' in a single loop
        for node in itertools.chain(datafm['source_node'].unique(), datafm['target_node'].unique()):
            parts = node.rsplit("_", 1)  # Efficient split from the right
            if len(parts) == 2:
                name, ext = parts
                if ext == 'prot':
                    proteins.add(name)
                else:
                    genes.add(name)

        return list(proteins), list(genes)

    def response_generator():
        response = random.choice(
            [
                "Hello there! How can I assist you today?",
                "Hi, human! Is there anything I can help you with?",
                "Do you need help?",
            ]
        )
        for word in response.split():
            yield word + " "
            time.sleep(0.05)


    if not st.session_state.enable_chat_bot:
        Col5.subheader("You have not made any comparisons between groups yet.", divider=True)
    else:
        Col5.title("💬 Chatbot")
        diff_graph = lrpgraph.LRPGraph(
            edges_sample_i=st.session_state["edge_df"],
            source_column="source_node",
            target_column="target_node",
            edge_attrs=["LRP", "LRP_norm"],
            top_n_edges=st.session_state['top_diff_n'],
            sample_ID='DIFFERENCE ' + st.session_state["compare_grp_selected"][i] + ' vs ' +
                      st.session_state["compare_grp_selected"][j],
        )
        st.session_state['civic_data'].get_molecular_profiles_matching_nodes(diff_graph)
        st.session_state['civic_data'].get_mps_summaries()
        st.session_state['civic_data'].get_features_matching_nodes(diff_graph)
        st.session_state['civic_data'].get_features_descriptions()
        st.session_state['civic_data'].get_evidence_ids_df()
        st.session_state['civic_data'].get_evidence_desctiptions()
        st.session_state['civic_data'].agragate_all_facts()
        all_facts = st.session_state['civic_data'].all_facts
        prot_gen_results = return_gen_prot(st.session_state["edge_df"])
        prompt_part1 = (f"You are an expert in molecular genomics and cancer research. You always focus on specific proteins and genes of interest and the facts provided to you. Your reasoning is based on the context provided.\n\n"
                        f"I will provide you with context in the next part, followed by a list of key molecular facts and specific biomolecules of interest. "
                        f"Your task is to generate an in-depth analysis integrating these components.\n\n"

                        f"**Key Molecular Facts:** {all_facts}\n\n"

                        f"**Focus Areas:**\n"
                        f"- **Proteins of Interest:** {prot_gen_results[0]}\n"
                        f"- **Genes of Interest:** {prot_gen_results[1]}\n\n"

                        f"**Task:**\n"
                        f"Analyze the given molecular and cancer-related context, emphasizing the provided proteins and genes.\n"
                        f" Your response should be concise and specific, no need to explain basic facts, as the intended reader is an oncologist.\n\n"
                        #f"1. Explain the functional roles of these proteins in cancer progression.\n"
                        #f"2. Describe the genomic implications and interactions of the specified genes.\n"
                        #f"3. Integrate relevant molecular pathways and their impact on tumorigenesis.\n"
                        #f"4. Provide insights into potential therapeutic targets or biomarker relevance.\n\n"

                        f"Format your response with clear sectioning for readability. \n\n"

                        f"In the next part, I will provide the **specific context** for your analysis."
                        )
        agent = OpenAIAgent(system_role=prompt_part1)
        st.session_state["context_input"] = ""
        Col_sub_5_1, Col_sub_5_2 = Col5.columns([1, 2])
        Con_Col_sub_5_1 = Col_sub_5_1.container(height=400)
        Con_Col_sub_5_2 = Col_sub_5_2.container(height=325)
        with Con_Col_sub_5_1:
            Col5_context_form = st.form(key='context_form')
            st.session_state["context_input"] = Col5_context_form.text_area("Please provide some context", st.session_state["context_input"])
            submit_button_cb_f = Col5_context_form.form_submit_button(label='Submit')

            if submit_button_cb_f and st.session_state["awaiting_context"]:
                st.warning(" ℹ Context provided. ")
                st.session_state["messages"] = []
                prompt_part2 = (f"**Context:** {st.session_state['context_input']}\n\n"
                                f'At the end of your response, confirm by saying: **"OK, I am ready."**'
                                )
                agent.query(prompt_part2)
                st.session_state["context_input"] = ""
                st.session_state["awaiting_context"] = False

        with Con_Col_sub_5_2:
            if not st.session_state["awaiting_context"]:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if prompt := Col_sub_5_2.chat_input("Please write your question"):
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    # Display user message in chat message container
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Display assistant response in chat message container
                    with st.chat_message("assistant"):
                        response = st.write_stream(agent.query(prompt))
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})


















