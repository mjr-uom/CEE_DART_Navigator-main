import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import seaborn as sns
import altair as alt
import networkx as nx
import matplotlib as mt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from pyvis.network import Network
import os
import hashlib
import itertools
import importlib, sys
from htmls.dynamicLegends import generate_legend_html
from len_gen import generate_legend_table
path_to_functions_directory = r'./source'
if path_to_functions_directory not in sys.path:
    sys.path.append(path_to_functions_directory)

import dataloaders as dtl

importlib.reload(dtl)

import functions_graphs as fg

importlib.reload(fg)

import LRPGraph_code as lrpgraph

importlib.reload(lrpgraph)

if 'first_form_completed' not in st.session_state:
    st.session_state['first_form_completed'] = False

if 'second_form_completed' not in st.session_state:
    st.session_state['second_form_completed'] = False

if 'G_dict' not in st.session_state:
    st.session_state['G_dict'] = {}

if 'keywords' not in st.session_state:
    st.session_state['keywords'] = list()

if 'tumor_tissue_site' not in st.session_state:
    st.session_state['tumor_tissue_site'] = list()

if 'ttss_selected' not in st.session_state:
    st.session_state['ttss_selected'] = list()

if 'acronym' not in st.session_state:
    st.session_state['acronym'] = list()

if 'lrp_df' not in st.session_state:
    st.session_state['lrp_df'] = pd.DataFrame([])

if 'filtered_tts_lrp_df' not in st.session_state:
    st.session_state['filtered_tts_lrp_df'] = pd.DataFrame([])

if 'metadata_df' not in st.session_state:
    st.session_state['metadata_df'] = pd.DataFrame([])

if 'f_tumor_tissue_site' not in st.session_state:
    st.session_state['f_tumor_tissue_site'] = pd.DataFrame([])

if 'f_acronym' not in st.session_state:
    st.session_state['f_acronym'] = pd.DataFrame([])

if 'filters_form_completed' not in st.session_state:
    st.session_state['filters_form_completed'] = None

if 'tts_filter_button' not in st.session_state:
    st.session_state['tts_filter_button'] = None

if 'frequent_kws' not in st.session_state:
    st.session_state['frequent_kws'] = pd.DataFrame([])

if 'LRP_to_graphs_stratified' not in st.session_state:
    st.session_state['LRP_to_graphs_stratified'] = pd.DataFrame([])

if 'calculate_button' not in st.session_state:
    st.session_state['calculate_button'] = False

if 'comparison_grp_button' not in st.session_state:
    st.session_state['comparison_grp_button'] = False

if 'compare_form_complete' not in st.session_state:
    st.session_state['compare_form_complete'] = False

if 'top_n' not in st.session_state:
    st.session_state['top_n'] = 0

if 'top_n_similar' not in st.session_state:
    st.session_state['top_n_similar'] = None

if 'compare_grp_selected' not in st.session_state:
    st.session_state["compare_grp_selected"] = list()

st.set_page_config(layout="wide")


def assign_colors(strings):
    color_names = sorted(list(mcolors.CSS4_COLORS.keys()))

    def get_color(string):
        hash_value = int(hashlib.sha256(string.encode()).hexdigest(), 16)
        return color_names[hash_value % len(color_names)]

    keys = {string.split("_")[-1] for string in strings}  # Set of unique keys
    sorted_keys = sorted(keys)  # Sort the keys for deterministic order

    color_dict = {key: get_color(key) for key in sorted_keys}
    return color_dict


def save_my_uploaded_file(path, uploaded_file):
    repository_folder = path
    save_path = os.path.join(repository_folder, uploaded_file.name)
    with open(save_path, mode='wb') as w:
        w.write(uploaded_file.getvalue())
    return save_path

def find_my_ttss(df_ttss_org):
    df_ttss = df_ttss_org.data
    col_name = list(df_ttss.columns)
    col_name.remove('tumor_tissue_site')
    df_tmp = df_ttss.set_index(col_name).tumor_tissue_site.str.get_dummies(sep='|').stack().reset_index().loc[lambda x: x[0] != 0].drop(0, axis=1)
    df_tmp = df_tmp.rename({i: "tumor_tissue_site" for i in df_tmp.columns if i not in col_name}, axis="columns")
    tmp_lst = list(df_tmp['tumor_tissue_site'].unique())
    tmp_lst.sort()
    return tmp_lst

def find_my_metadata_catalog(col_name):
    return list(set(st.session_state['metadata_df'].data[col_name].tolist()))

def find_my_keywords(lrp_df):
    option_tmp = []
    for opr_element in lrp_df.columns:
        tmp_ele = "_".join(opr_element.split("-")[-1].strip().split("_")[:-1])
        option_tmp.append(tmp_ele)

    keywords = list(dict.fromkeys(option_tmp))
    keywords.sort()
    return keywords

def create_pyvis_graph(net, G):
    # Position nodes using a layout where edge weights affect distances
    #pos = nx.kamada_kawai_layout(G, weight="LRP_norm")  # Kamada-Kawai considers edge weights
    pos = nx.spring_layout(G, weight="LRP_norm", scale=1000)

    for node in G.nodes():
        x, y = pos[node]
        #print("({:.4f} , {:.4f})".format(x, y))
        net.add_node(node, x = x , y=y , physics=False)

    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, LRP_norm=data["LRP_norm"], physics=False)

    #net.set_options('''
    #                    var options = {
    #                                    "nodes": {
    #                                                "fixed": {
    #                                                            "x": true,
    #                                                            "y": true
    #                                                         }
    #                                              }
    #                                  }
    #                ''')


def plot_my_graph(container, graph):
    #fg.plot_graph(graph, node_color_mapper)
    visor = Network(
        height='600px',
        width='100%',
        bgcolor='#FFFFFF',
        font_color='black',
        #select_menu=True,
        #cdn_resources='remote',
        #filter_menu=True,
    )
    #visor.from_nx(graph.G)
    create_pyvis_graph(visor, graph.G)
    node_color_mapper = assign_colors(sorted([node["id"] for node in visor.nodes]))
    neighbor_map = visor.get_adj_list()
    edge_dist = list(nx.get_edge_attributes(graph.G, 'LRP_norm').values())
    norm_colour = mt.colors.Normalize(vmin=0.0, vmax=max(edge_dist), clip=True)
    colour_mapper = (cm.ScalarMappable(norm=norm_colour, cmap=cm.Greys))
    for edge in visor.edges:
        edge["title"] = 'LRP_norm: {:.4f}'.format(edge["LRP_norm"])
        edge["color"] = mt.colors.rgb2hex(colour_mapper.to_rgba(edge["LRP_norm"]))
        edge["minlen"] = edge["LRP_norm"]
        edge["value"] = edge["LRP_norm"]

    node_type = []
    for node in visor.nodes:
        node["label"] = "_".join(node["id"].split('_')[:-1])
        node["_type"] = node["id"].split('_')[-1]
        node["color"] = node_color_mapper[node["id"].split('_')[-1]]
        node["title"] = node["label"]
        node["title"] += " Neighbors:"
        node_type.append(node["id"].split('_')[-1])

        for _, item in enumerate(neighbor_map[node["id"]]):
            node["title"] += "\n" + "_".join(item.split('_')[:-1])
            node["value"] = len(neighbor_map[node["id"]])

    #visor.repulsion(
    #    node_distance=420,
    #    central_gravity=0.33,
    #    spring_length=110,
    #    spring_strength=0.10,
    #    damping=0.95
    #)
    visor.show_buttons(filter_=['nodes','edges','physics'])
    #visor.set_options('''
    #                    var options = {
    #                                    "nodes": {
    #                                                "fixed": {
    #                                                            "x": true,
    #                                                            "y": true
    #                                                         }
    #                                              }
    #                                  }
    #                ''')

    if isinstance(graph.sample_ID, int):
        graph.sample_ID = str(graph.sample_ID)

    try:
        path = '/tmp'
        visor.save_graph(f'{path}/pyvis_graph_' + graph.sample_ID + '.html')
        HtmlFile = open(f'{path}/pyvis_graph_' + graph.sample_ID + '.html', 'r', encoding='utf-8')

    except:
        path = '/html_files'
        visor.save_graph(f'{path}/pyvis_graph_' + graph.sample_ID + '.html')
        HtmlFile = open(f'{path}/pyvis_graph_' + graph.sample_ID + '.html', 'r', encoding='utf-8')

    path = './htmls/legends.html'
    #chart_legend_css = generate_legend_html(list(set(node_type)))
    chart_legend_css =  generate_legend_table(node_color_mapper)

    style_heading = 'text-align: center'
    css = r'''<style>
                             body {
                              margin: 0; /* Remove default margin */
                             padding: 10px; /* Add padding inside the border */
                              border: 1px solid white; /* Border around the entire page */
                         font-family: Arial, sans-serif;
                               }
               </style>
                 '''

    container.markdown(css, unsafe_allow_html=True)
    container.markdown(f"<h1 style='{style_heading}'>Sample '{graph.sample_ID}' </h1>", unsafe_allow_html=True)
    container.markdown(
        f"<h2 style='{style_heading}'>Graph of the top '{graph.top_n_edges}' edges with the highest LRP values </h2>",
        unsafe_allow_html=True)

    zoom_options = """
    var options = {
    "interaction": {
        "zoomView": true,  
        "zoomSpeed": 1.2   
    },
    "layout": {
        "randomSeed": 42,  
        "scale": 50.0       
    }
       }
    """
    visor.set_options(zoom_options)

    with container:
        subCol1, subCol2 = st.columns([5, 1])
        with subCol1:
            components.html(HtmlFile.read(), height=620, scrolling=True)
            with subCol2:
                components.html(chart_legend_css, height=620)


def make_heatmap(input_df, input_y, input_x, input_color, input_color_theme):
    heatmap = alt.Chart(input_df).mark_rect().encode(
            y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="Year", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
            x=alt.X(f'{input_x}:O', axis=alt.Axis(title="", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
            color=alt.Color(f'max({input_color}):Q',
                             legend=None,
                             scale=alt.Scale(scheme=input_color_theme)),
            stroke=alt.value('black'),
            strokeWidth=alt.value(0.25),
        ).properties(width=900
        ).configure_axis(
        labelFontSize=12,
        titleFontSize=12
        )
    # height=300
    return heatmap

#def initialise_my_app():
#    pyautogui.hotkey("ctrl","F5")

def create_multiselect(catalog_name: str, values: list, container: object):
    with container:
        selected_values = st.multiselect("Please select \"" + catalog_name + "\" : ",
            values,
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
    st.sidebar.title('፨ LPR Dashboard')
    uploader_placeholder = st.sidebar.empty()
    path_to_LRP_data = uploader_placeholder.file_uploader("Upload data LPR")

    if path_to_LRP_data is not None:
        st.session_state['lrp_df'] = dtl.LRPData(file_path=save_my_uploaded_file('/tmp', path_to_LRP_data),
                                                 delimiter=",").read_and_validate()
        uploader_placeholder.empty()
        uploader_placeholder.info('File {0} has been analysed.'.format(path_to_LRP_data.name))

    uploader_placeholder_md = st.sidebar.empty()
    path_to_metadata = uploader_placeholder_md.file_uploader("Upload data-meta data")

    if path_to_metadata is not None:
        st.session_state['metadata_df'] = dtl.MetaData(file_path=save_my_uploaded_file('/tmp', path_to_metadata),
                                                       delimiter=",")
        uploader_placeholder_md.empty()
        st.sidebar.info('File {0} has been analysed.'.format(path_to_metadata.name))

#### Filters

    if path_to_LRP_data and path_to_metadata:
        filter_catalog = get_column_values(st.session_state['metadata_df'])

        filters_form = st.sidebar.form('filters')
        with (filters_form):
            filters = {}
            for key, values in filter_catalog.items():
                filters[key] =  create_multiselect(key, values, filters_form)

            sm_button = filters_form.form_submit_button(label='Filter')

            if sm_button:
                desired_barcodes = st.session_state['metadata_df'].data
                for filter, values in filters.items():
                    if len(values) > 0:
                        filter_query = filter + ' == [ ' +  ', '.join(f'"{i}"' for i in values) + ' ]'
                        desired_barcodes = desired_barcodes.query(filter_query)

                valid_bc = list(set(desired_barcodes.index.tolist()))
                st.session_state['filtered_tts_lrp_df'] = st.session_state['lrp_df'].filter(items=valid_bc, axis=0)
                if st.session_state['filtered_tts_lrp_df'].empty:
                    sm_button = False
                    filters_form.warning("The selection criteria you have chosen are not yielding any results. Please select alternative values.")
                else:
                    filters_form.success("The selection criteria you have chosen are filtered successfully! Proceed to the next step.")
                    st.session_state['filters_form_completed'] = True

#### Keywords
if st.session_state.get('filters_form_completed', False):
    uploader_placeholder_kwf = st.sidebar.empty()
    path_to_plkeywords = uploader_placeholder_kwf.file_uploader("Upload frequent keywords")

    if path_to_plkeywords is not None:
        st.session_state['frequent_kws'] = pd.read_csv(save_my_uploaded_file('/tmp', path_to_plkeywords), header=None)
        uploader_placeholder_kwf.empty()
        st.sidebar.info('File {0} has been analysed.'.format(path_to_plkeywords.name))

    if path_to_LRP_data and path_to_metadata:
        if st.session_state['filtered_tts_lrp_df'].empty:
            st.session_state['keywords'] = find_my_keywords(st.session_state['lrp_df'])
        else:
            st.session_state['keywords'] = find_my_keywords(st.session_state['filtered_tts_lrp_df'])

        keywords_form = st.sidebar.form('Keywords')
        with (keywords_form):
            if st.session_state['frequent_kws'].empty:
                keywords_selected = keywords_form.multiselect("Please select your keyword: ",
                                                              st.session_state['keywords'],
                                                              [],
                                                              placeholder="Choose a keyword.")
            else:
                keywords_selected = keywords_form.multiselect("Please select your keyword: ",
                                                              st.session_state['keywords'],
                                                              st.session_state['frequent_kws'][0].tolist(),
                                                              placeholder="Choose a keyword.")


            filter_button = keywords_form.form_submit_button(label='Filter')
            if filter_button:
                if not keywords_selected:
                    keywords_selected = None
                filtered_df = fg.filter_columns_by_keywords(st.session_state['filtered_tts_lrp_df'],
                                                            keywords_selected)
                st.session_state['filtered_data'] = fg.prepare_lrp_to_graphs(filtered_df)
                st.session_state['first_form_completed'] = True
                uploader_placeholder_kwf.empty()
                keywords_form.success("Keywords filtered successfully! Proceed to the next step.")

################
if st.session_state.get('first_form_completed', False):
    node_selection_form = st.sidebar.form('TopNSelection')
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
    # Col1, Col2 = st.columns(2)
    Col1, Col2, Col3, Col4 = st.tabs(
        ["፨ Selected sample", "⩬ Top N similar", "🔍 Group comparison ", "⚖️ Graph differences"])
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

    # Display the modified text
    # container_main.title("{0} most similar graphs.".format(top_n_similar))
    # Get the top n most similar samples
    G = st.session_state['G_dict'][map_index_to_unsorted(disp_list.index(st.session_state["selected_gId"]), disp_list, sampleIDs)]
    embeddings_df = fg.extract_raveled_fixed_size_embedding_all_graphs(st.session_state['G_dict'])
    sorted_distance_df = fg.compute_sorted_distances(embeddings_df, G.sample_ID)


    def new_stratify_by_callback():
        st.session_state["top_n_similar"] = st.session_state.new_top_n_similar


    st.session_state["top_n_similar"] = Col2.number_input(
        "Please provide the number of similar graphs to display:",
        min_value=1,
        max_value=6,
        step=1,
        key="new_top_n_similar",
        placeholder="Select a value...",
        on_change=new_stratify_by_callback
    )
    if st.session_state["top_n_similar"] > 0:
        top_n_samples = sorted_distance_df.head(st.session_state["top_n_similar"] + 1)
        Col2_subC_1, Col2_subC_2 = Col2.columns(2)
        for i in range(st.session_state["top_n_similar"] + 1):
            # if i == 0:
            #    continue
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
    #if len(st.session_state["LRP_to_graphs_stratified"].columns.tolist()) > 4:
    filtered_columns = [col for col in st.session_state["LRP_to_graphs_stratified"].columns
                        if (isinstance(col, str) and not any(
            substr in col for substr in ["index", "source_node", "target_node"])) or
                        isinstance(col, int)]
    # filtered_columns = [col for col in st.session_state["LRP_to_graphs_stratified"].columns.tolist()
    #                    if
    #                    isinstance(col, str) and "index" not in col and "source_node" not in col and "target_node" not in col]

    compare_form = Col3.form('Compare')
    with (compare_form):
        if len(filtered_columns):
            st.session_state["compare_grp_selected"] = compare_form.multiselect("Please select your comparison groups: ",
                                                                                filtered_columns,
                                                                                [],
                                                                                placeholder="Choose a comparison group.")
        else:
            compare_form.warning("Unfortunately, your selection criteria are not generating any comparison group.")

        st.session_state["comparison_grp_button"]  = compare_form.form_submit_button(label='Submit')

        if st.session_state["comparison_grp_button"] and st.session_state["compare_grp_selected"]:
            LRP_to_graphs_stratified_sel_grps = st.session_state["LRP_to_graphs_stratified"][
                ['index', 'source_node', 'target_node'] + st.session_state["compare_grp_selected"]]

            G_dict12 = fg.get_all_graphs_from_lrp(LRP_to_graphs_stratified_sel_grps,
                                                  st.session_state['top_n'])
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
if st.session_state.get('compare_form_complete', False):
    if len(st.session_state["compare_grp_selected"]) < 2:
        Col4.subheader(
            "Regrettably, your selection criteria are not generating a sufficient number of comparison groups.",
            divider=True)
    else:
        LRP_to_graphs_stratified_sel_grps = st.session_state["LRP_to_graphs_stratified"][
            ['index', 'source_node', 'target_node'] + st.session_state["compare_grp_selected"]]

        G_dict12 = fg.get_all_graphs_from_lrp(LRP_to_graphs_stratified_sel_grps,
                                              st.session_state['top_n'])

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
            # st.session_state['calculate_button'] = st.form_submit_button(label='Calculate')
            # print(st.session_state['calculate_button'])
            # if st.session_state.get('calculate_button', False):
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
                            top_n_edges=st.session_state['top_n'],
                            sample_ID='DIFFERENCE ' + st.session_state["compare_grp_selected"][i] + ' vs ' +
                                      st.session_state["compare_grp_selected"][j],
                        )
                        diff_plots_container = Col4.container(border=False)
                        plot_my_graph(diff_plots_container, diff_graph)

                        sb_t_col1, sb_t_col2 = Col4.columns(2)
                        container_difn_x = sb_t_col1.container(border=False)
                        container_difn_y = sb_t_col2.container(border=False)
                        plot_my_graph(container_difn_x, G_dict12[i])
                        plot_my_graph(container_difn_y, G_dict12[j])

                        with sb_t_col1:
                            st.subheader("All differences")
                            fig, ax = plt.subplots(figsize=(16, 12))
                            sns.heatmap(adj_diff, xticklabels=1, yticklabels=1, linewidths=0.2,
                                        cmap='Reds', vmin=0, vmax=1, ax=ax)
                            st.pyplot(fig)

                        with sb_t_col2:
                            st.subheader("Differences above the threshold")
                            fig2, ax2 = plt.subplots(figsize=(16, 12))
                            sns.heatmap(adj_diff, xticklabels=1, yticklabels=1, linewidths=0.2,
                                        cmap='Reds', vmin=0, vmax=1, mask=adj_diff < diff_thres,
                                        ax=ax2)
                            st.pyplot(fig2)

                        Col4.divider()
                    pair_counter += 1
