import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import altair as alt
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pyvis.network import Network
import os
import importlib, sys

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

if 'lrp_df' not in st.session_state:
    st.session_state['lrp_df'] = pd.DataFrame([])


def save_my_uploaded_file(path, uploaded_file):
    repository_folder = path
    save_path = os.path.join(repository_folder, uploaded_file.name)
    with open(save_path, mode='wb') as w:
        w.write(uploaded_file.getvalue())
    return save_path


def find_my_keywords(lrp_df):
    option_tmp = []
    for opr_element in lrp_df.columns:
        tmp_ele = opr_element.split("-")[1].split("_")[0].strip()
        option_tmp.append(tmp_ele)

    keywords = list(dict.fromkeys(option_tmp))
    keywords.sort()
    return keywords

def plot_my_graph(container, graph):
    node_color_mapper = {'exp': 'gray', 'mut': 'red', 'amp': 'orange', 'del': 'green', 'fus': 'blue'}
    fg.plot_graph(graph, node_color_mapper)
    visor = Network(
        height='600px',
        width='100%',
        bgcolor='#FFFFFF',
        font_color='black'
    )
    visor.from_nx(graph.G)
    neighbor_map = visor.get_adj_list()
    for edge in visor.edges:
        edge["title"] = 'LRP_norm: {:.4f}'.format(edge["LRP_norm"])
        edge["color"] = "#888888"
        for node in visor.nodes:
            node["label"] = node["id"].split('_')[0]
            node["color"] = node_color_mapper[node["id"].split('_')[1]]
            node["title"] = node["label"]
            node["title"] += " Neighbors:"
            for i, item in enumerate(neighbor_map[node["id"]]):
                node["title"] += "\n" + item.split('_')[0]
                node["value"] = len(neighbor_map[node["id"]])

    visor.repulsion(
        node_distance=420,
        central_gravity=0.33,
        spring_length=110,
        spring_strength=0.10,
        damping=0.95
    )
    visor.show_buttons()

    try:
        path = '/tmp'
        visor.save_graph(f'{path}/pyvis_graph_'+graph.sample_ID+'.html')
        HtmlFile = open(f'{path}/pyvis_graph_'+graph.sample_ID+'.html', 'r', encoding='utf-8')

    except:
        path = '/html_files'
        visor.save_graph(f'{path}/pyvis_graph_' + graph.sample_ID + '.html')
        HtmlFile = open(f'{path}/pyvis_graph_' + graph.sample_ID + '.html', 'r', encoding='utf-8')

    path = './htmls/legends.html'
    chart_legend_css = open(path, 'r', encoding='utf-8')

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

    with container:
        subCol1, subCol2 = st.columns([7, 1])
        with subCol1:
            components.html(HtmlFile.read(), height=620, scrolling=True)
            with subCol2:
                components.html(chart_legend_css.read(), height=620)


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
        metadata_df = dtl.MetaData(file_path=save_my_uploaded_file('/tmp', path_to_metadata), delimiter=",")
        uploader_placeholder_md.empty()
        st.sidebar.info('File {0} has been analysed.'.format(path_to_metadata.name))

    if path_to_LRP_data and path_to_metadata:
        st.session_state['keywords'] = find_my_keywords(st.session_state['lrp_df'])

        keywords_form = st.sidebar.form('Keywords')
        with (keywords_form):
            keywords_selected = keywords_form.multiselect("Please select your keyword: ", st.session_state['keywords'],
                                                          placeholder="Choose a keyword.")
            filter_button = keywords_form.form_submit_button(label='Filter')

            if filter_button and keywords_selected:
                filtered_df = fg.filter_columns_by_keywords(st.session_state['lrp_df'], keywords_selected)
                LRP_to_graphs = fg.prepare_lrp_to_graphs(filtered_df)
                st.session_state['filtered_data'] = LRP_to_graphs
                st.session_state['first_form_completed'] = True
                keywords_form.success("Keywords filtered successfully! Proceed to the next step.")

        if st.session_state.get('first_form_completed', False):
            node_selection_form = st.sidebar.form('TopNSelection')
            with node_selection_form:
                LRP_to_graphs = st.session_state.get('filtered_data')
                top_n = node_selection_form.slider(
                    "Please select the number of top n edges",
                    min_value=1,
                    max_value=len(LRP_to_graphs.index),
                    value=len(LRP_to_graphs.index) // 2
                )
                submit_button = node_selection_form.form_submit_button(label='Submit')
                if submit_button:
                    G_dict = fg.get_all_graphs_from_lrp(LRP_to_graphs, top_n)
                    # Validation
                    assert len(G_dict[1].G.edges) == top_n, "Edge count mismatch."
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

    if "selected_gId" not in st.session_state:
        st.session_state['selected_gId'] = sampleIDs[0]


    def new_gid_callback():
        st.session_state["selected_gId"] = st.session_state.new_gId


    st.session_state["selected_gId"] = st.selectbox("Please select the sample you want to see:",
                                                    sampleIDs,
                                                    index=None,
                                                    help="Choose from the available graphs listed below.",
                                                    key="new_gId",
                                                    placeholder="Select a graph...",
                                                    on_change=new_gid_callback,
                                                    )

    if st.session_state["selected_gId"]:
        print("You selected index: {0}".format(sampleIDs.index(st.session_state["selected_gId"])))
        G = st.session_state['G_dict'][sampleIDs.index(st.session_state["selected_gId"])]
        container_main = st.container(border=False)
        plot_my_graph(container_main, G)

        embeddings_df = fg.extract_raveled_fixed_size_embedding_all_graphs(st.session_state['G_dict'])
        sorted_distance_df = fg.compute_sorted_distances(embeddings_df, G.sample_ID)
        top_n_similar = 3
        # Display the modified text
        container_main.title("{0} most similar graphs.".format(top_n_similar))
        # Get the top n most similar samples
        top_n_samples = sorted_distance_df.head(top_n_similar + 1)
        for i in range(top_n_similar + 1):
            if i == 0:
                continue
            sample_ID = top_n_samples.iloc[i, 0]
            G = next(G for G in st.session_state['G_dict'].values() if G.sample_ID == sample_ID)
            container_topn = st.container(border=False)
            plot_my_graph(container_topn, G)
