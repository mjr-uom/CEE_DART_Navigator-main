
import pandas as pd
import networkx as nx
import LRPGraph_code as lrpgraph
import numpy as np


def split_and_aggregate_lrp(LRP_to_graphs, metadata_df, stratify_by, agg_func="mean"):
    """
    Split the transposed `LRP_to_graphs` dataframe into groups based on the specified column 
    from `metadata_df`, compute the mean or median values for each group, and re-transpose.

    Parameters:
    - LRP_to_graphs (pd.DataFrame): The dataframe to be split and aggregated.
    - metadata_df (pd.DataFrame): The dataframe containing metadata for grouping.
    - stratify_by (str): The column name in `metadata_df` used to define groups.
    - agg_func (str): The aggregation function to use ('mean' or 'median'). Default is 'mean'.

    Returns:
    - pd.DataFrame: A dataframe with original orientation containing aggregated values 
                    for each group and preserving other relevant columns.
    """
    # Check if the specified column exists in metadata_df
    if stratify_by not in metadata_df.columns:
        raise ValueError(f"Column '{stratify_by}' not found in metadata_df")

    # Check if the aggregation function is valid
    if agg_func not in ["mean", "median"]:
        raise ValueError("agg_func must be either 'mean' or 'median'")

    # Identify sample columns in LRP_to_graphs that exist in metadata_df
    sample_columns = [col for col in LRP_to_graphs.columns if col in metadata_df.index]

    # Ensure we have relevant sample columns
    if not sample_columns:
        raise ValueError("No matching sample columns found between LRP_to_graphs and metadata_df")

    # Transpose the LRP_to_graphs dataframe for easier grouping
    transposed_df = LRP_to_graphs[sample_columns].T

    # Add the grouping information from metadata
    transposed_df["group"] = metadata_df.loc[transposed_df.index, stratify_by]

    # Perform the grouping and aggregation
    if agg_func == "mean":
        aggregated_df = transposed_df.groupby("group").mean()
    else:
        aggregated_df = transposed_df.groupby("group").median()

    # Re-transpose the aggregated dataframe
    re_transposed_df = aggregated_df.T

    # Add the 'index', 'source_node', and 'target_node' columns back
    result_df = LRP_to_graphs[["index", "source_node", "target_node"]].copy()
    result_df = result_df.join(re_transposed_df, how="left")

    return result_df



def filter_columns_by_keywords(df, keywords = None):
    """
    Filters columns of the dataframe where column names contain keywords from the input list.
    If the keywords list is empty, no filtering is applied.

    A column is included if, after splitting the name on ' - ', the resulting two strings
    contain at least one keyword from the input list.

    Parameters:
        df (pd.DataFrame): The dataframe with columns to filter.
        keywords (list): List of keywords to filter by.

    Returns:
        pd.DataFrame: A dataframe containing only the filtered columns.
    """
    if keywords is None:
        return df

    filtered_columns = []

    for col in df.columns:
        parts = col.split(" - ")
        if len(parts) == 2:  # Ensure the column name splits into exactly two parts
            if any(keyword in parts[0] or keyword in parts[1] for keyword in keywords):
                filtered_columns.append(col)

    return df[filtered_columns]


def prepare_lrp_to_graphs(LRP_df):
    """
    Prepare LRP_to_graphs by splitting the 'edge' column into 'source_node' and 'target_node'.

    Parameters:
    LRP_df (pd.DataFrame): DataFrame containing LRP data with an 'edge' column.

    Returns:
    pd.DataFrame: A DataFrame with 'source_node' and 'target_node' columns added,
                  along with the original LRP values.
    """
    # Transpose the DataFrame and reset the index
    LRP_to_graphs = LRP_df.T.copy().reset_index()

    # Split the 'index' column into 'source_node' and 'target_node'
    LRP_to_graphs["source_node"] = LRP_to_graphs["index"].str.split(" - ", expand=True)[
        0
    ]
    LRP_to_graphs["target_node"] = LRP_to_graphs["index"].str.split(" - ", expand=True)[
        1
    ]

    return LRP_to_graphs


def create_edges_from_lrp(LRP_to_graphs, i):
    """
    Create edges from LRP (Layer-wise Relevance Propagation) data.
    This function takes a DataFrame containing LRP data and an index, and creates a new DataFrame
    representing edges between source and target nodes, sorted by the LRP values in descending order.

    Parameters:
    LRP_df (pd.DataFrame): A DataFrame containing LRP data with an 'edge' column and LRP values of all samples in subsequent columns.
    i (int): An index indicating which LRP column (which sample) to use for creating edges.

    Returns:
    pd.DataFrame: A DataFrame with columns 'source_node', 'target_node', and 'LRP', sorted by 'LRP' in descending order.
    """
    # Extract LRP values for the given index
    lrp = LRP_to_graphs.iloc[
        :, i
    ].values  # Offset by 3 to skip metadata columns; adjust as needed.

    # Create 'edges' DataFrame
    edges = pd.DataFrame(
        {
            "source_node": LRP_to_graphs["source_node"],
            "target_node": LRP_to_graphs["target_node"],
            "LRP": lrp,
        }
    )

    # Sort by LRP in descending order
    edges = edges.sort_values("LRP", ascending=False).reset_index(drop=True)
    return edges





def get_all_graphs_from_lrp(LRP_to_graphs: pd.DataFrame, top_n_edges: int = None) -> dict:
    """
    Generate a dictionary of graphs from LRP data.
    This function processes LRP (Layer-wise Relevance Propagation) data to create a dictionary of graphs.
    Each graph represents the relationships between nodes based on the LRP values.

    Parameters:
    LRP_to_graphs (pd.DataFrame): DataFrame containing LRP data.
    n_ (int, optional): Number of edges to select for each graph. Default is None.

    Returns:
    dict: A dictionary where each key is a sample index and the value is another dictionary containing the graph 'G'.
    """
    # make sure that columns ['index', 'source_node', 'target_node'] are the first 3 columns in the LRP_to_graphs dataframe
    remaining_columns = [col for col in LRP_to_graphs.columns if col not in ["index", "source_node", "target_node"]]
    LRP_to_graphs = LRP_to_graphs[
        ["index", "source_node", "target_node"] + remaining_columns
    ]

    graphs_dict = {}
    for i in range(
        LRP_to_graphs.shape[1] - 3
    ):  # Iterate over the number of samples (adjust as needed for your data)
        print(f"Processing sample {i}")
        edges_sample_i = create_edges_from_lrp(LRP_to_graphs, 3 + i)

        graphs_dict[i] = {}  # Initialize dictionary for this sample

        graph = lrpgraph.LRPGraph(
            edges_sample_i=edges_sample_i,
            source_column="source_node",
            target_column="target_node",
            edge_attrs=["LRP", "LRP_norm"],
            top_n_edges = top_n_edges,
            sample_ID = LRP_to_graphs.columns[i+3],
        )

        graphs_dict[i] = graph

    return graphs_dict

def get_all_fixed_size_adjacency_matrices(G_dict: dict):
    """
    Generate adjacency matrices for all graphs in the input dictionary.

    Parameters:
    G_dict (dict): A dictionary of graphs where each value is a graph object.

    Returns:
    dict: A dictionary where each key is a sample index and the value is the adjacency matrix of the corresponding graph.
    """
    all_nodes = set()
    for i in G_dict:
        G = G_dict[i].G
        all_nodes.update(G.nodes())
    all_nodes = list(all_nodes)
    all_nodes.sort()

    for i in G_dict:
        G = G_dict[i]
        G.create_fixed_size_adjacency_matrix(all_nodes)
        G.all_nodes = all_nodes

def get_all_fixed_size_embeddings(G_dict: dict):
    """
    Generate fixed-size embeddings for all graphs in the input dictionary.

    Parameters:
    G_dict (dict): A dictionary of graphs where each value is a graph object.

    Returns:
    dict: A dictionary where each key is a sample index and the value is the fixed-size embedding of the corresponding graph.
    """
    all_nodes = set()
    for i in G_dict:
        G = G_dict[i].G
        all_nodes.update(G.nodes())
    all_nodes = list(all_nodes)

    for i in G_dict:
        G = G_dict[i]
        G.compute_laplacian_embedding()
        G.create_fixed_size_embedding(all_nodes)
        G.ravel_fixed_size_embedding()

def extract_raveled_fixed_size_embedding_all_graphs(G_dict: dict) -> np.ndarray:
    """
    Extracts raveled fixed-size embeddings from all graphs in the given dictionary and converts them into a numpy array.
    
    Args:
        G_dict (dict): A dictionary where keys are graph identifiers and values are graph objects. Each graph object 
                       is expected to have an attribute 'raveled_fixed_size_embedding'.
    
    Returns:
        numpy.ndarray: A 2D numpy array where each row corresponds to the raveled fixed-size embedding of a graph.
    """
    # Extract all raveled_fixed_size_embeddings from the graphs
    embeddings = [G_dict[i].raveled_fixed_size_embedding for i in G_dict]

    # get samples names to use as index
    samples_names = [G_dict[i].sample_ID for i in G_dict]

    # Convert the list of embeddings to a pandas df
    embeddings_df = pd.DataFrame(embeddings, index=samples_names)

    return embeddings_df
    

import matplotlib.pyplot as plt
def plot_graph(graph, node_color_mapper):
    """
    Plots a graph with specified node colors and edge widths.

    Parameters:
    graph 
    node_color_mapper (dict): A dictionary mapping node types to colors.

    Returns:
    None
    """
    G = graph.G
    degrees = np.array(list(nx.degree_centrality(G).values()))
    degrees_norm = degrees / np.max(degrees)
    widths = list(nx.get_edge_attributes(G, 'LRP_norm').values())
    widths = [x * 2 for x in widths]

    edge_colors = plt.cm.Greys(widths)

    fig, ax = plt.subplots(figsize=(12, 12))
    ls = list(node_color_mapper.keys())
    cl = list(node_color_mapper.values())
    for label, color in zip(ls, cl):
        ax.plot([], [], 'o', color=color, label=label)
    ax.legend(title='Nodes', loc='best')

    node_colors = pd.Series([i.split('_')[1] for i in list(G.nodes)]).map(node_color_mapper)
    node_labels = {node: node.split('_')[0] for node in G.nodes()}

    pos = nx.spring_layout(G, weight='LRP_norm')

    nx.draw(G, with_labels=True,
            labels=node_labels,
            node_color=node_colors,
            width=np.array(widths)*2,
            pos=pos,
            edge_color=edge_colors,
            ax=ax,
            node_size=degrees_norm * 500,
            edgecolors='white',
            linewidths=0.5,
            font_size=8)
    ax.set_title('Sample ' + graph.sample_ID + '\nGraph of the top {} edges with the highest LRP values'.format(graph.top_n_edges))


from scipy.spatial.distance import cdist

def compute_sorted_distances(embeddings_df, sample_ID, metric='euclidean'):
    """
    Compute and sort distances between a selected sample and all other samples in the embeddings dataframe.

    Parameters:
    embeddings_df (pd.DataFrame): DataFrame containing the embeddings of all samples.
    sample_ID (str): The ID of the sample to compare against all other samples.
    metric (str): The distance metric to use. Default is 'euclidean'.

    Returns:
    pd.DataFrame: DataFrame containing the sample IDs and their corresponding distances, sorted by distance.
    """

    selected_row = embeddings_df.loc[sample_ID, :].values.reshape(1, -1)

    # Compute the distance between the selected row and all other rows
    distances = cdist(selected_row, embeddings_df.values, metric=metric).flatten()

    # Create a dataframe with distances and sort by distance
    distance_df = pd.DataFrame({'Sample': embeddings_df.index, 'Distance': distances})
    sorted_distance_df = distance_df.sort_values(by='Distance')

    return sorted_distance_df




def calculate_adjacency_difference(graph1, graph2):
    """
    Calculate the adjacency difference matrix between two graphs and apply a threshold.

    Parameters:
        graph1 (Graph): The first graph object with a fixed_size_adjacency_matrix attribute.
        graph2 (Graph): The second graph object with a fixed_size_adjacency_matrix attribute.
        diff_thres (float): The difference threshold for including edges.

    Returns:
        pd.DataFrame: A dataframe representing the adjacency difference matrix with applied threshold.
    """
    adj1 = graph1.fixed_size_adjacency_matrix
    adj2 = graph2.fixed_size_adjacency_matrix

    adj_diff = adj1 - adj2
    adj_diff = pd.DataFrame(adj_diff, index=list(graph1.all_nodes), columns=list(graph1.all_nodes))

    return adj_diff

# define difference threshold
import seaborn as sns  
def apply_threshold_on_adj_diff(adj_diff: pd.DataFrame, threshold: float):
    """
    Apply a threshold to the adjacency difference matrix.

    This function sets all elements in the adjacency difference matrix 
    that are below the given threshold to zero.

    Parameters:
    adj_diff (pd.DataFrame): The adjacency difference matrix.
    threshold (float): The threshold value. All elements in adj_diff 
                       below this value will be set to zero.

    Returns:
    None: The function modifies the input matrix in place.
    """
    adj_diff[adj_diff < threshold] = 0

    
def create_edge_dataframe_from_adj_diff(adj_diff, threshold):
    """
    Create a dataframe with columns edge, source_node, target_node, LRP_norm based on edges 
    that are created from the adjacency matrix after applying the threshold.

    Parameters:
        adj_diff (pd.DataFrame): The adjacency difference matrix.
        threshold (float): The difference threshold for including edges.

    Returns:
        pd.DataFrame: A dataframe with selected edges and their LRP values.
    """
    # Apply threshold
    adj_diff = adj_diff.copy()
    apply_threshold_on_adj_diff(adj_diff, threshold)

    # Create a list to store edge data
    edge_data = []

    # Iterate over the adjacency matrix to extract edges
    for source_node in adj_diff.index:
        for target_node in adj_diff.columns:
            if adj_diff.loc[source_node, target_node] > 0:
                edge_data.append({
                    'edge': f"{source_node} - {target_node}",
                    'source_node': source_node,
                    'target_node': target_node,
                    'LRP': adj_diff.loc[source_node, target_node],
                    'LRP_norm': adj_diff.loc[source_node, target_node]
                })

    # Create a dataframe from the edge data
    edge_df = pd.DataFrame(edge_data)

    return edge_df