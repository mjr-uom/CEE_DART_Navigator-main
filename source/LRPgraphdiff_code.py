# get the adjacency matrices
import functions_graphs as fg
import LRPGraph_code as lrpgraph
import numpy as np
import pandas as pd
from gprofiler import GProfiler


class LRPGraphDiff:
    def __init__(self, LRPGraph1, LRPGraph2, diff_thres=0.6):
        self.LRPGraph1 = LRPGraph1
        self.LRPGraph2 = LRPGraph2
        self.diff_thres = diff_thres
        self.adj_diff = fg.calculate_adjacency_difference(
            self.LRPGraph1, self.LRPGraph2
        )
        self.edge_df = fg.create_edge_dataframe_from_adj_diff(
            self.adj_diff, self.diff_thres
        )
        self.diff_graph = lrpgraph.LRPGraph(
            edges_sample_i=self.edge_df,
            source_column="source_node",
            target_column="target_node",
            edge_attrs=["LRP", "LRP_norm"],
            top_n_edges=None,
            sample_ID="Diff. graph",
        )
        self.diff_graph.get_communitites()

    def get_edges_and_nodes_vs_threshold(self):
        edge_node_df_sizes = []
        for diff_thres in np.arange(0.0, 1.0, 0.02):
            edge_df = fg.create_edge_dataframe_from_adj_diff(self.adj_diff, diff_thres)
            num_edges = len(edge_df)
            nodes = set(edge_df['source_node']).union(set(edge_df['target_node']))
            num_nodes = len(nodes)
            edge_node_df_sizes.append((diff_thres, num_edges, num_nodes))
        # convert to dataframe
        self.edge_node_df_sizes = pd.DataFrame(
            edge_node_df_sizes, columns=["threshold", "num_edges", "num_nodes"]
        )

    def run_GE_on_nodes(self, user_threshold=1e-2, max_temp_size=50):
        """
        Run gene enrichment analysis on the nodes of the differential graph using g:Profiler.
        Parameters:
        user_threshold (float): The significance threshold for the enrichment analysis. Default is 1e-2.
        max_temp_size (int): The maximum term size to filter out very general terms. Default is 50.
        Returns:
        pandas.DataFrame: A DataFrame containing the enrichment results, filtered and sorted by intersection size.
                          Returns None if the gene list is empty or if an error occurs during the analysis.
        Notes:
        - The gene list is obtained from the set of node names in the differential graph.
        - The analysis is performed for the organism 'hsapiens' (human).
        - The significance threshold method used is 'g_SCS' to correct for multiple testing.
        - If the gene list is empty, the function prints a message and returns None.
        - If an error occurs during the analysis, the function prints the error message and returns None.
        """
        # Run gene enrichment analysis using gprofiler on gene names.
        gene_list = list(self.diff_graph.set_of_node_names_no_type)  # ...existing value...
        if not gene_list:
            print("Gene list is empty; skipping GE analysis.")
            return None
        gp = GProfiler(return_dataframe=True)
        try:
            enrichment_results = gp.profile(organism='hsapiens',
                                             query=gene_list, 
                                            #sources =['GO:MF'], #only look into Gene Ontology terms.
                                            user_threshold=user_threshold, #reduce the significance threshold,
                                            significance_threshold_method='g_SCS') #use the g_SCS method to correct for multiple testing.
                                            
            # filter by 'term_size' to remove very general terms
            enrichment_results = enrichment_results[enrichment_results['term_size'] < max_temp_size]
            enrichment_results = enrichment_results.sort_values('intersection_size', ascending=False).reset_index(drop=True)

        except Exception as e:
            print("Error during GE analysis:", str(e))
            return None
        self.enrichment_results = enrichment_results
        return enrichment_results


# Example usage


# lrp_graph_diff = LRPGraphDiff(LRPGraph1, LRPGraph2)
# edges_vs_threshold = lrp_graph_diff.get_edges_vs_threshold()
