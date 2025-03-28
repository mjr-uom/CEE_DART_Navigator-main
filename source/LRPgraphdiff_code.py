# get the adjacency matrices
import functions_graphs as fg
import LRPGraph_code as lrpgraph
import numpy as np
import pandas as pd
from gprofiler import GProfiler
import os

class LRPGraphDiff:
    def __init__(self, LRPGraph1, LRPGraph2, diff_thres=0.6):
        """
        Initializes the comparison of two LRPGraph objects and computes their differences.

        Args:
            LRPGraph1 (LRPGraph): The first LRPGraph object to compare.
            LRPGraph2 (LRPGraph): The second LRPGraph object to compare.
            diff_thres (float, optional): The threshold for determining significant differences
                in edge attributes. Defaults to 0.6.

        Attributes:
            LRPGraph1 (LRPGraph): The first LRPGraph object.
            LRPGraph2 (LRPGraph): The second LRPGraph object.
            diff_thres (float): The threshold for significant differences.
            adj_diff (numpy.ndarray): The adjacency difference matrix calculated from the two graphs.
            edge_df (pandas.DataFrame): A DataFrame containing edges with differences above the threshold.
            diff_graph (LRPGraph): An LRPGraph object representing the difference graph,
                including significant edges and their attributes.
        """
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
        """
        Computes the number of edges and nodes in a graph for varying threshold values
        and stores the results in a DataFrame.

        This method iterates over a range of threshold values from 0.0 to 1.0 with a step
        size of 0.02. For each threshold, it creates an edge DataFrame from the adjacency
        difference matrix, calculates the number of edges, and determines the unique nodes
        involved. The results are stored as a DataFrame with columns for the threshold,
        number of edges, and number of nodes.

        Attributes:
            self.edge_node_df_sizes (pd.DataFrame): A DataFrame containing the threshold
                values, number of edges, and number of nodes for each threshold.

        Returns:
            None
        """
        edge_node_df_sizes = []
        for diff_thres in np.arange(0.0, 1.0, 0.02):
            edge_df = fg.create_edge_dataframe_from_adj_diff(self.adj_diff, diff_thres)
            num_edges = len(edge_df)
            nodes = set(edge_df["source_node"]).union(set(edge_df["target_node"]))
            num_nodes = len(nodes)
            edge_node_df_sizes.append((diff_thres, num_edges, num_nodes))
        # convert to dataframe
        self.edge_node_df_sizes = pd.DataFrame(
            edge_node_df_sizes, columns=["threshold", "num_edges", "num_nodes"]
        )

    def run_GE_on_nodes(self, user_threshold=1e-2, max_temp_size=50, on_aliases=False):
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
        if on_aliases:
            gene_list = list(self.diff_graph.node_names_aliases)
        else:
            gene_list = list(self.diff_graph.node_names_no_type)
        print(gene_list)
        if not gene_list:
            print("Gene list is empty; skipping GE analysis.")
            return None
        
        gp = GProfiler(return_dataframe=True)
        try:
            enrichment_results = gp.profile(
                organism="hsapiens",
                query=gene_list,
                # sources =['GO:MF'], #only look into Gene Ontology terms.
                user_threshold=user_threshold,  # reduce the significance threshold,
                no_evidences=True,
                significance_threshold_method="g_SCS",
            )  # use the g_SCS method to correct for multiple testing.

            # filter by 'term_size' to remove very general terms
            enrichment_results = enrichment_results[
                enrichment_results["term_size"] < max_temp_size
            ]
            enrichment_results = enrichment_results.sort_values(
                "intersection_size", ascending=False
            ).reset_index(drop=True)

        except Exception as e:
            print("Error during GE analysis:", str(e))
            return None
        self.enrichment_results = enrichment_results
        return enrichment_results

    def verbalize_enrichment_results(self):
        """
        Verbalize and print the gene enrichment analysis results in a human-readable format.
        This method checks if enrichment results are available and generates a textual
        representation of the enriched processes or pathways based on the analysis.
        If no enrichment results are found, it notifies the user.
        Attributes:
            enrichment_results (DataFrame): A DataFrame containing the enrichment analysis results.
                Each row should have at least the following columns:
                - 'name': The name of the enriched process or pathway.
                - 'description': A description of the enriched process or pathway.
            diff_graph.set_of_node_names_no_type (iterable): A collection of entity names
                involved in the analysis.
        Returns:
            str: A human-readable string summarizing the enrichment results, or a message
            indicating that no results are available.
        Side Effects:
            - Prints the enrichment results or a message indicating their absence.
            - Sets the `ge_verbalized` attribute with the generated text.

        """
        if not hasattr(self, "enrichment_results") or self.enrichment_results is None:
            print("No enrichment results to verbalize.")
            return

        ge_text = (
            f"Based on Gene Enrichment analysis, the entities:\n{self.diff_graph.node_names_no_type}\n"
            "the following processes or pathways were found to be enriched:\n"
        )
        for i, row in self.enrichment_results.iterrows():
            if row["name"] == row["description"]:
                ge_text += f"\n-- {row['name']}"
            else:
                ge_text += f"\n-- {row['name']}; described as {row['description']}"

        self.ge_verbalized = ge_text
        return ge_text


    def get_pharmakb_knowledge(self, files_path=r'C:\Users\owysocky\Documents\GitHub\CCE_scGeneRAI\resources\pharmgkb'):
        """
        Extract and filter PharmGKB annotation data based on the gene list from the differential graph.
        This method loads PharmGKB annotation data from CSV files, filters the data based on the 
        gene list present in the differential graph, and stores the filtered data as attributes 
        of the class instance.
        Args:
            files_path (str, optional): The file path to the directory containing the PharmGKB 
                annotation CSV files. Defaults to 
                r'C:\Users\owysocky\Documents\GitHub\CCE_scGeneRAI\resources\pharmgkb'.
        Attributes:
            pharmakb_var_pheno_ann_filtered (pd.DataFrame): Filtered variant-phenotype annotation data.
            pharmakb_var_drug_ann_filtered (pd.DataFrame): Filtered variant-drug annotation data.
            pharmakb_var_fa_ann_filtered (pd.DataFrame): Filtered variant-functional annotation data.
        """
        

        # Load PharmGKB annotation data
        var_pheno_ann = pd.read_csv(os.path.join(files_path, 'var_pheno_ann_simple.csv'))
        var_drug_ann = pd.read_csv(os.path.join(files_path, 'var_drug_ann_simple.csv'))
        var_fa_ann = pd.read_csv(os.path.join(files_path, 'var_fa_ann_simple.csv'))

        # Get the gene list from the differential graph
        gene_list = list(self.diff_graph.node_names_aliases)

        # Define a helper function to filter dataframes by gene list
        def filter_by_gene(df, gene_list):
            return df[df['Gene'].isin(gene_list)].reset_index(drop=True)

        # Filter the dataframes and store them as attributes
        self.pharmakb_var_pheno_ann_filtered = filter_by_gene(var_pheno_ann, gene_list)
        self.pharmakb_var_drug_ann_filtered = filter_by_gene(var_drug_ann, gene_list)
        self.pharmakb_var_fa_ann_filtered = filter_by_gene(var_fa_ann, gene_list)

    def verbalise_pharmakb_knowledge(self):
        """
        Verbalizes PharmGKB knowledge by consolidating and formatting information 
        from multiple dataframes into a human-readable text summary.
        This method performs the following steps:
        1. Concatenates the dataframes containing PharmGKB knowledge 
           (`pharmakb_var_pheno_ann_filtered`, `pharmakb_var_drug_ann_filtered`, 
           `pharmakb_var_fa_ann_filtered`) into a single dataframe.
        2. Removes duplicate entries, sorts the data by the 'Gene' column, 
           and resets the index.
        3. Constructs a textual summary of the knowledge for each gene, 
           including associated drugs, statements, notes, and PubMed IDs (PMIDs).
        Returns:
            str: A formatted string summarizing the PharmGKB knowledge for the 
            genes in the differential graph.
        """
        # concat all the dataframes with pharmakb knowledge into one
        pharmakb_knowledge = pd.concat([self.pharmakb_var_pheno_ann_filtered, self.pharmakb_var_drug_ann_filtered, self.pharmakb_var_fa_ann_filtered])
        # drop duplicates
        pharmakb_knowledge = pharmakb_knowledge.drop_duplicates().sort_values(by='Gene').reset_index(drop=True)
        # verbalize the knowledge
        pharmakb_knowledge_text = f"Based on PharmGKB knowledge, the following information was found for the genes in the differential graph:\n"
        for i, row in pharmakb_knowledge.iterrows():
            pharmakb_knowledge_text += f"\nGene: {row['Gene']}; "
            pharmakb_knowledge_text += f"Drug(s): {row['Drug(s)']}; "
            pharmakb_knowledge_text += f"Statement: {row['Sentence']}; "
            pharmakb_knowledge_text += f"Notes: {row['Notes']}; "
            pharmakb_knowledge_text += f"(PMID: {row['PMID']})."

        self.pharmakb_knowledge_verbalized = pharmakb_knowledge_text
        return pharmakb_knowledge_text
        



# Example usage

# lrp_graph_diff = LRPGraphDiff(LRPGraph1, LRPGraph2)
# edges_vs_threshold = lrp_graph_diff.get_edges_vs_threshold()
