import numpy as np
import pandas as pd
from gprofiler import GProfiler
import os

class GE_Analyser:
    def __init__(self, gene_list):
        """
        Initialize the gene enrichment analysis object with a gene list.
        Parameters:
        gene_list (iterable): A collection of gene names or gene identifiers.
        """
        self.gene_list = gene_list
        self.ge_results = None
        self.ge_verbalized = None



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
        gene_list = list(self.gene_list)
        
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
        self.ge_results = enrichment_results
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
        if not hasattr(self, "ge_results") or self.ge_results is None:
            print("No enrichment results to verbalize.")
            return

        ge_text = (
            f"Based on Gene Enrichment analysis, the entities:\n{self.gene_list}\n"
            "the following processes or pathways were found to be enriched:\n"
        )
        for i, row in self.ge_results.iterrows():
            if row["name"] == row["description"]:
                ge_text += f"\n-- {row['name']}"
            else:
                ge_text += f"\n-- {row['name']}; described as {row['description']}"

        self.ge_verbalized = ge_text
        return ge_text