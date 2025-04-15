import numpy as np
import pandas as pd
from gprofiler import GProfiler
import os
class GE_Analyser:
    def __init__(self, gene_list):
        """
        Initializes an instance of the GE_Analyser class with a given list of genes.

        Args:
            gene_list (list): A list of gene identifiers to be used for analysis.

        Attributes:
            gene_list (list): Stores the input list of gene identifiers.
            ge_results (pandas.DataFrame or None): Placeholder for storing gene enrichment results.
            ge_verbalized (str or None): Placeholder for storing verbalized gene enrichment results.
        """
        self.gene_list = gene_list
        self.ge_results = None
        self.ge_verbalized = None

    def run_GE_on_nodes(self, user_threshold=1e-2, max_temp_size=50, on_aliases=False):
        """
        Run gene enrichment analysis on the provided gene list using g:Profiler.

        Args:
            user_threshold (float, optional): The significance threshold for the enrichment analysis. Default is 1e-2.
            max_temp_size (int, optional): The maximum term size to filter out very general terms. Default is 50.
            on_aliases (bool, optional): Placeholder for future functionality. Default is False.

        Returns:
            pandas.DataFrame or None: A DataFrame containing the enrichment results, filtered and sorted by intersection size.
                                      Returns None if the gene list is empty or if an error occurs during the analysis.

        Notes:
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
                user_threshold=user_threshold,
                no_evidences=True,
                significance_threshold_method="g_SCS",
            )

            # Filter by 'term_size' to remove very general terms
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
        Generate a human-readable summary of the gene enrichment analysis results.

        Returns:
            str or None: A human-readable string summarizing the enrichment results, or None if no results are available.

        Side Effects:
            - Prints the enrichment results or a message indicating their absence.
            - Sets the `ge_verbalized` attribute with the generated text.

        Notes:
            - The method checks if enrichment results are available before generating the summary.
            - If no enrichment results are found, it notifies the user.
        """
        if not hasattr(self, "ge_results") or self.ge_results is None:
            print("No enrichment results to verbalize.")
            return None

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
