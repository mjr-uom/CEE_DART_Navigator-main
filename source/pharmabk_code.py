    
import numpy as np
import pandas as pd
import os

class pharmGKB_Analyzer:

    def __init__(self, gene_list):
        """
        Initializes the class with a list of genes and sets up placeholders for 
        filtered PharmGKB annotations and verbalized knowledge.
        Args:
            gene_list (list): A list of gene identifiers to be used for analysis.
        Attributes:
            gene_list (list): Stores the input list of gene identifiers.
            pharmGKB_var_drug_ann_filtered (NoneType): Placeholder for filtered 
                variant-drug annotations from PharmGKB.
            pharmGKB_var_pheno_ann_filtered (NoneType): Placeholder for filtered 
                variant-phenotype annotations from PharmGKB.
            pharmGKB_var_fa_ann_filtered (NoneType): Placeholder for filtered 
                variant-functional annotations from PharmGKB.
            pharmGKB_knowledge_verbalized (NoneType): Placeholder for verbalized 
                knowledge derived from PharmGKB data.
        """
        
        self.gene_list = gene_list
        self.pharmGKB_var_drug_ann_filtered = None
        self.pharmGKB_var_pheno_ann_filtered = None
        self.pharmGKB_var_fa_ann_filtered = None
        self.pharmGKB_knowledge_verbalized = None
        

    def get_pharmGKB_knowledge(self, files_path):
        """
        Extract and filter PharmGKB annotation data based on the gene list from the differential graph.
        This method loads PharmGKB annotation data from CSV files, filters the data based on the 
        gene list present in the differential graph, and stores the filtered data as attributes 
        of the class instance.
        Args:
            files_path (str): The file path to the directory containing the PharmGKB 
                annotation CSV files.
        Attributes:
            pharmGKB_var_pheno_ann_filtered (pd.DataFrame): Filtered variant-phenotype annotation data.
            pharmGKB_var_drug_ann_filtered (pd.DataFrame): Filtered variant-drug annotation data.
            pharmGKB_var_fa_ann_filtered (pd.DataFrame): Filtered variant-functional annotation data.
        """
        

        # Load PharmGKB annotation data
        var_pheno_ann = pd.read_csv(os.path.join(files_path, 'var_pheno_ann_simple.csv'))
        var_drug_ann = pd.read_csv(os.path.join(files_path, 'var_drug_ann_simple.csv'))
        var_fa_ann = pd.read_csv(os.path.join(files_path, 'var_fa_ann_simple.csv'))

        # Get the gene list from the differential graph
        gene_list = list(self.gene_list)

        # Define a helper function to filter dataframes by gene list
        def filter_by_gene(df, gene_list):
            if 'Gene' not in df.columns:
                raise ValueError("The required column 'Gene' is missing in the dataframe.")
            return df[df['Gene'].isin(gene_list)].reset_index(drop=True)

        # Filter the dataframes and store them as attributes
        self.pharmGKB_var_pheno_ann_filtered = filter_by_gene(var_pheno_ann, gene_list)
        self.pharmGKB_var_drug_ann_filtered = filter_by_gene(var_drug_ann, gene_list)
        self.pharmGKB_var_fa_ann_filtered = filter_by_gene(var_fa_ann, gene_list)

    def verbalise_pharmGKB_knowledge(self):
        """
        Verbalizes PharmGKB knowledge by consolidating and formatting information 
        from multiple dataframes into a human-readable text summary.
        This method performs the following steps:
        1. Concatenates the dataframes containing PharmGKB knowledge 
           (`pharmGKB_var_pheno_ann_filtered`, `pharmGKB_var_drug_ann_filtered`, 
           `pharmGKB_var_fa_ann_filtered`) into a single dataframe.
        2. Removes duplicate entries, sorts the data by the 'Gene' column, 
           and resets the index.
        3. Constructs a textual summary of the knowledge for each gene, 
           including associated drugs, statements, notes, and PubMed IDs (PMIDs).
        Returns:
            str: A formatted string summarizing the PharmGKB knowledge for the 
            genes in the differential graph.
        """
        # concat all the dataframes with pharmGKB knowledge into one
        pharmGKB_knowledge = pd.concat([self.pharmGKB_var_pheno_ann_filtered, self.pharmGKB_var_drug_ann_filtered, self.pharmGKB_var_fa_ann_filtered])
        # sort the dataframe by 'Gene' column
        pharmGKB_knowledge = pharmGKB_knowledge.sort_values(by='Gene').reset_index(drop=True)

        # verbalize the knowledge
        required_columns = ['Gene', 'Drug(s)', 'Sentence', 'Notes', 'PMID']
        for col in required_columns:
            if col not in pharmGKB_knowledge.columns:
                raise ValueError(f"The required column '{col}' is missing in the dataframe.")
        
        pharmGKB_knowledge_text = f"Based on PharmGKB knowledge, the following information was found for the genes in the differential graph:\n"
        for i, row in pharmGKB_knowledge.iterrows():
            pharmGKB_knowledge_text += f"\nGene: {row['Gene']}; "
            pharmGKB_knowledge_text += f"Drug(s): {row['Drug(s)']}; "
            pharmGKB_knowledge_text += f"Statement: {row['Sentence']}; "
            pharmGKB_knowledge_text += f"Notes: {row['Notes']}; "
            pharmGKB_knowledge_text += f"(PMID: {row['PMID']})."
            pharmGKB_knowledge_text += f"Notes: {row['Notes']}; "
            pharmGKB_knowledge_text += f"(PMID: {row['PMID']})."

        self.pharmGKB_knowledge_verbalized = pharmGKB_knowledge_text
        return pharmGKB_knowledge_text
        