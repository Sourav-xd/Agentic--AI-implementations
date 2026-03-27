import os
import yaml
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class DocumentClustering_ins:
    """
    Class to group documents into clusters using embeddings and K-Means.
    
    Attributes:
        ModelName_str (str): The embedding model used for vectorization.
        Encoder_ins (SentenceTransformer): The model instance.
        Cluster_ins (KMeans): The clustering algorithm instance.
    """

    def __init__(self, ModelName_str: str, NumClusters_int: int = 3):
        """
        Initializes the clustering engine.
        
        Args:
            ModelName_str (str): Embedding model identifier.
            NumClusters_int (int): The number of clusters (K) to discover.
        """
        self.ModelName_str = ModelName_str
        self.NumClusters_int = NumClusters_int
        self.Encoder_ins = SentenceTransformer(ModelName_str)
        self.Cluster_ins = KMeans(n_clusters=NumClusters_int, random_state=42, n_init=10)

    def group_documents(self, Documents_lst: List[str]) -> Dict[str, Any]:
        """
        Embeds documents and performs unsupervised clustering.
        
        Input:
            Documents_lst (List[str]): Raw text documents to group.
            
        Output:
            Results_dct (Dict): Contains the cluster assignments and Silhouette Score.
        """
        # 1. Feature Extraction (Embeddings)
        Embeddings_vec = self.Encoder_ins.encode(Documents_lst)

        # 2. Fit K-Means
        self.Cluster_ins.fit(Embeddings_vec)
        Labels_arr = self.Cluster_ins.labels_
        
        # 3. Evaluation (Silhouette Score measures cluster separation)
        # Ranges from -1 (bad) to +1 (perfect)
        Score_flt = silhouette_score(Embeddings_vec, Labels_arr)

        # 4. Organize Results
        Clustered_df = pd.DataFrame({
            "document_text": Documents_lst,
            "cluster_id_int": Labels_arr
        }).sort_values(by="cluster_id_int")

        Results_dct = {
            "clustered_data_df": Clustered_df,
            "silhouette_score_flt": round(float(Score_flt), 4),
            "k_value_int": self.NumClusters_int,
            "model_used_str": self.ModelName_str
        }
        
        return Results_dct

def run_task_04():
    # 1. Paths and Config
    InpRelPath_str = "config/config.yaml"
    InpAbsPath_str = os.path.abspath(InpRelPath_str)

    with open(InpAbsPath_str, 'r') as file:
        Config_dct = yaml.safe_load(file)
    
    # Model and Parameter selection from config
    ModelID_str = Config_dct['models'].get('clustering_encoder', 'all-MiniLM-L6-v2')
    TargetK_int = 3  # Assume we are looking for 3 main topics
    
    # 2. Instance setup
    Clusterer_ins = DocumentClustering_ins(ModelID_str, NumClusters_int=TargetK_int)

    # 3. Unlabeled Corpus (Diverse topics: Space, Finance, Cooking)
    Corpus_lst = [
        "The James Webb telescope captured images of distant galaxies.",
        "Stellar evolution explains how stars are born and die.",
        "NASA is planning a manned mission to Mars by 2030.",
        "The stock market saw a significant dip in tech shares today.",
        "Inflation rates are affecting consumer spending globally.",
        "Federal reserve interest rates remain unchanged.",
        "Add two teaspoons of salt to the boiling pasta water.",
        "Slow-cook the beef brisket for at least eight hours.",
        "The secret to a good crust is using chilled butter."
    ]

    # 4. Execute Task
    FinalResults_dct = Clusterer_ins.group_documents(Corpus_lst)

    # 5. Output display
    print(f"--- Clustering Results (K={FinalResults_dct['k_value_int']}) ---")
    print(f"Silhouette Score: {FinalResults_dct['silhouette_score_flt']}")
    print("\nGrouped Documents:")
    print(FinalResults_dct['clustered_data_df'])

if __name__ == "__main__":
    run_task_04()