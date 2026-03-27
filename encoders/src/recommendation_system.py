import os
import yaml
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util

class ContentRecommender_ins:
    """
    Class to recommend items based on semantic similarity of descriptions.
    
    Attributes:
        ModelName_str (str): The embedding model for product vectorization.
        Encoder_ins (SentenceTransformer): The model instance.
    """

    def __init__(self, ModelName_str: str):
        """
        Initializes the recommendation engine.
        """
        self.ModelName_str = ModelName_str
        self.Encoder_ins = SentenceTransformer(ModelName_str)

    def get_recommendations(self, TargetItem_str: str, ItemCatalog_lst: List[str], TopK_int: int = 3) -> Dict[str, Any]:
        """
        Finds the most similar items to a target input from a catalog.
        
        Input:
            TargetItem_str (str): The product/article description the user is currently viewing.
            ItemCatalog_lst (List[str]): The database of available items.
            TopK_int (int): Number of recommendations to return.
            
        Output:
            Results_dct (Dict): Recommended items and similarity scores.
        """
        # 1. Vectorize the target and the catalog
        TargetEmbed_vec = self.Encoder_ins.encode(TargetItem_str, convert_to_tensor=True)
        CatalogEmbeds_vec = self.Encoder_ins.encode(ItemCatalog_lst, convert_to_tensor=True)

        # 2. Compute Cosine Similarity
        CosineScores_vec = util.cos_sim(TargetEmbed_vec, CatalogEmbeds_vec)[0]

        # 3. Rank and Filter (excluding the target item itself if it's in the catalog)
        TopResults_idx = CosineScores_vec.argsort(descending=True)
        
        Recs_lst = []
        for idx in TopResults_idx:
            # Simple check to avoid recommending the exact same string
            if ItemCatalog_lst[idx].strip().lower() == TargetItem_str.strip().lower():
                continue
                
            Recs_lst.append({
                "recommended_item": ItemCatalog_lst[idx],
                "similarity_score_flt": round(float(CosineScores_vec[idx]), 4)
            })
            
            if len(Recs_lst) >= TopK_int:
                break

        # 4. Result Formatting
        Recs_df = pd.DataFrame(Recs_lst)
        
        Results_dct = {
            "recommendations_df": Recs_df,
            "target_input_str": TargetItem_str,
            "model_cfg_str": self.ModelName_str
        }
        
        return Results_dct

def run_task_05():
    # 1. Config Management
    InpRelPath_str = "config/config.yaml"
    InpAbsPath_str = os.path.abspath(InpRelPath_str)

    with open(InpAbsPath_str, 'r') as file:
        Config_dct = yaml.safe_load(file)
    
    # Model configuration from global config
    ModelID_str = Config_dct['models'].get('recommender_encoder', 'all-MiniLM-L6-v2')
    
    # 2. Instance setup
    RecEngine_ins = ContentRecommender_ins(ModelID_str)

    # 3. Sample Item Catalog (e.g., a Library or Store)
    Catalog_lst = [
        "Noise-cancelling wireless headphones with 30-hour battery life.",
        "Ergonomic mechanical keyboard with RGB lighting.",
        "High-definition web camera for professional video calls.",
        "Lightweight running shoes with breathable mesh.",
        "Compact Bluetooth speaker with waterproof design.",
        "Smartwatch with heart rate monitor and GPS tracking."
    ]

    # 4. Current View (User is looking at a similar product)
    CurrentProduct_str = "Over-ear headphones with long battery and sound isolation."

    # 5. Execute Recommendation
    FinalResults_dct = RecEngine_ins.get_recommendations(CurrentProduct_str, Catalog_lst)

    # 6. Output
    print(f"--- Recommendations for: '{FinalResults_dct['target_input_str']}' ---")
    print(FinalResults_dct['recommendations_df'])

if __name__ == "__main__":
    run_task_05()