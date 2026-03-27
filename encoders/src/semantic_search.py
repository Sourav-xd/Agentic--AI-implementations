import os
import yaml
import pandas as pd
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util

class SemanticSearch_ins:
    """
    Class to perform semantic search using embedding models.
    
    Attributes:
        ModelName_str (str): The name of the model to load from HuggingFace.
        Model_ins (SentenceTransformer): The loaded embedding model instance.
    """

    def __init__(self, ModelName_str: str):
        """
        Initializes the model based on configuration.
        
        Args:
            ModelName_str (str): Identifier for the model.
        """
        self.ModelName_str = ModelName_str
        self.Model_ins = SentenceTransformer(ModelName_str)

    def perform_search(self, Query_str: str, Documents_lst: List[str], TopK_int: int = 3) -> Dict[str, Any]:
        """
        Encodes query and documents, then performs cosine similarity search.
        
        Input:
            Query_str (str): User search query.
            Documents_lst (List[str]): List of document strings to search through.
            TopK_int (int): Number of top results to return.
            
        Output:
            Results_dct (Dict): Dictionary containing 'matches' (DataFrame) and 'top_score'.
        """
        # Encode inputs into dense vectors
        QueryEmbed_vec = self.Model_ins.encode(Query_str, convert_to_tensor=True)
        DocEmbeds_vec = self.Model_ins.encode(Documents_lst, convert_to_tensor=True)

        # Compute cosine similarity
        Scores_vec = util.cos_sim(QueryEmbed_vec, DocEmbeds_vec)[0]
        
        # Sort results
        TopResults_idx = Scores_vec.argsort(descending=True)[:TopK_int]
        
        SearchData_lst = []
        for idx in TopResults_idx:
            SearchData_lst.append({
                "document": Documents_lst[idx],
                "score": float(Scores_vec[idx])
            })

        # Organize into a proper DataFrame name
        SearchResults_df = pd.DataFrame(SearchData_lst)
        
        Results_dct = {
            "matches": SearchResults_df,
            "top_score_flt": float(Scores_vec[TopResults_idx[0]]) if len(TopResults_idx) > 0 else 0.0,
            "model_used_str": self.ModelName_str
        }
        
        return Results_dct


#test code
def run_task_01():
    # 1. Path Management
    BaseRelPath_str = "."
    BaseAbsPath_str = os.path.abspath(BaseRelPath_str)
    
    ConfigRelPath_str = os.path.join(BaseRelPath_str, "config/config.yaml")
    
    # 2. Load Config
    with open(ConfigRelPath_str, 'r') as file:
        Config_dct = yaml.safe_load(file)
    
    # 3. Setup Model via Config
    ModelID_str = Config_dct['models']['default_search']
    Search_ins = SemanticSearch_ins(ModelID_str)
    
    # 4. Input Data
    Query_str = "refund eligibility rules"
    Docs_lst = [
        "The return policy details how to get your money back.",
        "Our office is located in downtown New York.",
        "You can claim a refund within 30 days of purchase.",
        "The weather is sunny today."
    ]
    
    # 5. Execute
    FinalResults_dct = Search_ins.perform_search(Query_str, Docs_lst)
    
    print(f"Results using {FinalResults_dct['model_used_str']}:")
    print(FinalResults_dct['matches'])

if __name__ == "__main__":
    run_task_01()