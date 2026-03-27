import os
import yaml
import pandas as pd
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util

class CrossLingualRetriever_ins:
    """
    Class to perform retrieval across different languages using 
    multilingual embedding models.
    
    Attributes:
        ModelName_str (str): The multilingual model name (e.g., 'BAAI/bge-m3').
        Model_ins (SentenceTransformer): The model instance.
    """

    def __init__(self, ModelName_str: str):
        """
        Initializes the cross-lingual model.
        """
        self.ModelName_str = ModelName_str
        self.Model_ins = SentenceTransformer(ModelName_str)

    def retrieve_cross_lingual(self, Query_str: str, DocumentPool_dct: Dict[str, str], TopK_int: int = 2) -> Dict[str, Any]:
        """
        Retrieves the most semantically similar documents regardless of language.
        
        Input:
            Query_str (str): The search query (e.g., in English).
            DocumentPool_dct (Dict): Key is Language, Value is the text.
            TopK_int (int): Number of matches to return.
            
        Output:
            Results_dct (Dict): Best matches across languages.
        """
        LangKeys_lst = list(DocumentPool_dct.keys())
        Docs_lst = list(DocumentPool_dct.values())

        # 1. Encode query and the multilingual document pool
        QueryEmbed_vec = self.Model_ins.encode(Query_str, convert_to_tensor=True)
        DocEmbeds_vec = self.Model_ins.encode(Docs_lst, convert_to_tensor=True)

        # 2. Compute similarity in the shared space
        CosineScores_vec = util.cos_sim(QueryEmbed_vec, DocEmbeds_vec)[0]
        
        # 3. Rank and format
        TopIndices_idx = CosineScores_vec.argsort(descending=True)[:TopK_int]
        
        Matches_lst = []
        for idx in TopIndices_idx:
            Matches_lst.append({
                "detected_lang_str": LangKeys_lst[idx],
                "content_str": Docs_lst[idx],
                "alignment_score_flt": round(float(CosineScores_vec[idx]), 4)
            })

        CrossLingual_df = pd.DataFrame(Matches_lst)

        Results_dct = {
            "query_str": Query_str,
            "matches_df": CrossLingual_df,
            "best_match_lang_str": Matches_lst[0]["detected_lang_str"] if Matches_lst else None,
            "model_cfg_str": self.ModelName_str
        }

        return Results_dct

def run_task_07():
    # 1. Configuration (Relative and Absolute Paths)
    InpRelPath_str = "config/config.yaml"
    InpAbsPath_str = os.path.abspath(InpRelPath_str)

    with open(InpAbsPath_str, 'r') as file:
        Config_dct = yaml.safe_load(file)
    
    # Recommendation: Use BGE-M3 for state-of-the-art multilingual support in 2026
    ModelID_str = Config_dct['models']['multilingual_encoder'].get('multilingual_encoder', 'BAAI/bge-m3')
    
    # 2. Instance setup
    CL_ins = CrossLingualRetriever_ins(ModelID_str)

    # 3. Multilingual Document Pool (Mixed Languages)
    # The system doesn't need to know the language beforehand
    Docs_dct = {
        "English": "The annual report shows a 10% increase in revenue.",
        "Spanish": "El informe anual muestra un aumento del 10% en los ingresos.",
        "French": "Le rapport annuel montre une augmentation de 10% des revenus.",
        "German": "Der Jahresbericht zeigt eine Umsatzsteigerung von 10%.",
        "Italian": "Il tempo a Roma è molto bello oggi." # Irrelevant distractor
    }

    # 4. Search Query (Query in English, searching for meaning)
    SearchQuery_str = "How much did the company's earnings grow this year?"

    # 5. Execute
    FinalResults_dct = CL_ins.retrieve_cross_lingual(SearchQuery_str, Docs_dct)

    # 6. Output
    print(f"--- Cross-Lingual Search Results for: '{FinalResults_dct['query_str']}' ---")
    print(FinalResults_dct['matches_df'])

if __name__ == "__main__":
    run_task_07()