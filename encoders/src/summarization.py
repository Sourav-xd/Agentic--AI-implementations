import os
import yaml
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util

class ExtractiveSummarizer_ins:
    """
    Class to perform extractive summarization by ranking sentences 
    based on semantic centrality.
    
    Attributes:
        ModelName_str (str): The name of the embedding model.
        Encoder_ins (SentenceTransformer): The model instance.
    """

    def __init__(self, ModelName_str: str):
        """
        Initializes the summarizer.
        """
        self.ModelName_str = ModelName_str
        self.Encoder_ins = SentenceTransformer(ModelName_str)

    def summarize(self, FullText_str: str, SummarySize_int: int = 3) -> Dict[str, Any]:
        """
        Summarizes text by selecting the most semantically central sentences.
        
        Input:
            FullText_str (str): The long document to summarize.
            SummarySize_int (int): How many sentences to include in the summary.
            
        Output:
            Results_dct (Dict): Summary text and ranking details.
        """
        # 1. Preprocessing: Split text into sentences
        # Simple split used here; for production, use a library like nltk or spacy
        Sentences_lst = [s.strip() for s in FullText_str.split('.') if len(s.strip()) > 10]
        
        if len(Sentences_lst) <= SummarySize_int:
            return {"summary_str": FullText_str, "status": "Text too short for summarization"}

        # 2. Encode all sentences
        SentenceEmbeds_vec = self.Encoder_ins.encode(Sentences_lst, convert_to_tensor=True)

        # 3. Create 'Document Centroid' (Mean of all sentence vectors)
        DocCentroid_vec = SentenceEmbeds_vec.mean(dim=0, keepdim=True)

        # 4. Calculate similarity of each sentence to the document centroid
        CosineScores_vec = util.cos_sim(SentenceEmbeds_vec, DocCentroid_vec)[:, 0]

        # 5. Rank sentences and select top-N
        TopIndices_arr = CosineScores_vec.argsort(descending=True)[:SummarySize_int]
        # Sort indices to maintain the original flow of the story
        SelectedIndices_lst = sorted(TopIndices_arr.tolist())

        SummarySentences_lst = [Sentences_lst[idx] for idx in SelectedIndices_lst]
        Summary_str = ". ".join(SummarySentences_lst) + "."

        # 6. Organize metrics into a DataFrame
        Ranking_df = pd.DataFrame({
            "sentence": Sentences_lst,
            "centrality_score_flt": CosineScores_vec.tolist()
        }).sort_values(by="centrality_score_flt", ascending=False)

        Results_dct = {
            "summary_str": Summary_str,
            "ranking_details_df": Ranking_df,
            "num_sentences_int": len(SummarySentences_lst),
            "model_cfg_str": self.ModelName_str
        }

        return Results_dct

def run_task_06():
    # 1. Config/Path Management
    InpRelPath_str = "config/config.yaml"
    InpAbsPath_str = os.path.abspath(InpRelPath_str)

    with open(InpAbsPath_str, 'r') as file:
        Config_dct = yaml.safe_load(file)
    
    ModelID_str = Config_dct['models'].get('summarization_encoder', 'all-MiniLM-L6-v2')
    
    # 2. Instance setup
    Summarizer_ins = ExtractiveSummarizer_ins(ModelID_str)

    # 3. Sample Article (Topic: Artificial Intelligence)
    Article_str = (
        "Artificial Intelligence has transformed the modern technology landscape. "
        "It involves creating algorithms that can simulate human intelligence. "
        "Machine learning is a subset of AI focused on building systems that learn from data. "
        "Deep learning uses neural networks with many layers to solve complex problems. "
        "Despite its benefits, ethical concerns regarding bias and privacy remain prominent. "
        "Regulators are working on frameworks to ensure AI is used responsibly. "
        "The future of AI holds potential for breakthroughs in medicine and energy."
    )

    # 4. Execute
    FinalResults_dct = Summarizer_ins.summarize(Article_str, SummarySize_int=3)

    # 5. Output
    print(f"--- Extractive Summary (Size: {FinalResults_dct['num_sentences_int']}) ---")
    print(f"Summary: {FinalResults_dct['summary_str']}")
    print("\nTop Ranked Sentences by Centrality:")
    print(FinalResults_dct['ranking_details_df'].head(3))

if __name__ == "__main__":
    run_task_06()