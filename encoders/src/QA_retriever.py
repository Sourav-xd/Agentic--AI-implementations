import os
import yaml
import pandas as pd
from typing import List, Dict, Any, Union
from sentence_transformers import SentenceTransformer, util

class RetrievalQA_ins:
    """
    Class to retrieve relevant document passages for Question Answering.
    
    Attributes:
        ModelName_str (str): The name of the model to load from config.
        Model_ins (SentenceTransformer): The embedding model instance.
    """

    def __init__(self, ModelName_str: str):
        """
        Initializes the QA retriever.
        
        Args:
            ModelName_str (str): Name/Path of the embedding model.
        """
        self.ModelName_str = ModelName_str
        # Specialized models for QA like 'multi-qa-mpnet-base-dot-v1' are preferred
        self.Model_ins = SentenceTransformer(ModelName_str)

    def get_context(self, Question_str: str, Corpus_lst: List[str], TopK_int: int = 5) -> Dict[str, Any]:
        """
        Finds the top-N passages from a corpus that likely contain the answer.
        
        Input:
            Question_str (str): The user's question.
            Corpus_lst (List[str]): List of passages/documents.
            TopK_int (int): How many context snippets to retrieve.
            
        Output:
            Results_dct (Dict): Dictionary containing retrieved context and metadata.
        """
        # Encode question and corpus
        QuestionEmbed_vec = self.Model_ins.encode(Question_str, convert_to_tensor=True)
        CorpusEmbeds_vec = self.Model_ins.encode(Corpus_lst, convert_to_tensor=True)

        # Semantic similarity (Cosine similarity is standard for these models)
        Hits_lst = util.semantic_search(QuestionEmbed_vec, CorpusEmbeds_vec, top_k=TopK_int)[0]
        
        Context_lst = []
        for hit in Hits_lst:
            Context_lst.append({
                "passage_idx_int": hit['corpus_id'],
                "content_str": Corpus_lst[hit['corpus_id']],
                "confidence_score_flt": round(hit['score'], 4)
            })

        # Organize into a clean DataFrame for the result dictionary
        Context_df = pd.DataFrame(Context_lst)
        
        # Consolidate results
        Results_dct = {
            "retrieved_context_df": Context_df,
            "best_passage_str": Context_lst[0]["content_str"] if Context_lst else "",
            "total_retrieved_int": len(Context_lst),
            "config_model_str": self.ModelName_str
        }
        
        return Results_dct

def run_task_02():
    # 1. Path Management (Absolute vs Relative)
    InpRelPath_str = "config/config.yaml"
    InpAbsPath_str = os.path.abspath(InpRelPath_str)
    
    # 2. Load Config to fetch generic model settings
    if not os.path.exists(InpAbsPath_str):
        print(f"Error: Config not found at {InpAbsPath_str}")
        return

    with open(InpAbsPath_str, 'r') as file:
        Config_dct = yaml.safe_load(file)
    
    # Model name fetched dynamically from config
    ModelID_str = Config_dct['models'].get('qa_retrieval', 'multi-qa-mpnet-base-dot-v1')
    
    # 3. Instance creation
    QA_ins = RetrievalQA_ins(ModelID_str)
    
    # 4. Mock Corpus (A collection of knowledge)
    KnowledgeBase_lst = [
        "To claim a refund, visit the 'My Orders' section and click 'Request Refund'.",
        "Our shipping takes 3-5 business days for domestic orders.",
        "Refunds are processed within 7-10 business days after approval.",
        "The standard warranty for all electronics is 2 years.",
        "Subscription cancellations must be made 24 hours before renewal."
    ]
    
    UserQuestion_str = "How long does it take to get my money back?"
    
    # 5. Execute Task
    QA_Results_dct = QA_ins.get_context(UserQuestion_str, KnowledgeBase_lst, TopK_int=2)
    
    # Output display
    print(f"--- Question QA Task using {ModelID_str} ---")
    print(f"Question: {UserQuestion_str}")
    print("\nTop Retrieved Contexts:")
    print(QA_Results_dct['retrieved_context_df'])

if __name__ == "__main__":
    run_task_02()