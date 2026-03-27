import os
import yaml
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util

class ZeroShotClassifier_ins:
    """
    Class to perform zero-shot classification by matching text embeddings 
    to label embeddings.
    
    Attributes:
        ModelName_str (str): The name of the embedding model.
        Encoder_ins (SentenceTransformer): The model instance.
    """

    def __init__(self, ModelName_str: str):
        """
        Initializes the zero-shot engine.
        """
        self.ModelName_str = ModelName_str
        self.Encoder_ins = SentenceTransformer(ModelName_str)

    def classify(self, InputText_str: str, CandidateLabels_lst: List[str]) -> Dict[str, Any]:
        """
        Predicts the label for a text input without any prior training on the labels.
        
        Input:
            InputText_str (str): The sentence to classify.
            CandidateLabels_lst (List[str]): A list of potential category names.
            
        Output:
            Results_dct (Dict): Sorted probabilities for each label.
        """
        # 1. We wrap labels in a prompt to improve semantic alignment
        # This is a common practice in Zero-Shot learning
        PromptLabels_lst = [f"This text is about {label}" for label in CandidateLabels_lst]

        # 2. Encode the input text and the label prompts
        TextEmbed_vec = self.Encoder_ins.encode(InputText_str, convert_to_tensor=True)
        LabelEmbeds_vec = self.Encoder_ins.encode(PromptLabels_lst, convert_to_tensor=True)

        # 3. Compute Cosine Similarity
        CosineScores_vec = util.cos_sim(TextEmbed_vec, LabelEmbeds_vec)[0]
        
        # 4. Convert scores to a distribution (Softmax-like) for 'confidence'
        # We use simple normalization here for transparency
        Scores_arr = CosineScores_vec.cpu().numpy()
        ExpScores_arr = np.exp(Scores_arr)
        Probs_arr = ExpScores_arr / np.sum(ExpScores_arr)

        # 5. Build Result DataFrame
        ClassResults_df = pd.DataFrame({
            "label_str": CandidateLabels_lst,
            "confidence_flt": [round(float(p), 4) for p in Probs_arr]
        }).sort_values(by="confidence_flt", ascending=False)

        Results_dct = {
            "top_label_str": ClassResults_df.iloc[0]["label_str"],
            "all_predictions_df": ClassResults_df,
            "input_text_str": InputText_str,
            "model_cfg_str": self.ModelName_str
        }

        return Results_dct

def run_task_08():
    # 1. Path and Config Logic
    InpRelPath_str = "config/config.yaml"
    InpAbsPath_str = os.path.abspath(InpRelPath_str)

    with open(InpAbsPath_str, 'r') as file:
        Config_dct = yaml.safe_load(file)
    
    # Using a high-quality model for semantic alignment
    ModelID_str = Config_dct['models'].get('zero_shot_encoder', 'all-mpnet-base-v2')
    
    # 2. Instance setup
    ZS_ins = ZeroShotClassifier_ins(ModelID_str)

    # 3. Test Inputs (New unseen data)
    Ticket_str = "The checkout button is overlapping with the sidebar and I cannot click it."
    
    # Labels that the model has never been specifically 'trained' to recognize
    PossibleCategories_lst = ["UI Bug", "Billing Issue", "Feature Request", "Account Security"]

    # 4. Execute Classification
    FinalResults_dct = ZS_ins.classify(Ticket_str, PossibleCategories_lst)

    # 5. Output display
    print(f"--- Zero-Shot Task Output ---")
    print(f"Input: {FinalResults_dct['input_text_str']}")
    print(f"Predicted Category: {FinalResults_dct['top_label_str']}")
    print("\nConfidence Scores:")
    print(FinalResults_dct['all_predictions_df'])

if __name__ == "__main__":
    run_task_08()