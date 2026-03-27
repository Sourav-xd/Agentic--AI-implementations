import os
import yaml
import pandas as pd
from typing import List, Dict, Any
from transformers import pipeline

class MaskedLanguageModel_ins:
    """
    Class to perform Fill-in-the-Blank (MLM) tasks using encoder models.
    
    Attributes:
        ModelName_str (str): The name of the transformer model (e.g., 'bert-base-uncased').
        Pipeline_ins (pipeline): The HuggingFace fill-mask pipeline.
    """

    def __init__(self, ModelName_str: str):
        """
        Initializes the MLM pipeline.
        """
        self.ModelName_str = ModelName_str
        # We use the 'fill-mask' pipeline which is specific to MLM tasks
        self.Pipeline_ins = pipeline("fill-mask", model=ModelName_str)

    def fill_blank(self, MaskedText_str: str, TopK_int: int = 5) -> Dict[str, Any]:
        """
        Predicts the most likely tokens to replace the [MASK] in a sentence.
        
        Input:
            MaskedText_str (str): Sentence containing the mask token (e.g., [MASK] or <mask>).
            TopK_int (int): Number of suggestions to return.
            
        Output:
            Results_dct (Dict): Predictions with scores and the completed strings.
        """
        # 1. Execute prediction
        Predictions_lst = self.Pipeline_ins(MaskedText_str, top_k=TopK_int)

        # 2. Format results into a list of dictionaries
        FormattedResults_lst = []
        for pred in Predictions_lst:
            FormattedResults_lst.append({
                "predicted_token_str": pred['token_str'],
                "score_flt": round(float(pred['score']), 4),
                "completed_sentence_str": pred['sequence']
            })

        # 3. Organize into a DataFrame
        Predictions_df = pd.DataFrame(FormattedResults_lst)

        Results_dct = {
            "predictions_df": Predictions_df,
            "original_mask_str": MaskedText_str,
            "best_guess_str": FormattedResults_lst[0]["predicted_token_str"],
            "model_cfg_str": self.ModelName_str
        }

        return Results_dct

def run_task_09():
    # 1. Configuration Management
    InpRelPath_str = "config/config.yaml"
    InpAbsPath_str = os.path.abspath(InpRelPath_str)

    with open(InpAbsPath_str, 'r') as file:
        Config_dct = yaml.safe_load(file)
    
    # Selecting an MLM-capable model (BERT, RoBERTa, or DistilBERT)
    ModelID_str = Config_dct['models'].get('mlm_encoder', 'distilbert-base-uncased')
    
    # 2. Instance setup
    MLM_ins = MaskedLanguageModel_ins(ModelID_str)

    # 3. Masked Input
    # Note: Different models use different mask tokens (BERT uses [MASK], RoBERTa uses <mask>)
    InputWithMask_str = "The stock market saw a significant [MASK] in tech shares today."

    # 4. Execute Prediction
    FinalResults_dct = MLM_ins.fill_blank(InputWithMask_str)

    # 5. Output display
    print(f"--- MLM Task Output using {FinalResults_dct['model_cfg_str']} ---")
    print(f"Input: {FinalResults_dct['original_mask_str']}")
    print(f"Best Guess: {FinalResults_dct['best_guess_str']}")
    print("\nTop Suggestions:")
    print(FinalResults_dct['predictions_df'])

if __name__ == "__main__":
    run_task_09()