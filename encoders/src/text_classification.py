import os
import yaml
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

class TextClassifier_ins:
    """
    Class to perform text classification using embeddings as features.
    
    Attributes:
        ModelName_str (str): The embedding model used for feature extraction.
        Encoder_ins (SentenceTransformer): The embedding model instance.
        Classifier_ins (LogisticRegression): The ML classifier (e.g., Logistic Regression).
    """

    def __init__(self, ModelName_str: str):
        """
        Initializes the classifier with a specific embedding model.
        """
        self.ModelName_str = ModelName_str
        self.Encoder_ins = SentenceTransformer(ModelName_str)
        # We use Logistic Regression as a standard, robust head for embedding features
        self.Classifier_ins = LogisticRegression(max_iter=1000, class_weight='balanced')

    def train(self, TrainTexts_lst: List[str], TrainLabels_lst: List[Union[str, int]]) -> Dict[str, Any]:
        """
        Trains the classifier on the provided text-label pairs.
        
        Input:
            TrainTexts_lst (List[str]): Training sentences.
            TrainLabels_lst (List): Corresponding class labels.
            
        Output:
            Results_dct (Dict): Training metadata and status.
        """
        # Convert text to dense feature vectors
        X_train_vec = self.Encoder_ins.encode(TrainTexts_lst)
        y_train_arr = np.array(TrainLabels_lst)

        # Fit the classifier
        self.Classifier_ins.fit(X_train_vec, y_train_arr)

        Results_dct = {
            "status_str": "Success",
            "feature_dim_int": X_train_vec.shape[1],
            "train_size_int": len(TrainTexts_lst)
        }
        return Results_dct

    def predict(self, TestTexts_lst: List[str], ActualLabels_lst: List[Any] = None) -> Dict[str, Any]:
        """
        Predicts classes for new texts and evaluates performance if labels are provided.
        
        Input:
            TestTexts_lst (List[str]): Texts to classify.
            ActualLabels_lst (List): Optional true labels for metric calculation.
            
        Output:
            FinalResults_dct (Dict): Predictions, Accuracy, and Classification Report.
        """
        # Feature extraction
        X_test_vec = self.Encoder_ins.encode(TestTexts_lst)
        
        # Inference
        Predictions_arr = self.Classifier_ins.predict(X_test_vec)
        Probabilities_arr = self.Classifier_ins.predict_proba(X_test_vec)

        # Build results dataframe
        Classification_df = pd.DataFrame({
            "text_str": TestTexts_lst,
            "predicted_label": Predictions_arr,
            "confidence_flt": np.max(Probabilities_arr, axis=1)
        })

        FinalResults_dct = {
            "predictions_df": Classification_df,
            "model_str": self.ModelName_str
        }

        # If actual labels provided, calculate metrics
        if ActualLabels_lst is not None:
            Acc_flt = accuracy_score(ActualLabels_lst, Predictions_arr)
            Report_str = classification_report(ActualLabels_lst, Predictions_arr)
            FinalResults_dct["accuracy_flt"] = round(Acc_flt, 4)
            FinalResults_dct["report_str"] = Report_str

        return FinalResults_dct


#test
def run_task_03():
    # 1. Configuration and Paths
    InpRelPath_str = "config/config.yaml"
    InpAbsPath_str = os.path.abspath(InpRelPath_str)

    with open(InpAbsPath_str, 'r') as file:
        Config_dct = yaml.safe_load(file)
    
    # Generic model selection from config
    ModelID_str = Config_dct['models'].get('classification_encoder', 'all-MiniLM-L6-v2')
    
    # 2. Instance setup
    Clf_ins = TextClassifier_ins(ModelID_str)

    # 3. Labeled Training Data (Intent Classification Example)
    TrainData_lst = [
        "I want to buy a new laptop", "How much does this cost?", "Place an order for me", # Sales
        "My app is crashing on startup", "I forgot my password", "System error 404",     # Support
        "Where is your office located?", "What are your working hours?"                # Info
    ]
    TrainLabels_lst = ["Sales", "Sales", "Sales", "Support", "Support", "Support", "Info", "Info"]

    # 4. Training
    TrainMeta_dct = Clf_ins.train(TrainData_lst, TrainLabels_lst)
    print(f"Training complete. Features: {TrainMeta_dct['feature_dim_int']} dimensions.")

    # 5. Testing/Inference
    TestTexts_lst = ["The software won't open", "What is the price?"]
    TrueLabels_lst = ["Support", "Sales"]

    FinalEval_dct = Clf_ins.predict(TestTexts_lst, TrueLabels_lst)

    # 6. Output
    print("\n--- Classification Results ---")
    print(FinalEval_dct['predictions_df'])
    print(f"\nOverall Accuracy: {FinalEval_dct.get('accuracy_flt')}")

if __name__ == "__main__":
    run_task_03()