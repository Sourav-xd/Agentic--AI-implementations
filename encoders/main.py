import os
import yaml
import importlib
from typing import Dict, Any

class EmbeddingPipelineWrapper_ins:
    """
    A unified wrapper class to configure and execute any of the 10 embedding tasks.
    
    Attributes:
        ConfigAbsPath_str (str): Absolute path to the configuration file.
        Config_dct (Dict): Loaded configuration dictionary.
    """

    def __init__(self, ConfigRelPath_str: str = "config/config.yaml"):
        """
        Initializes the wrapper and loads the global configuration.
        
        Args:
            ConfigRelPath_str (str): Relative path to the config file.
        """
        self.ConfigAbsPath_str = os.path.abspath(ConfigRelPath_str)
        
        if not os.path.exists(self.ConfigAbsPath_str):
            raise FileNotFoundError(f"Config file not found at {self.ConfigAbsPath_str}")
            
        with open(self.ConfigAbsPath_str, 'r') as file:
            self.Config_dct = yaml.safe_load(file)

    def execute_task(self, TaskID_str: str, Payload_dct: Dict[str, Any]) -> Dict[str, Any]:
        """
        Routes the execution to the appropriate task module based on TaskID.
        
        Input:
            TaskID_str (str): The identifier of the task (e.g., 'task_01_search').
            Payload_dct (Dict): The input data required for the specific task.
            
        Output:
            FinalResults_dct (Dict): Standardized dictionary containing task outputs.
        """
        # Fetch the model defined in the config for this specific task
        ModelName_str = self.Config_dct['models'].get(TaskID_str)
        if not ModelName_str:
            raise ValueError(f"No model configured for task ID: {TaskID_str}")

        print(f"Initializing {TaskID_str} using model: {ModelName_str}...")

        # --- Task Routing Logic ---
        # Note: In a production environment, you can use importlib to dynamically load 
        # these to save memory, e.g., module = importlib.import_module(f"src.{TaskID_str}")
        
        if TaskID_str == "task_01_search":
            from src.semantic_search import SemanticSearch_ins
            Task_ins = SemanticSearch_ins(ModelName_str)
            return Task_ins.perform_search(
                Query_str=Payload_dct["query_str"], 
                Documents_lst=Payload_dct["documents_lst"]
            )

        elif TaskID_str == "task_04_clustering":
            from src.task_04_clustering import DocumentClustering_ins
            K_int = self.Config_dct['task_settings'].get('clustering_k_int', 3)
            Task_ins = DocumentClustering_ins(ModelName_str, NumClusters_int=K_int)
            return Task_ins.group_documents(Documents_lst=Payload_dct["documents_lst"])

        elif TaskID_str == "task_08_zeroshot":
            from src.task_08_zeroshot import ZeroShotClassifier_ins
            Task_ins = ZeroShotClassifier_ins(ModelName_str)
            return Task_ins.classify(
                InputText_str=Payload_dct["input_text_str"], 
                CandidateLabels_lst=Payload_dct["candidate_labels_lst"]
            )

        # rest of the tasks

        else:
            raise NotImplementedError(f"Task logic for {TaskID_str} is not yet integrated into the wrapper.")


def main():
    # 1. Path definitions
    ConfigRelPath_str = "config/config.yaml"
    
    # 2. Instantiate the Unified Wrapper
    Pipeline_ins = EmbeddingPipelineWrapper_ins(ConfigRelPath_str)
    
    # 3. Formulate payloads for different tasks
    SearchPayload_dct = {
        "query_str": "refund eligibility rules",
        "documents_lst": [
            "The return policy details how to get your money back.",
            "Our office is located in downtown New York."
        ]
    }
    
    ZeroShotPayload_dct = {
        "input_text_str": "The checkout button is overlapping with the sidebar.",
        "candidate_labels_lst": ["UI Bug", "Billing Issue", "Feature Request"]
    }

    # 4. Execute Tasks dynamically
    try:
        print("\n--- Running Task 1 ---")
        SearchRes_dct = Pipeline_ins.execute_task("task_01_search", SearchPayload_dct)
        print(SearchRes_dct['matches'])

        # print("\n--- Running Task 8 ---")
        # ZeroShotRes_dct = Pipeline_ins.execute_task("task_08_zeroshot", ZeroShotPayload_dct)
        # print(ZeroShotRes_dct['all_predictions_df'])
        
    except Exception as e:
        print(f"Pipeline Execution Failed: {str(e)}")

if __name__ == "__main__":
    main()