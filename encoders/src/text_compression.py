import os
import yaml
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

class TextCompressor_ins:
    """
    Class to compress text meaning into fixed-size, low-dimensional vectors.
    
    Attributes:
        ModelName_str (str): The embedding model for initial encoding.
        Encoder_ins (SentenceTransformer): The model instance.
        Reducer_ins (PCA): The dimensionality reduction instance.
    """

    def __init__(self, ModelName_str: str, TargetDim_int: int = 2):
        """
        Initializes the compression engine.
        
        Args:
            ModelName_str (str): Model name for vectorization.
            TargetDim_int (int): The final compressed size of the vector.
        """
        self.ModelName_str = ModelName_str
        self.TargetDim_int = TargetDim_int
        self.Encoder_ins = SentenceTransformer(ModelName_str)
        self.Reducer_ins = PCA(n_components=TargetDim_int)

    def compress_corpus(self, Documents_lst: List[str]) -> Dict[str, Any]:
        """
        Encodes and then reduces the dimensionality of text data.
        
        Input:
            Documents_lst (List[str]): List of texts to compress.
            
        Output:
            Results_dct (Dict): Compressed vectors and compression ratio analysis.
        """
        # 1. Primary Compression (Text to Dense Vector)
        FullEmbeds_vec = self.Encoder_ins.encode(Documents_lst)
        OriginalDim_int = FullEmbeds_vec.shape[1]

        # 2. Secondary Compression (Dense Vector to Low-Dim Vector)
        # PCA requires at least as many samples as the target dimension
        CompressedEmbeds_vec = self.Reducer_ins.fit_transform(FullEmbeds_vec)
        
        # 3. Calculate Variance Retained (How much 'meaning' did we keep?)
        VarianceRetained_flt = float(np.sum(self.Reducer_ins.explained_variance_ratio_))

        # 4. Data Packaging
        Compressed_df = pd.DataFrame(
            CompressedEmbeds_vec, 
            columns=[f"dim_{i}_flt" for i in range(self.TargetDim_int)]
        )
        Compressed_df['original_text'] = Documents_lst

        Results_dct = {
            "compressed_vectors_df": Compressed_df,
            "original_dim_int": OriginalDim_int,
            "compressed_dim_int": self.TargetDim_int,
            "variance_retained_flt": round(VarianceRetained_flt, 4),
            "compression_ratio_str": f"{OriginalDim_int} -> {self.TargetDim_int}"
        }

        return Results_dct

def run_task_10():
    # 1. Config/Path Management
    InpRelPath_str = "config/config.yaml"
    InpAbsPath_str = os.path.abspath(InpRelPath_str)

    with open(InpAbsPath_str, 'r') as file:
        Config_dct = yaml.safe_load(file)
    
    ModelID_str = Config_dct['models'].get('compression_encoder', 'all-MiniLM-L6-v2')
    
    # 2. Instance setup (Compressing to 3 dimensions)
    Compressor_ins = TextCompressor_ins(ModelID_str, TargetDim_int=3)

    # 3. Sample Data (Multiple documents)
    Corpus_lst = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast dark-colored canine leaps above a sleepy hound.",
        "Data science involves statistics and programming.",
        "Machine learning is a subset of artificial intelligence.",
        "Quantum computing uses qubits for processing.",
        "The weather in London is often rainy and grey."
    ]

    # 4. Execute Compression
    FinalResults_dct = Compressor_ins.compress_corpus(Corpus_lst)

    # 5. Output display
    print(f"--- Text Compression Task (Ratio: {FinalResults_dct['compression_ratio_str']}) ---")
    print(f"Semantic Variance Retained: {FinalResults_dct['variance_retained_flt'] * 100}%")
    print("\nCompressed Vector Samples:")
    print(FinalResults_dct['compressed_vectors_df'].head())

    #print("Original Dim:", FinalResults_dct['original_dim_int'])
    #print("Compressed Dim:", FinalResults_dct['compressed_dim_int'])


if __name__ == "__main__":
    run_task_10()