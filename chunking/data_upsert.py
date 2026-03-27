"""
ingest_data.py
Handles the reading of a large text, semantic chunking, and upserting into Pinecone Vector DB.
"""

import os
from dotenv import load_dotenv
from data import dataset
from pinecone import Pinecone
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.splitters import RollingWindowSplitter
from semantic_router.schema import DocumentSplit

# Load environment variables
load_dotenv()

def read_text_data(InpAbsPath_str: str) -> dict:
    """
    Reads text data from a given absolute path.

    Args:
        InpAbsPath_str (str): The absolute path to the text file.

    Returns:
        Result_dct (dict): A dictionary containing the status and the string content.
    """
    Result_dct = {"status": "failed", "content_str": ""}
    
    try:
        
        TextData_str = dataset.test
        Result_dct["content_str"] = TextData_str
        Result_dct["status"] = "success"
    except Exception as Error_ins:
        print(f"Error reading file: {Error_ins}")
        
    return Result_dct

def create_metadata(DocSplits_lst: list[DocumentSplit], DocId_str: str) -> dict:
    """
    Creates metadata with pre-chunk and post-chunk context for each split.

    Args:
        DocSplits_lst (list): A list of DocumentSplit objects from the semantic router.
        DocId_str (str): A unique identifier for the document source.

    Returns:
        Result_dct (dict): A dictionary containing a list of formatted metadata dictionaries.
    """
    Result_dct = {"status": "success", "metadata_lst": []}
    Metadata_lst = []
    
    for Index_int, Split_ins in enumerate(DocSplits_lst):
        PrechunkId_str = "" if Index_int == 0 else f"{DocId_str}#{Index_int - 1}"
        PostchunkId_str = "" if Index_int + 1 == len(DocSplits_lst) else f"{DocId_str}#{Index_int + 1}"
        
        Metadata_lst.append({
            "id": f"{DocId_str}#{Index_int}",
            "content": Split_ins.content,
            "prechunk_id": PrechunkId_str,
            "postchunk_id": PostchunkId_str,
            "doc_id": DocId_str
        })
        
    Result_dct["metadata_lst"] = Metadata_lst
    return Result_dct

def process_and_upsert(Text_str: str, IndexName_str: str) -> dict:
    """
    Processes the raw string into semantic chunks, embeds them, and upserts to Pinecone.

    Args:
        Text_str (str): The large raw text string to be chunked.
        IndexName_str (str): The name of the Pinecone index.

    Returns:
        Result_dct (dict): A dictionary containing operation status and vector count.
    """
    Result_dct = {"status": "failed", "upserted_count_int": 0}
    
    # 1. Initialize local embedding model (Free)
    print("Loading local embedding model...")
    Encoder_ins = HuggingFaceEncoder(name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 2. Initialize Semantic Splitter
    Splitter_ins = RollingWindowSplitter(
        encoder=Encoder_ins,
        dynamic_threshold=True,
        min_split_tokens=50,
        max_split_tokens=300,
        window_size=2
    )
    
    # 3. Create Splits
    print("Performing semantic chunking...")
    Splits_lst = Splitter_ins([Text_str])
    
    # 4. Generate Metadata
    DocId_str = "doc_001"
    MetadataResult_dct = create_metadata(DocSplits_lst=Splits_lst, DocId_str=DocId_str)
    Metadata_lst = MetadataResult_dct["metadata_lst"]
    
    # 5. Initialize Pinecone
    PineconeKey_str = os.getenv("PINECONE_API_KEY")
    Pinecone_ins = Pinecone(api_key=PineconeKey_str)
    Index_ins = Pinecone_ins.Index(IndexName_str)
    
    # 6. Embed and Upsert
    BatchSize_int = 100
    TotalUpserted_int = 0
    
    print("Embedding and uploading to Pinecone...")
    for i_int in range(0, len(Metadata_lst), BatchSize_int):
        End_int = min(len(Metadata_lst), i_int + BatchSize_int)
        Batch_lst = Metadata_lst[i_int:End_int]
        
        Ids_lst = [Item_dct["id"] for Item_dct in Batch_lst]
        Contents_lst = [Item_dct["content"] for Item_dct in Batch_lst]
        
        # Generate embeddings
        Embeds_lst = Encoder_ins(Contents_lst)
        
        # Upsert
        Index_ins.upsert(vectors=zip(Ids_lst, Embeds_lst, Batch_lst))
        TotalUpserted_int += len(Batch_lst)
        
    Result_dct["status"] = "success"
    Result_dct["upserted_count_int"] = TotalUpserted_int
    return Result_dct

if __name__ == "__main__":
    # Define paths without hardcoding
    CurrentDir_str = os.path.dirname(os.path.abspath(__file__))
    InpRelPath_str = "source_data.txt"
    InpAbsPath_str = os.path.join(CurrentDir_str, InpRelPath_str)
    
    # Create a dummy file if it doesn't exist for testing
    if not os.path.exists(InpAbsPath_str):
        with open(InpAbsPath_str, "w", encoding="utf-8") as f:
            f.write("Artificial intelligence is a fascinating field. " * 50)
            f.write("Machine learning relies heavily on data. " * 50)
            f.write("Vector databases store embeddings for quick retrieval. " * 50)
            
    # Execute Pipeline
    ReadResult_dct = read_text_data(InpAbsPath_str=InpAbsPath_str)
    
    if ReadResult_dct["status"] == "success":
        TextContent_str = ReadResult_dct["content_str"]
        TargetIndex_str = "semantic-chunking"
        
        FinalResult_dct = process_and_upsert(Text_str=TextContent_str, IndexName_str=TargetIndex_str)
        print(f"Pipeline finished. Status: {FinalResult_dct['status']}, Vectors upserted: {FinalResult_dct['upserted_count_int']}")