"""
query_rag.py
Handles user queries, retrieves context from Pinecone, and generates an answer using Google Gemini.
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone
#import google as genai
from google.genai import Client
from semantic_router.encoders import HuggingFaceEncoder
from data import dataset

# Load environment variables
load_dotenv()

def retrieve_context(Query_str: str, IndexName_str: str, TopK_int: int = 2) -> dict:
    """
    Embeds the query, searches Pinecone, and retrieves contextual chunks (including pre/post chunks).

    Args:
        Query_str (str): The question being asked.
        IndexName_str (str): The name of the Pinecone index.
        TopK_int (int): The number of top semantic matches to retrieve.

    Returns:
        Result_dct (dict): A dictionary containing the stitched context string.
    """
    Result_dct = {"status": "failed", "context_str": ""}
    
    try:
        # 1. Initialize same local embedding model
        Encoder_ins = HuggingFaceEncoder(name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 2. Initialize Pinecone
        PineconeKey_str = os.getenv("PINECONE_API_KEY")
        Pinecone_ins = Pinecone(api_key=PineconeKey_str)
        Index_ins = Pinecone_ins.Index(IndexName_str)
        
        # 3. Embed Query
        QueryVector_lst = Encoder_ins([Query_str])[0]
        
        # 4. Search Vector DB
        Matches_ins = Index_ins.query(
            vector=QueryVector_lst,
            top_k=TopK_int,
            include_metadata=True
        )
        
        ContextChunks_lst = []
        
        # 5. Fetch Surrounding Context (Pre and Post chunks)
        for Match_dct in Matches_ins["matches"]:
            CurrentContent_str = Match_dct["metadata"]["content"]
            PreId_str = Match_dct["metadata"]["prechunk_id"]
            PostId_str = Match_dct["metadata"]["postchunk_id"]
            
            IdsToFetch_lst = [Id_str for Id_str in [PreId_str, PostId_str] if Id_str]
            
            PreChunkText_str = ""
            PostChunkText_str = ""
            
            if IdsToFetch_lst:
                # Fetch missing surrounding chunks directly by their ID
                OtherChunks_dct = Index_ins.fetch(ids=IdsToFetch_lst).vectors
                if PreId_str in OtherChunks_dct:
                    PreChunkText_str = OtherChunks_dct[PreId_str]["metadata"]["content"]
                if PostId_str in OtherChunks_dct:
                    PostChunkText_str = OtherChunks_dct[PostId_str]["metadata"]["content"]
            
            # Stitch the chunks back together
            CombinedContext_str = f"{PreChunkText_str}\n{CurrentContent_str}\n{PostChunkText_str}"
            ContextChunks_lst.append(CombinedContext_str)
            
        Result_dct["context_str"] = "\n\n---\n\n".join(ContextChunks_lst)
        Result_dct["status"] = "success"
        
    except Exception as Error_ins:
        print(f"Error during retrieval: {Error_ins}")
        
    return Result_dct

def generate_answer(Query_str: str, Context_str: str) -> dict:
    """
    Uses Google Gemini (Free tier) to generate an answer based on the retrieved context.

    Args:
        Query_str (str): The user's original question.
        Context_str (str): The contextual text retrieved from the vector database.

    Returns:
        Result_dct (dict): A dictionary containing the AI-generated answer.
    """
    Result_dct = {"status": "failed", "answer_str": ""}
    
    try:
        # Initialize the new genai Client
        GeminiKey_str = os.getenv("GEMINI_API_KEY")
        Client_ins = Client(api_key=GeminiKey_str)
        
        Prompt_str = f"""
        You are a helpful AI assistant. Answer the user's question based strictly on the provided context. 
        If the answer is not in the context, say "I don't have enough information to answer that."

        Context:
        {Context_str}

        Question: {Query_str}
        
        Answer:
        """
        
        # Use the new generate_content syntax
        Response_ins = Client_ins.models.generate_content(
            model="gemini-2.5-flash-lite", 
            contents=Prompt_str
        )
        
        Result_dct["answer_str"] = Response_ins.text
        Result_dct["status"] = "success"
        
    except Exception as Error_ins:
        print(f"Error during generation: {Error_ins}")
        
    return Result_dct

if __name__ == "__main__":
    # Test the Retrieval Augmented Generation pipeline
    UserQuery_str = "What exactly is semantic chunking in rag?"
    TargetIndex_str = "semantic-chunking"
    
    print("Retrieving context from Vector DB...")
    RetrievalResult_dct = retrieve_context(Query_str=UserQuery_str, IndexName_str=TargetIndex_str)
    
    if RetrievalResult_dct["status"] == "success":
        ContextData_str = RetrievalResult_dct["context_str"]
        print(f"Context retrieved successfully. Length: {len(ContextData_str)} characters.")
        print("\n--- WHAT GEMINI IS READING (CONTEXT) ---")
        print(ContextData_str)
        print("----------------------------------------\n")
        print("Generating answer with Gemini...")
        GenerationResult_dct = generate_answer(Query_str=UserQuery_str, Context_str=ContextData_str)
        
        if GenerationResult_dct["status"] == "success":
            print("\n--- FINAL ANSWER ---")
            print(GenerationResult_dct["answer_str"])
            print("--------------------\n")

