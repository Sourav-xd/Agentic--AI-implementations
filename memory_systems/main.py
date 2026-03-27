#main.py

from google import genai
from memory_manager import PythonicMemory
import os

# 1. Setup the NEW Gemini Client
# The client automatically looks for an environment variable 'GOOGLE_API_KEY'
# Or you can pass it explicitly: api_key="YOUR_API_KEY"
client = genai.Client(api_key="AIzaSyDFcJoV1OKH4u01O2kFbkkDolNIFQ3C7PI")
MODEL_ID = "gemini-2.5-flash-lite" # The latest, fastest, and cheapest model

memory = PythonicMemory()

def chat_with_memory(user_query):
    # Step 1: Retrieve Short and Long term context from your memory_manager
    st_mem, lt_mem = memory.get_context(user_query)
    
    # Step 2: Build the 'Memory-Augmented' Prompt
    context_str = "--- LONG TERM MEMORY (Archived Facts) ---\n"
    for m in lt_mem:
        context_str += f"Past Topic: {m['user']} -> Result: {m['ai']}\n"
        
    context_str += "\n--- SHORT TERM MEMORY (Recent Chat) ---\n"
    for m in st_mem:
        context_str += f"User: {m['user']}\nAI: {m['ai']}\n"

    full_prompt = f"""
    You are a Memory-Augmented AI. Use the provided context to answer.
    If the answer is in the memory, prioritize that information.
    
    {context_str}
    
    Current User Query: {user_query}
    Response:
    """

    # Step 3: Generate using the new SDK syntax
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=full_prompt
    )
    
    ai_text = response.text
    
    # Step 4: Save to our Python-based memory files
    memory.save_interaction(user_query, ai_text)
    return ai_text

if __name__ == "__main__":
    print(f"--- Using Model: {MODEL_ID} ---")
    print("AI: Hello! I'm your memory-augmented assistant.")
    
    # Test Interaction
    user_in = "My project name is 'Alpha-Omega'."
    print(f"\n[User]: {user_in}")
    print(f"AI: {chat_with_memory(user_in)}")
    
    # Fill short term to force archival
    print("\n...Updating short term memory...")
    for i in range(5):
        chat_with_memory(f"Just saving random data point {i}")
    
    # Test Long Term Retrieval
    query = "What was my project name again?"
    print(f"\n[User]: {query}")
    print(f"AI: {chat_with_memory(query)}")