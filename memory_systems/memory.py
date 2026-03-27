import time
from collections import deque
from typing import List, Dict, Any

# --- 1. The Memory Module ---
class MemoryModule:
    def __init__(self):
        # Short-Term: Deque acts like a sliding window (Redis simulation)
        # Keeps only the last 5 interactions for immediate context.
        self.short_term_memory = deque(maxlen=5)
        
        # Long-Term: A list of dictionaries (Vector DB simulation)
        # Stores persistent facts/preferences with metadata.
        self.long_term_memory = []

    def update_short_term(self, role: str, content: str):
        """Log immediate interaction (User or AI)."""
        self.short_term_memory.append({
            "timestamp": time.time(),
            "role": role,
            "content": content
        })

    def save_to_long_term(self, key_fact: str, tags: List[str]):
        """Save important insights for the future."""
        # In a real app, you would embed 'key_fact' into a vector here.
        self.long_term_memory.append({
            "fact": key_fact,
            "tags": tags,
            "created_at": time.time()
        })
        print(f"💾 [Long-Term Memory Updated]: {key_fact}")

    def retrieve_context(self, query: str) -> Dict[str, Any]:
        """The 'Dual Retrieval' Logic."""
        
        # 1. Retrieve Short-Term (Recent Chat History)
        recent_chat = list(self.short_term_memory)
        
        # 2. Retrieve Long-Term (Simulating Semantic Search)
        # We look for keywords in the query that match stored tags/facts.
        relevant_facts = []
        for entry in self.long_term_memory:
            # Simple keyword matching logic (Simulating Vector Similarity)
            if any(tag in query.lower() for tag in entry['tags']):    #faiss, pinecone, weaviate
                relevant_facts.append(entry['fact'])
        
        return {
            "short_term_context": recent_chat,
            "long_term_context": relevant_facts
        }

# --- 2. The RAG System (Simulated) ---
class MemoryAugmentedRAG:
    def __init__(self):
        self.memory = MemoryModule()
        # Simulating Static Docs (The "Retrieval" in RAG)
        self.static_docs = {
            "api": "The API endpoint for login is /v1/auth.",
            "deploy": "Deployment requires the --prod flag."
        }

    def generate_response(self, user_query: str):
        print(f"\n--- Processing Query: '{user_query}' ---")
        
        # Step A: Check Static Docs (Standard RAG)
        static_knowledge = "No static docs found."
        for key, text in self.static_docs.items():
            if key in user_query.lower():
                static_knowledge = text

        # Step B: Check Memory (Memory-Augmented)
        memory_context = self.memory.retrieve_context(user_query)
        
        # Step C: Reasoning (Fusion)
        # We combine Static Docs + Short Term + Long Term
        
        # Check if we know the user's preference from long-term memory
        user_preference = "Standard Mode"
        if "concise" in str(memory_context['long_term_context']).lower():
            user_preference = "Concise Mode (User hates fluff)"
            
        print(f"🔍 [Reasoning]: Combining Contexts...")
        print(f"   -> Static Knowledge: {static_knowledge}")
        print(f"   -> User Preference (Long-Term): {user_preference}")
        print(f"   -> Recent History (Short-Term): {len(memory_context['short_term_context'])} items")

        # Step D: Generation (Simulated Output)
        response = f"Here is the answer based on {user_preference}: {static_knowledge}"
        
        # Step E: Update Memory Loop
        self.memory.update_short_term("user", user_query)
        self.memory.update_short_term("ai", response)
        
        return response

# --- 3. Running the Scenario ---

# Initialize System
bot = MemoryAugmentedRAG()

# Scenario 1: User sets a preference (Long Term Memory Event)
# The system detects a preference and saves it to Long Term storage.
bot.memory.save_to_long_term("User prefers concise, code-only answers.", tags=["style", "api", "deploy"])

# Scenario 2: User asks a question
# The system uses Short Term (to know who asked) and Long Term (to know HOW to answer).
response = bot.generate_response("How do I deploy?")
print(f"🤖 AI Response: {response}")

# Scenario 3: Contextual Continuity
# If we ask another question, Short Term memory now contains the previous Q&A.
response_2 = bot.generate_response("What about the api?")
print(f"🤖 AI Response: {response_2}")