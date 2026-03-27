from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

# 1. Load environment variables
load_dotenv()

# 2. Connect to the existing Neo4j Graph
# This will automatically fetch the schema (nodes and relationships) you created in the previous script
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)
# 3. Initialize the LLM
# Note: Ensure you use a model version available to your API key
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)
# 4. Create the Graph RAG Chain
# This chain translates natural language -> Cypher query -> Database Result -> Natural Language Answer
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,  # Set to True to see the generated Cypher query in the console
    allow_dangerous_requests=True # Required to allow the LLM to execute queries
)

# 5. Ask a question
question = "Who is Marie Curie's husband and what distinction do they share?"

print(f"--- Asking: {question} ---")
response = chain.invoke({"query": question})

print("\n--- Final Answer ---")
print(response["result"])


