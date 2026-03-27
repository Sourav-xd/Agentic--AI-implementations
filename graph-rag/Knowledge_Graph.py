from dotenv import load_dotenv
import os
from langchain_experimental.graph_transformers import LLMGraphTransformer  
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph

load_dotenv()

graph = Neo4jGraph(url = os.getenv("NEO4J_URI") , username= os.getenv( "NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"))

api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    google_api_key=api_key
) 

# res = llm.invoke("hello, reply in one any word ")
# print(res)

llm_transformer = LLMGraphTransformer(llm=llm)

text = """
Mirzapur is a violent crime saga set in eastern Uttar Pradesh, revolving around illegal power, guns, politics, and generational control of the carpet and arms trade, dominated initially by the Tripathi crime family where Akhandanand Tripathi, known as Kaleen Bhaiya, rules Mirzapur with iron discipline while maintaining a legitimate carpet business front, grooming his volatile elder son Munna Tripathi as heir despite his impulsive brutality and emotional instability, while his younger son Guddu Tripathi remains sidelined yet morally conflicted; parallelly, brothers Guddu Pandit and Bablu Pandit are drawn into the criminal ecosystem after a wedding massacre orchestrated by Munna, transforming from ordinary youths into key players as Guddu becomes muscle and Bablu the strategist, forming alliances with Kaleen Bhaiya before betraying him, which entangles their family including their father Ramakant Pandit, an honest lawyer, and Guddu’s wife Sweety Gupta, whose death becomes a catalyst for vengeance; political power intersects through JP Yadav, who manipulates criminal factions to maintain electoral dominance, while rival don Sharad Shukla quietly plans revenge for his father’s murder by Kaleen Bhaiya, forming a slow-burn alliance with Madhuri Yadav, whose arc shifts from idealistic daughter to pragmatic Chief Minister willing to sacrifice personal emotions for state power, even as she marries Munna to consolidate authority; meanwhile Beena Tripathi navigates survival through manipulation and secret affairs, ultimately positioning her unborn child as a future power token, while enforcers like Dadda Tyagi and his sons Bharat Tyagi and Shatrughan Tyagi expand the conflict beyond Mirzapur into Purvanchal, turning the narrative into a multi-state crime war where loyalty is transactional, revenge spans generations, and governance, crime, and family bloodlines collapse into a single ecosystem of power.
"""

documents = [Document (page_content=text)]
graph_documents = llm_transformer.convert_to_graph_documents(documents)
print(f"Nodes: {graph_documents[0].nodes}")
print(f"Relationships: {graph_documents[0].relationships}")

graph.add_graph_documents(graph_documents)