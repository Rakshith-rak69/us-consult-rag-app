import streamlit as st
import os
import sys
import sqlite3




import re
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

# === Paths & Setup ===

CHROMA_DB_PATH = "./us_law_chroma_db/us_law_chroma_db"
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# Connect to ChromaDB persistent client
@st.cache_resource
def get_chroma_collection(path):
    """Initializes and returns the ChromaDB collection."""
    chroma_client = chromadb.PersistentClient(path)
    collection = chroma_client.get_or_create_collection(
        name="law_consult",
        metadata={"hnsw:space": "cosine"} 
    )
    st.success("ChromaDB loaded successfully!")
    return collection

# Call the cached function to get your collection
collection = get_chroma_collection(CHROMA_DB_PATH)
# Text splitter (if needed)
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", "? ", "! "],
    chunk_size=1000,
    chunk_overlap=250
)

# OpenAI / OpenRouter client (update your API key here)
import streamlit as st
from openai import OpenAI
import toml

# Use Streamlit secret if available, else fallback to local secrets.toml
if "OPENROUTER_API_KEY" in st.secrets:
    api_key = st.secrets["OPENROUTER_API_KEY"]
else:
    secrets = toml.load("secrets.toml")
    api_key = secrets["openrouter"]["api_key"]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

if "OPENROUTER_API_KEY" in st.secrets:
    st.success("✅ API_KEY found in Streamlit secrets!")
else:
    st.error("❌ API_KEY NOT found in Streamlit secrets! Please check your Streamlit Cloud secrets configuration.")


system_prompt = ("You're a helpful assistant who looks answers up for a user in a textbook and returns the "
                 "answer to the user's question. If the answer is not in the textbook, you say "
                 "'I'm sorry, I don't have access to that information.'")

# --- Utility functions ---

def completion_chat(user_prompt, system_prompt):
    completion = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return completion.choices[0].message.content

def extract_title(full_text):
    metadata_match = re.search(r"--- METADATA ---\s*(.*?)\s*-{5,}", full_text, re.DOTALL)
    if not metadata_match:
        return None
    metadata_block = metadata_match.group(1)
    title_match = re.search(r"Title:\s*(.*)", metadata_block)
    if title_match:
        return title_match.group(1).strip()
    return None

def make_rag_prompt(query, result_str):
    return f"""
Your task is to answer the following user question using the supplied search results. At the end of each search result will be Metadata. Cite the passages, title, their chunk index, and their URL in your answer.
User Question: {query}
Search Results: {result_str}

Your answer:
"""

def populate_rag_query(query, n_results=1):
    search_results = collection.query(query_texts=[query], n_results=n_results)
    result_str = ""
    for idx, result in enumerate(search_results["documents"][0]):
        metadata = search_results["metadatas"][0][idx]
        formatted_result = f"""<SEARCH RESULT>
<DOCUMENT>{result}</DOCUMENT>
<METADATA>
<TITLE>{metadata.get('title', 'N/A')}</TITLE>
<URL>{metadata.get('url', 'N/A')}</URL>
</METADATA>
</SEARCH RESULT>"""
        result_str += formatted_result
    return result_str

def get_RAG_completion(query, n_results=1):
    results = collection.query(query_texts=[query], n_results=n_results)
    result_str = ""
    for result in results['documents'][0]:
        result_str += result
    formatted_query = make_rag_prompt(query, result_str)
    return formatted_query

#DECOUPLING

def get_previous_and_next_chunks(chunk_idx):
    previous_idx = collection.get(where={'chunk_idx': {"$eq": chunk_idx - 1}})
    next_idx = collection.get(where={'chunk_idx': {"$eq": chunk_idx + 1}})
    return previous_idx, next_idx

def expanded_search(original_chunk):
    original_metadata = original_chunk['metadatas'][0][0]
    original_document = original_chunk['documents'][0][0]

    result_str = f"""<SEARCH RESULT>
<DOCUMENT>{original_document}</DOCUMENT>
<METADATA>
<TITLE>{original_metadata.get('title', 'N/A')}</TITLE>
<URL>{original_metadata.get('url', 'N/A')}</URL>
</METADATA>
</SEARCH RESULT>"""

    previous_idx, next_idx = get_previous_and_next_chunks(original_metadata['chunk_idx'])

    for chunk in [previous_idx, next_idx]:
        if chunk and len(chunk['metadatas']) > 0:
            metadata = chunk['metadatas'][0]
            formatted_str = f"""<SEARCH RESULT>
<DOCUMENT>{chunk["documents"][0]}</DOCUMENT>
<METADATA>
<TITLE>{metadata['title']}</TITLE>
<URL>{metadata['url']}</URL>
</METADATA>
</SEARCH RESULT>"""
            result_str += formatted_str
    return result_str

def make_decoupled_prompt(query, n_results=1):
    search_results = collection.query(query_texts=[query], n_results=n_results)
    total_result_str = expanded_search(search_results)
    rag_prompt = make_rag_prompt(query, total_result_str)
    complete_result = completion_chat(rag_prompt, system_prompt)
    return complete_result


# CQR
chat_memory = []

def rewrite_query(query, chat_history):
    prompt = f"""<INSTRUCTIONS>
Given the following chat history and the user's latest query, rewrite the query to include relevant context.
</INSTRUCTIONS>

<CHAT_HISTORY>
{chat_history}
</CHAT_HISTORY>

<LATEST_QUERY>
{query}
</LATEST_QUERY>

Your rewritten query:"""
    return completion_chat(prompt, system_prompt)

def perform_cqr_rag(query, chat_history, n_results=2):
    refined_query = rewrite_query(query, chat_history)
    result_str = populate_rag_query(refined_query, n_results=n_results)
    rag_prompt = make_rag_prompt(refined_query, result_str)
    rag_completion = completion_chat(rag_prompt, system_prompt)
    chat_memory.append({"role": "user", "content": query})
    chat_memory.append({"role": "assistant", "content": rag_completion})
    return rag_completion

#HyDE
def make_hyde_prompt(query):
    return f"""<INSTRUCTIONS>
Try to answer the query.
If you don’t know the answer, try to sound like you do.
Use language you think will be in the actual answer.
Make the answer roughly a paragraph long.
</INSTRUCTIONS>

<QUERY>{query}</QUERY>
"""

def answer_with_hyde(query, n_results=1):
    hyde_prompt = make_hyde_prompt(query)
    hyde_query = completion_chat(hyde_prompt, system_prompt)
    result_str = populate_rag_query(hyde_query, n_results=n_results)
    rag_prompt = make_rag_prompt(query, result_str)
    rag_completion = completion_chat(rag_prompt, system_prompt)
    return rag_completion

#FUSION SEARCH
import json
def generate_subquestions(query):
    prompt = f"""<INSTRUCTIONS>
Given the following user query, generate a list of 2-4 subquestions that would help in answering the original query.
Return the result as JSON with the key "data" and the list as its value.
Only output valid JSON and no other characters.
Do not output markdown backticks. Just output raw JSON only.
</INSTRUCTIONS>

<QUERY>{query}</QUERY>
"""

    response = completion_chat(prompt, system_prompt)
#     if the model generates invalid json, uncomment the print line and inspect what went wrong
#     print(response)
    ## YOUR SOLUTION HERE ##
    subquery_dict = json.loads(response)
    return subquery_dict['data']



def get_and_concat(query, n_results=1):
  subquestions = generate_subquestions(query)
  subquestion_context = []
  for subquestion in subquestions:
    result_str = populate_rag_query(subquestion, n_results=n_results)
    rag_prompt = make_rag_prompt(subquestion, result_str)
    answer = completion_chat(rag_prompt, system_prompt)
    subquestion_context.append(f"Q: {subquestion}\n\nA: {answer}")
  return "\n\n".join(subquestion_context)


def fusion_search(query, n_results=1):
  subquestion_context = get_and_concat(query, n_results=n_results)
  st.subheader("Subquestion Context:")
  st.markdown(subquestion_context)
  final_prompt = f"""<INSTRUCTIONS>
    **You are an expert at synthesizing information.**
    **Given the answers to several subquestions provided in <SUBQUESTION_INFO> and the <ORIGINAL_QUERY>,**
    **synthesize a comprehensive and concise answer to the <ORIGINAL_QUERY> by combining the relevant information from the subquestion answers.**
    **Ensure your final answer directly addresses the original query using *only* the information provided in the <SUBQUESTION_INFO>.**
    **Do not just list the subquestion answers; integrate them into a coherent response.**
    **If the provided <SUBQUESTION_INFO> is insufficient to answer the <ORIGINAL_QUERY>, state that you cannot provide an answer based on the given information.**
    </INSTRUCTIONS>

    <SUBQUESTION_INFO>
    {subquestion_context}
    </SUBQUESTION_INFO>

    <ORIGINAL_QUERY>{query}</ORIGINAL_QUERY>

    Final Answer:"""
  final_answer = completion_chat(final_prompt, system_prompt)
  return final_answer

#Default

def perform_default_rag(query, n_results=2):
    result_str = populate_rag_query(query, n_results=n_results)
    rag_prompt = make_rag_prompt(query, result_str)
    rag_completion = completion_chat(rag_prompt, system_prompt)
    return rag_completion

# =========== Streamlit UI =============

st.title("⚖️ US Law Consult RAG System")
st.markdown("Your AI-powered legal research assistant.")
st.markdown("### 🧠 Try asking questions like:")
st.markdown("1. **What are Improvements of Disaster Response and Loans?**")
st.markdown("2. **According to the National Defense Authorization Act for Fiscal Year 2012, what is its short title, what are the five main divisions into which it is organized, and what is its stated primary purpose?**")
st.markdown("3. **Explain the National Highway Program and its importance in U.S. infrastructure development.**")

user_question = st.text_area("Ask a question", key="user_question")

with st.sidebar:
    st.title("⚙️ RAG Settings")
    st.markdown("Adjust the configurations for your legal queries.")

    st.subheader("Language Model")
    selected_model = st.selectbox(
        "Choose the Large Language Model (LLM):",
        ("gpt-4o-mini", "gpt-3.5-turbo"),
        key='llm_selector'
    )
    st.info(f"Selected LLM: **{selected_model}**")

    st.subheader("RAG Technique")
    selected_technique = st.radio(
        "Select a RAG approach:",
        ("HyDE", "CQR", "Decoupled", "Fusion Search", "Default"),
        key='rag_technique_selector'
    )
    st.info(f"Selected Technique: **{selected_technique}**")

    st.subheader("Search Parameters")
    n_results = st.number_input(
        "Number of Results to Retrieve:",
        min_value=1,
        max_value=10,
        value=2,
        key='num_results_input'
    )
    st.info(f"Retrieving: {n_results} documents.")

    st.markdown("---")
    st.markdown("Developed by Rakshith")

# --- Handle button click to get answers ---

if st.button("Get Answers"):
    if not user_question.strip():
        st.error("Please enter a question.")
    else:
        st.write(f"**Question:** {user_question}")
        st.write(f"**Using model:** {selected_model}")
        st.write(f"**Using RAG technique:** {selected_technique}")

        # Process the selected RAG technique
        if selected_technique == "HyDE":
            response = answer_with_hyde(user_question, n_results)
        elif selected_technique == "CQR":
            chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_memory])
            response = perform_cqr_rag(user_question, chat_history_str, n_results)
        elif selected_technique == "Decoupled":
            response = make_decoupled_prompt(user_question, n_results)
        elif selected_technique == "Fusion Search":
            response = fusion_search(user_question)  
        else:  # Default RAG
            response = perform_default_rag(user_question, n_results)

        st.success("✅ Answer generated!")
        st.markdown("### 📄 Response:")
        st.markdown(response)

