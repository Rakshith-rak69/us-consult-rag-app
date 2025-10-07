# **⚖️ US Law Consult RAG System**

This project implements an AI-powered legal research assistant leveraging various Retrieval Augmented Generation (RAG) techniques to answer user questions based on a corpus of US legislative text. The system is built using Streamlit for the interactive user interface and ChromaDB for vector storage.

## **Capstone Overview**

### **💡 Project Goal & Overview**

**Goal:** To develop an intelligent legal research assistant that can accurately answer questions about US law by retrieving relevant information from a predefined legislative text corpus and then generating concise, informed responses.

**Dataset:**

* **Corpus:** A collection of US legislative text sourced from https://www.govinfo.gov/bulkdata/COMPS/. The system uses a subset of this data due to the significant time required for initial document splitting and adding to the ChromaDB collection. The local directory my\_chroma\_db\_folder is where this processed portion of the corpus is stored.  
* **Vector Database:** ChromaDB is used for persistent storage and efficient retrieval of text chunks.

**Models:**

* **Language Models:** The system utilizes openai/gpt-4o-mini (and offers gpt-3.5-turbo as an alternative) via the OpenRouter API for generating responses. These models are crucial for understanding user queries, re-writing them for better retrieval, and synthesizing final answers.

### **🧠 RAG Techniques Implemented**

The system explores and compares several advanced RAG techniques to enhance the quality and relevance of the generated answers:

* **Default RAG:**  
  * **Description:** A standard RAG approach where the user's query is directly used to retrieve a specified number of relevant documents. The retrieved documents are then passed to the LLM to formulate an answer.  
* **HyDE (Hypothetical Document Embedding):**  
  * **Description:** Instead of directly searching with the user's query, HyDE first generates a hypothetical answer to the user's question using the LLM. This hypothetical answer is then used as a query for document retrieval, often leading to more semantically relevant results. The actual user query is then used with the retrieved documents to generate the final answer.  
* **CQR (Conversational Query Rewriting):**  
  * **Description:** Designed for multi-turn conversations, CQR rewrites the user's latest query by incorporating the context from the chat history. This ensures that follow-up questions are understood within the broader conversational flow, leading to more coherent and relevant answers over time.  
* **Decoupled RAG:**  
  * **Description:** This technique aims to provide more context by not only retrieving the most relevant chunk but also fetching its immediate preceding and succeeding chunks from the document. This helps in maintaining document coherence and provides richer context to the LLM for answer generation.  
* **Fusion Search:**  
  * **Description:** For complex queries, Fusion Search breaks down the original user question into 2-4 smaller, more focused sub-questions. Each sub-question is then used to perform individual RAG queries. Finally, the answers to these sub-questions are synthesized by the LLM to provide a comprehensive response to the original complex query.
  * **Note that Fusion Search might take extra time compared to other methods due to the overhead of generating multiple sub-questions and performing separate RAG queries for each.**

### **🧪 Results & Performance (Conceptual based on RAG goals)**

The effectiveness of each RAG technique is observed through the quality of the generated answers, specifically in terms of relevance, coherence, and accuracy, especially when dealing with complex or conversational queries. While explicit quantitative metrics are not detailed in the provided finalrag.py, the comparison between techniques typically focuses on:

* **Relevance:** How well the retrieved documents match the user's intent.  
* **Completeness:** Whether the answers fully address the user's question.  
* **Coherence:** The flow and readability of the generated response.  
* **Contextual Understanding:** For CQR, how well the system maintains context across turns.  
* **Robustness:** How well the system handles nuanced or ambiguous queries (e.g., with HyDE and Fusion Search).

### **Getting Started**

To explore and run the code for this RAG system, follow these steps:

#### **Install Dependencies**

Ensure you have Python installed (preferably Python 3.9+). All required libraries and their versions can be installed.

1. **Create requirements.txt:** In the same directory as your finalrag.py file, create a file named requirements.txt and add the following content:
   ```bash
   streamlit\>=1.30.0  
   chromadb\>=0.4.0  
   langchain==0.0.196  
   openai\>=1.0.0  
   protobuf\>=4.21.6,\<5.0.0  
   grpcio\>=1.51.1,\<2.0.0  
   pysqlite3-binary==0.5.0
   ```

2. **Install:** Run the following command in your terminal:

   ```bash
   pip install \-r requirements.txt
   ```

#### **Set up OpenAI API Key**

This project uses the OpenAI API via OpenRouter. You need to set your OPENROUTER\_API\_KEY as a Streamlit secret.

1. **Obtain API Key:** Get your API key from OpenRouter.  
2. **Configure Streamlit Secrets:**  
   * If running locally, create a new folder named .streamlit in your project directory (if it doesn't already exist). Inside .streamlit, create a file named secrets.toml. Add your API key to this file:  
     OPENROUTER\_API\_KEY="your\_openrouter\_api\_key\_here"

     **Important:** Replace "your\_openrouter\_api\_key\_here" with your actual API key and keep the quotation marks.  
   * If deploying to Streamlit Cloud, go to your app's dashboard, click "..." \> "Settings" \> "Secrets", and add your key there.

#### **Run the Application**

Navigate to the project directory in your terminal and run the Streamlit application:

streamlit run finalrag.py

This will open the application in your web browser, where you can interact with the RAG system, ask legal questions, and experiment with different RAG techniques.

Developed by Rakshith