
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader, TextLoader, Docx2txtLoader,YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
from llama_index.core import VectorStoreIndex
from langchain.docstore.document import Document

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
from langchain import embeddings
import os
import streamlit as st


load_dotenv()

template = """
    You are an AI assistant that provides accurate answers based strictly on the given context. Do not generate or include any information not explicitly stated in the context.

If the context contains the answer to a user's question, provide it directly and accurately.
If the context does not contain the answer, state: "The provided context does not contain the answer to this question." You may then offer a generic answer, but must explicitly state that it is not based on the context.
Always communicate clearly and professionally, using bullet points or numbered lists for complex information when appropriate.
Context: {context}

Breakdown of the Prompt
Core Instruction
The AI is tasked with giving accurate, context-only answers.
Explicitly forbids hallucination by stating: "Do not generate or include any information not explicitly stated in the context."
Scenario 1: Answer in Context
When the context has the answer, the AI must provide it directly and accurately, ensuring reliability.
Scenario 2: Answer Not in Context
When the context lacks the answer, the AI must:
State: "The provided context does not contain the answer to this question."
Optionally provide a generic answer, but clearly note it’s not context-based (e.g., "This is a generic answer and not based on the provided context.").
Communication Style
Ensures clarity and professionalism, with a suggestion to use lists for complex info, keeping responses structured and readable.
    
    """

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

if not HUGGINGFACEHUB_API_TOKEN or not GOOGLE_API_KEY or not FIRECRAWL_API_KEY:
    st.error("API keys not found. Please set them in your .env file.")
    st.stop()

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

st.title("MultRAG")

st.sidebar.title("Collection Configuration ⚙️🛠️")
selected_model = st.sidebar.selectbox("Select Model", ["Gemini-2.0-flash", "Mistral-7B-Instruct-v0.3"])
selected_vector_store = st.sidebar.selectbox("Select Vector Store", ["FAISS", "ChromaDB"])
resource_type = st.sidebar.selectbox("Select Resource Type", ["URL", "PDF", "DOC", "TXT","YOUTUBE"])

uploaded_file, url = None, None

if resource_type in ["PDF", "DOC", "TXT"]:
    uploaded_file = st.sidebar.file_uploader(f"Upload a {resource_type} file", type=["pdf", "doc", "docx", "txt"])
    if uploaded_file:
        with st.spinner("Processing file..."):
            try:
                if resource_type == "PDF":
                    with open("temp.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    loader = PyPDFLoader("temp.pdf")
                elif resource_type == "DOC":
                    with open("temp.docx", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    loader = Docx2txtLoader("temp.docx")
                elif resource_type == "TXT":
                    with open("temp.txt", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    loader = TextLoader("temp.txt")
                pages = loader.load_and_split()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                pages_chunks = text_splitter.split_documents(pages)
                emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                if selected_vector_store == "FAISS":
                    vector_store = FAISS.from_documents(pages_chunks, emb_model)
                else:
                    persist_directory = 'db'
                    vector_store = Chroma.from_documents(documents=pages_chunks,embedding=emb_model,persist_directory=persist_directory)
                retriever = vector_store.as_retriever()
                st.success(f"File processed and stored in {selected_vector_store}")
                
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.stop()
                
elif resource_type == "URL":
    url = st.sidebar.text_input("Enter URL", placeholder="https://indeed.com")
    if url and st.sidebar.button("Process URL"):
                with st.spinner("Processing URL..."):

                    try:
                        loader = FireCrawlLoader(api_key=FIRECRAWL_API_KEY, url=url, mode="scrape")
                        pages = [doc for doc in loader.lazy_load()]
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                        pages_chunks = text_splitter.split_documents(pages)
                        emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                        if selected_vector_store == "FAISS":
                            vector_store = FAISS.from_documents(pages_chunks, emb_model)
                        else:
                            persist_directory = 'db'
                            vector_store = Chroma.from_documents(documents=pages_chunks,
                                    embedding=emb_model,
                                    persist_directory=persist_directory)
                        st.success(f"File processed and stored in {selected_vector_store}")
                        retriever = vector_store.as_retriever()
                    except Exception as e:
                        st.error(f"Error processing URL: {e}")
                        st.stop()


elif resource_type == "YOUTUBE":
    url_input = st.sidebar.text_input("Enter Youtube video URL", placeholder="https://www.youtube.com/SBgGe_NX")
    if url_input and st.sidebar.button("Process Video"):
            with st.spinner("Processing URL..."):
                try:
                    llm = GoogleGenAI(model="gemini-2.0-flash")
                    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
                    Settings.llm = llm
                    Settings.chunk_size = 100
                    Settings.chunk_overlap = 20

                    links = [url_input]
                    loader = YoutubeTranscriptReader()
                    documents = loader.load_data(ytlinks=links)
                    index = VectorStoreIndex.from_documents(documents)
                    engine = index.as_query_engine() #use openAI for this step.
                    st.session_state.youtube_engine = engine #Store the engine in the session state.
                    st.success("Youtube video processed and indexed")

                except Exception as e:
                    st.error(f"Error processing {resource_type}: {e}")
                    st.stop()


# --- YouTube Query Section ---
if 'youtube_engine' in st.session_state:
    if user_query := st.chat_input("Ask me anything about the YouTube video: "):
        with st.spinner("Generating response..."):
            ai_response = st.session_state.youtube_engine.query(user_query)
            st.write(ai_response.response)

# --- Model Loading & QA Chain Setup for Document-Based Retrieval ---
if 'retriever' in locals():
    if selected_model == "Gemini-2.0-flash":
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY)
    elif selected_model == "Mistral-7B-Instruct-v0.3":
        st.warning("Mistral-7B-Instruct-v0.3 not available!")
        st.stop()
        # llm = HuggingFaceEndpoint(task='text-generation', model="mistralai/Mistral-7B-Instruct-v0.3", max_new_tokens=1024, temperature=0.3, huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)
    QA_PROMPT = PromptTemplate(template=template, input_variables=["context"])
    query_retriever_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
else:
    query_retriever_chain = None

# --- Main Content Area for Document-Based Retrieval ---
if query_retriever_chain:
    if user_query := st.chat_input("Ask me anything: "):
        with st.spinner("Generating response..."):
            response = query_retriever_chain({"query": user_query})
        st.write(f"🤖 AI Answer: {response['result']}")
        with st.expander("Source Documents"):
            for doc in response["source_documents"]:
                st.write(doc.page_content)
else:
    st.subheader(f"🔎 {resource_type} Analysis")
    user_query = st.chat_input("Ask a question based on the selected resource:")
    if user_query:
        st.error("No knowlegde-base available. Please ensure a valid resource is processed.")