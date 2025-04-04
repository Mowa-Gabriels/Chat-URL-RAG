

import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint 
import os
import os
from dotenv import load_dotenv

load_dotenv()
import time




system_prompt = (
    "You are an exceptionally knowledgeable and precise AI assistant. "
    "Your primary task is to answer user questions based on the provided context. "
    "However, if the context is incomplete or does not fully address the query, you are allowed to think beyond the context and incorporate your broader understanding and reasoning. "
    "\n\n"
    "Please follow these guidelines:\n"
    "1. Primary Reliance on Context:** Begin your answer using the details in the provided context. Clearly reference or summarize the relevant parts if needed.\n"
    "2. Supplement with External Knowledge:** If the context does not offer a complete answer, supplement it with your broader expertise. When doing so, indicate that you are extending beyond the provided context.\n"
    "3. Factual Accuracy:** Ensure that any additional information is accurate and well-supported by your general knowledge. Avoid speculation unless explicitly necessary, and if you must speculate, note that it is an informed inference.\n"
    "4. Clarity and Conciseness:** Provide clear, well-organized, and succinct responses. Use bullet points or numbered lists for complex information.\n"
    "5. Admit Uncertainty:** If, after using both the context and your broader knowledge, the answer remains uncertain, clearly state: 'I don't know' or 'The available information is insufficient.'\n"
    "6. **Neutral and Professional Tone:** Maintain an objective, helpful, and professional tone throughout your response.\n\n"
    "Context: {context}"
)


HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")




# Check for API key  
if HUGGINGFACEHUB_API_TOKEN and GOOGLE_API_KEY:
    print("API key found.")
  
else:
    print("API key not found in .env file.")
    HUGGINGFACEHUB_API_TOKEN = input("Please enter your HuggingFcae API token: ")
    GOOGLE_API_KEY = input("Please enter your GOOGLE API KEY: ")
   
    with open('.env', 'a') as f:
        f.write(f'\nHUGGINGFACEHUB_API_TOKEN={HUGGINGFACEHUB_API_TOKEN}')
        f.write(f'\nGOOGLE_API_KEY={GOOGLE_API_KEY}')
    print("API key stored in .env file for future use.")



st.title("🔍 Chat-with-URL")

# Streamlit Sidebar Configuration
st.sidebar.title("🔧Configuration")
chunk_size = st.sidebar.slider("Chunk Size", min_value=100, max_value=1000, value=500, step=50)
selected_model = st.sidebar.selectbox("Select Model", ["gemini-1.5-pro", "Hugging Face Models"])

# User input for URLs
urls_input = st.sidebar.text_area("Enter URLs (comma-separated)", placeholder="https://blogpost.com, https://anotherblogpost.com" )
urls = [url.strip() for url in urls_input.split(",") if url.strip()]


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "ai", "content": "👋Welcome! Ask me anything based on the provided links."}
    ]

# Load and process documents
if urls:
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(data)
        
    emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
    vector_store = FAISS.from_documents(text_chunks, emb_model)
    vector_store.save_local("faiss_index")
    retriever = vector_store.as_retriever()
else:
    st.error("Please enter at least one URL.")


# Define the first LLM
llm_1 = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
)


# Let's define the alternaive LLM(one of those guys on huggingface)
huggingfacehub_api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

llm_2 = HuggingFaceEndpoint(
    task='text-generation',
    model="mistralai/Mistral-7B-Instruct-v0.3",
    max_new_tokens=1024,
    temperature=0.3,
    huggingfacehub_api_token=huggingfacehub_api_token
)

if selected_model == "gemini-1.5-pro":
    try:
        llm = llm_1
    except Exception as e:
        st.error("Google API limit reached. Switching to Alternatives...")
        llm = llm_2  # Our Backup model, just in case
        
else:
        st.warning("Hugging Face model support coming soon!")
        st.stop()

    



prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])


for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

if user_query := st.chat_input("Ask me anything: "):

    # Appends User Input Immediately and displays User Input Immediately
    st.session_state.chat_history.append({"role": "human", "content": user_query})
    with st.chat_message("human"):
        st.markdown(user_query)

     # Just a fancy Loader, nothing serious
        with st.spinner("🤖 Thinking..."):
            time.sleep(2)  # Simulate delay

    # Process AI Response
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": user_query})

    # This pretty much stores AI Response and displays AI Response
    ai_response = response["answer"]
    st.session_state.chat_history.append({"role": "ai", "content": ai_response})
    with st.chat_message("ai"):
        st.markdown(ai_response)

