import os
# Set environment variables to avoid PyTorch issues with Streamlit's watcher
os.environ["PYTHONPATH"] = os.getcwd()
os.environ["PYTORCH_JIT"] = "0"

import streamlit as st
import pickle
import time
import logging
import traceback
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Configure logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("seedhe_Bot_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log startup
logger.info("Starting Seedhe Bot application")

# Load environment variables
load_dotenv()  # take environment variables from .env (especially Google API key)
logger.debug(f"Environment loaded. GOOGLE_API_KEY present: {bool(os.getenv('GOOGLE_API_KEY'))}")

# Function to list available models
def list_available_models():
    """List all available Gemini models for the configured API key"""
    try:
        models = genai.list_models()
        available_models = [model.name for model in models if "gemini" in model.name.lower()]
        return available_models
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        return []

# Configure Google Gemini API
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    logger.info("Google Gemini API configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {str(e)}")
    logger.error(traceback.format_exc())

# Set up the Streamlit page
st.title("seedhe Bot 📈")
st.sidebar.title("News Article URLs")

# URL inputs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

file_path = "faiss_store_gemini.pkl"
main_placeholder = st.empty()

# Debug panel (can be toggled in sidebar)
show_debug = st.sidebar.checkbox("Show Debug Info", value=False)
debug_container = st.sidebar.container()

# Function to update debug info
def update_debug_info(message):
    if show_debug:
        with debug_container:
            st.text(f"[DEBUG] {message}")
    logger.debug(message)

update_debug_info("Application initialized")

# Initialize Gemini model - CORRECTLY CONFIGURED
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.7,
        max_output_tokens=500,
        top_k=40,
        top_p=0.8,
        # api_version="v1"  # Explicitly set the API version
    )
    update_debug_info("LLM initialized successfully with gemini-pro")
except Exception as e:
    error_msg = f"Failed to initialize LLM with gemini-pro: {str(e)}"
    logger.error(error_msg)
    
    try:
        # Fallback to another model
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",  # Try another available model
            temperature=0.7,
            max_output_tokens=500,
            top_k=40,
            top_p=0.8,
            api_version="v1"  # Also explicitly set the API version
        )
        update_debug_info("LLM initialized successfully with fallback to gemini-1.5-pro")
    except Exception as e2:
        # Try one more fallback
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.0-pro",
                temperature=0.7,
                max_output_tokens=500,
                top_k=40,
                top_p=0.8,
                api_version="v1"
            )
            update_debug_info("LLM initialized successfully with fallback to gemini-1.0-pro")
        except Exception as e3:
            error_msg = f"Failed to initialize LLM with all fallbacks: {str(e3)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            st.error("Failed to initialize Gemini API. Please check your API key and model availability.")

# Add model checker to sidebar
with st.sidebar.expander("Available Models"):
    if st.button("Check Available Models"):
        models = list_available_models()
        if models:
            st.write("Available Gemini models:")
            for model in models:
                st.write(f"- {model}")
        else:
            st.write("Failed to retrieve models or none available")

# Modified function to fix the caching issue
@st.cache_data
def get_text_chunks(_texts):
    """Split texts into chunks with caching to improve performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    return text_splitter.split_documents(_texts)

if process_url_clicked:
    # Filter out empty URLs
    valid_urls = [url for url in urls if url]
    
    if not valid_urls:
        st.sidebar.error("Please enter at least one URL")
        update_debug_info("No valid URLs provided")
    else:
        try:
            update_debug_info(f"Processing URLs: {valid_urls}")
            
            # Load data
            loader = UnstructuredURLLoader(urls=valid_urls)
            main_placeholder.text("Data Loading...Started...✅✅✅")
            data = loader.load()
            update_debug_info(f"Loaded {len(data)} documents from URLs")
            
            # Split data
            main_placeholder.text("Text Splitter...Started...✅✅✅")
            docs = get_text_chunks(data)
            update_debug_info(f"Split into {len(docs)} chunks")
            
            # Create embeddings and save it to FAISS index
            try:
                # Primary embedding method
                update_debug_info("Initializing HuggingFace embeddings")
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                update_debug_info("HuggingFace embeddings initialized successfully")
            except Exception as e:
                # Fallback if HuggingFace fails
                error_msg = f"HuggingFace embeddings failed: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                
                try:
                    # Try a simpler model
                    update_debug_info("Trying simpler embedding model")
                    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L3-v2")
                    update_debug_info("Simpler embeddings model initialized successfully")
                except Exception as e2:
                    error_msg = f"Simpler embeddings model also failed: {str(e2)}"
                    logger.error(error_msg)
                    st.error("Unable to initialize embeddings. Please see logs for details.")
                    if show_debug:
                        st.error(error_msg)
                    raise e2
            
            # Create vector store
            main_placeholder.text("Embedding Vector Started Building...✅✅✅")
            update_debug_info("Creating FAISS vector store")
            vectorstore = FAISS.from_documents(docs, embeddings)
            update_debug_info("FAISS vector store created successfully")
            time.sleep(2)
            
            # Save the FAISS index to a pickle file
            update_debug_info(f"Saving vector store to {file_path}")
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore, f)
            update_debug_info("Vector store saved successfully")
            
            main_placeholder.text("Ready to answer questions! ✨")
            
        except Exception as e:
            error_msg = f"Error processing URLs: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            main_placeholder.error("❌ Error processing URLs. See debug info for details.")
            if show_debug:
                st.error(error_msg)
                st.error(traceback.format_exc())

# Query input
query = st.text_input("Question: ")

if query:
    update_debug_info(f"Processing query: {query}")
    
    if os.path.exists(file_path):
        try:
            update_debug_info("Loading vector store from file")
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
            update_debug_info("Vector store loaded successfully")
            
            # Create QA chain
            update_debug_info("Creating QA chain")
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            update_debug_info("QA chain created successfully")
            
            # Get response with better error handling
            update_debug_info("Generating response")
            try:
                response = qa_chain({"query": query})
                update_debug_info("Response generated successfully")
                
                # Display answer
                st.header("Answer")
                st.write(response["result"])
                
                # Display sources
                st.subheader("Sources:")
                for i, doc in enumerate(response["source_documents"]):
                    st.write(f"Source {i+1}:")
                    st.write(doc.page_content[:300] + "...")
                    st.write(f"Source URL: {doc.metadata.get('source', 'Unknown')}")
                    st.write("---")
            except Exception as query_error:
                error_msg = f"Error querying Gemini API: {str(query_error)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                st.error("❌ Error processing your question. See debug info for details.")
                
                # Show diagnostic info
                if show_debug:
                    st.error(error_msg)
                    st.error(traceback.format_exc())
                    
                # Recommend solutions
                st.warning("""
                *Possible solutions:*
                1. Check if your Google API key is valid and has access to the Gemini API
                2. Try using a different Gemini model version
                3. Make sure you have the latest packages installed
                4. If the error persists, try restarting the application
                """)
                
                # Try to list available models to help debugging
                st.info("Checking available models...")
                models = list_available_models()
                if models:
                    st.info("Available Gemini models that could be used instead:")
                    for model in models:
                        st.write(f"- {model}")
                else:
                    st.info("Could not retrieve available models. Please check your API key.")
                
        except Exception as e:
            error_msg = f"Error setting up QA system: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            st.error("❌ Error processing your question. See debug info for details.")
            if show_debug:
                st.error(error_msg)
                st.error(traceback.format_exc())
    else:
        update_debug_info("No processed URLs found")
        st.error("Please process URLs first!")

# Add a section for troubleshooting tips
with st.sidebar.expander("Troubleshooting Tips"):
    st.markdown("""
    ### Common Issues:
    
    1. *Installation Problems:*
       - Try: pip install --only-binary :all: sentencepiece
       - Or use Conda: conda install -c conda-forge sentence-transformers
       
    2. *Embedding Errors:*
       - Try a simpler embedding model like "paraphrase-MiniLM-L3-v2"
       - Check your internet connection for model downloads
       
    3. *URL Loading Issues:*
       - Ensure URLs are accessible and not behind login walls
       - Install additional dependencies: pip install bs4 html5lib lxml
       
    4. *Memory Errors:*
       - Reduce chunk size in the text splitter
       - Process fewer URLs at once
       
    5. *API Key Issues:*
       - Check that your .env file contains GOOGLE_API_KEY
       - Verify API key is valid with proper permissions
       
    6. *Gemini API Issues:*
       - Use a valid model name: "gemini-pro", "gemini-1.5-pro", etc.
       - Always set api_version="v1" explicitly
       - Use the "Check Available Models" option to see what's accessible
    """)

# Add status indicator in footer
st.sidebar.markdown("---")
st.sidebar.text("App Status: Running ✅")
st.sidebar.text(f"Debug Mode: {'On' if show_debug else 'Off'}")