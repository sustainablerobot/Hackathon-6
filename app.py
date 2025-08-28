import os
import json
import re
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# --- 1. SET UP THE FLASK APP ---
app = Flask(__name__)
CORS(app)
# Create a temporary folder for uploads
UPLOAD_FOLDER = 'temp_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# In-memory storage for user vector stores (for demo purposes)
# A more robust solution would use a database or a proper cache
vector_stores = {}

# --- 2. SECURELY LOAD THE API KEY ---
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API Key not found. Please set GOOGLE_API_KEY in your environment variables.")
    os.environ["GOOGLE_API_KEY"] = api_key
    print("API Key loaded successfully.")
except Exception as e:
    print(f"Error loading API Key: {e}")

# --- 3. CREATE THE API ENDPOINTS ---

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handles file uploads, creates a vector store, and returns a session ID."""
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request."}), 400
    
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No selected files."}), 400

    saved_paths = []
    all_documents = []
    
    for file in files:
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            saved_paths.append(file_path)
            
            # Load documents from the saved PDF
            loader = PyPDFLoader(file_path)
            all_documents.extend(loader.load())
        else:
            return jsonify({"error": "Invalid file type. Only PDFs are accepted."}), 400

    try:
        # Create vector store from the uploaded documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(all_documents)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(docs, embeddings)
        
        # Clean up saved files after processing
        for path in saved_paths:
            os.remove(path)
            
        # Create a unique session ID and store the vector store
        session_id = str(uuid.uuid4())
        vector_stores[session_id] = vector_store
        
        return jsonify({"message": "Files processed successfully.", "session_id": session_id})
        
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return jsonify({"error": "Failed to process files."}), 500


@app.route('/query', methods=['POST'])
def query_documents():
    """Receives a query and session ID, and returns the evaluation."""
    data = request.get_json()
    if not data or 'query' not in data or 'session_id' not in data:
        return jsonify({"error": "Missing 'query' or 'session_id' in request body."}), 400

    session_id = data['session_id']
    user_query = data['query']
    
    if session_id not in vector_stores:
        return jsonify({"error": "Invalid or expired session ID."}), 404
        
    vector_store = vector_stores[session_id]

    try:
        # The RAG chain logic from before
        rag_chain = LLMChain(
            llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, convert_system_message_to_human=True),
            prompt=PromptTemplate.from_template("""
            You are an expert document evaluator. Your task is to analyze a user's query against a set of relevant clauses from a document they provided.
            Here are the relevant clauses:
            ---
            {context}
            ---
            Here is the user's query:
            ---
            {query}
            ---
            Based *only* on the provided context from the document, answer the user's query. If the context is not sufficient to answer, state that.
            Provide a clear and concise answer.
            """)
        )

        relevant_docs = vector_store.similarity_search(user_query)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        result = rag_chain.invoke({"context": context, "query": user_query})
        
        return jsonify({"answer": result.get('text', 'No answer found.')})
        
    except Exception as e:
        print(f"Error during query: {e}")
        return jsonify({"error": "Failed to process query."}), 500

# This allows Vercel/Render to run the Flask app
if __name__ == "__main__":
    app.run(debug=True)