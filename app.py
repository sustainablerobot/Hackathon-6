import os
import json
import re
from flask import Flask, request, jsonify
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# --- 1. SET UP THE FLASK APP ---
app = Flask(__name__)

# --- 2. SECURELY LOAD THE API KEY ---
# This will read the GOOGLE_API_KEY from Vercel's environment variables
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API Key not found. Please set GOOGLE_API_KEY in your environment variables.")
    os.environ["GOOGLE_API_KEY"] = api_key
    print("API Key loaded successfully.")
except Exception as e:
    print(f"Error loading API Key: {e}")

# --- 3. LOAD PDFs AND CREATE VECTOR STORE ON STARTUP ---
def create_vector_store():
    """Loads PDFs from the 'policy_docs' folder and creates a vector store."""
    all_documents = []
    docs_path = 'policy_docs'
    try:
        pdf_files = [f for f in os.listdir(docs_path) if f.endswith('.pdf')]
        if not pdf_files:
            print("No PDF files found in the 'policy_docs' folder.")
            return None

        print(f"Loading {len(pdf_files)} PDF(s)...")
        for file_name in pdf_files:
            file_path = os.path.join(docs_path, file_name)
            loader = PyPDFLoader(file_path)
            all_documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(all_documents)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(docs, embeddings)
        print("Vector store created successfully.")
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

# Create the vector store and RAG chain when the application starts
vector_store = create_vector_store()
rag_chain = LLMChain(
    llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, convert_system_message_to_human=True),
    prompt=PromptTemplate.from_template("""
    You are an expert insurance claim evaluator. Your task is to analyze a user's query against a set of relevant insurance policy clauses and determine if the claim should be approved.
    
    Here are the relevant policy clauses:
    ---
    {context}
    ---
    
    Here is the user's claim query:
    ---
    {query}
    ---
    
    Based *only* on the provided clauses and the user's query, perform the following steps:
    1.  Evaluate the query against the clauses.
    2.  Determine a final decision: "Approved" or "Rejected".
    3.  If approved, state the payout amount or coverage percentage if specified in the clauses. If no amount is specified, use "Not Applicable".
    4.  Provide a clear justification for your decision by referencing the specific clause(s) used.
    
    Return your final answer as a single, clean JSON object with no other text before or after it. The JSON object must have these exact keys: "decision", "amount", "justification".
    
    Final JSON Response:
    """)
)

def clean_and_parse_json(response_text):
    """Extracts and parses the JSON object from the AI's response."""
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if match:
        json_str = match.group(0)
        return json.loads(json_str)
    else:
        raise json.JSONDecodeError("No JSON object found in response", response_text, 0)

# --- 4. CREATE THE API ENDPOINT ---
@app.route('/evaluate', methods=['POST'])
def evaluate_claim():
    """Receives a query, evaluates it, and returns the result."""
    if not vector_store:
        return jsonify({"error": "Vector store is not available."}), 500

    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request body."}), 400

    user_query = data['query']
    
    try:
        # Retrieve relevant documents
        relevant_docs = vector_store.similarity_search(user_query)
        context = "\n".join([doc.page_content for doc in relevant_docs])

        # Run the RAG chain
        result = rag_chain.invoke({"context": context, "query": user_query})
        
        # Clean and return the JSON response
        response_json = clean_and_parse_json(result.get('text', '{}'))
        return jsonify(response_json)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return jsonify({"error": "Failed to process the request."}), 500

# This allows Vercel to run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
