## ✨ Features

* **Multi-Domain Framework**: The system is designed with a dynamic "AI Personalities" architecture, allowing it to be an expert on various domains like Insurance, Legal, HR, and Contract Management.
* **Contextual Q&A**: Instead of just searching, users can ask complex, natural language questions and get direct, reasoned answers based on the provided documents.
* **Multi-Format Support**: Can ingest and process multiple file types, including PDFs (`.pdf`), Word Documents (`.docx`), and Emails (`.eml`).
* **Live API**: The entire ML logic is exposed via a secure Flask & ngrok API, ready for frontend integration.

## 🛠️ Tech Stack

* **ML / Processing**: Python, LangChain, Google Generative AI (Gemini 1.5 Flash), FAISS, Unstructured.io
* **Backend / API**: Flask, ngrok
* **Frontend**: Next.js, React.js, Tailwind CSS

## 🚀 Getting Started

Follow these steps to set up and run the backend server on your local machine.

### Prerequisites

* Python 3.10+
* A Google AI API Key
* A free [ngrok](https://ngrok.com/) authtoken

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

### 2. Set Up the Environment

It's highly recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install all required libraries
pip install -r requirements.txt
```

### 3. Add Knowledge Base Documents

Place your documents into the correct sub-folder inside the `knowledge_base/` directory.

```
knowledge_base/
├── insurance/
│   └── policy1.pdf
├── legal/
│   └── contract.docx
└── hr/
    └── handbook.pdf
```

### 4. Set Environment Variables

You need to set your API keys. The most secure way is to set them as environment variables.

```bash
# On macOS/Linux:
export GOOGLE_API_KEY="YOUR_GOOGLE_AI_API_KEY"
export NGROK_TOKEN="YOUR_NGROK_AUTHTOKEN"

# On Windows (Command Prompt):
set GOOGLE_API_KEY="YOUR_GOOGLE_AI_API_KEY"
set NGROK_TOKEN="YOUR_NGROK_AUTHTOKEN"
```

## 🏃 Running the API Server

Run the `app.py` script from your terminal and specify which AI expert you want to activate. The script will automatically load the documents from the corresponding sub-folder.

* To run the **Insurance Expert**:
    ```bash
    python app.py insurance
    ```
* To run the **Legal Expert**:
    ```bash
    python app.py legal
    ```

The server will start, and a public `ngrok` URL will be printed. This is your live API endpoint.

## 📖 API Documentation

The frontend should send `POST` requests to the generated `ngrok` URL.

**Endpoint**: `/predict`
**Method**: `POST`

### Request Body

The frontend must send a JSON object with the user's query and the chosen domain.

```json
{
    "query": "What is the waiting period for maternity expenses?",
    "domain": "insurance"
}
```

### Success Response (Example)

The API will return a JSON object with the AI's analysis.

```json
{
  "decision": "Rejected",
  "amount": "Not Applicable",
  "justification": "The claim for maternity expenses is rejected because the policy specifies a 6-year waiting period for the Exclusive plan's maternity benefit."
}
```
