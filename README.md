# Enterprise Knowledge Base Q&A (Amazon Bedrock + Gemini)

A Streamlit app that answers questions from your documents.

- **Bedrock mode** — Retrieval-Augmented Generation over an **Amazon Bedrock Knowledge Base** (Titan Text Embeddings V2, S3 data source).
- **Local document mode** — Upload a **PDF / TXT / Markdown** file and ask questions without any AWS dependency. Useful when AWS isn't available or for quick tests.

Answers are generated with **Google Gemini** (`gemini-2.5-flash`).

---

## Architecture

```
                 ┌────────────────────────┐
                 │        Streamlit       │
                 │  (Bedrock / Local tab) │
                 └──────────┬─────────────┘
                            │
     ┌──────────────────────┴───────────────────────┐
     │                                              │
     ▼                                              ▼
 Bedrock mode                                 Local mode
 ───────────                                  ──────────
 S3 → Bedrock KB → retrieve top chunks        PDF/TXT read with pypdf
       │                                       │
       ▼                                       ▼
           context ─────────────► Gemini (gemini-2.5-flash) ─► Answer
```

---

## Features

- Choose context source: **Bedrock Knowledge Base** or **Local document**.
- Bedrock retrieval via `bedrock-agent-runtime.retrieve` (top-k = 5).
- Local extraction with **pypdf** (PDF) and UTF-8 read for TXT/MD; long text is truncated to a safe character budget.
- Reads config from **Streamlit Secrets** (cloud) or **`.env`** (local).
- Clean error messages for missing keys / no context / AWS failures.

---

## Prerequisites

- Python 3.10+
- A **Google Gemini API key** — https://aistudio.google.com/app/apikey
- Optional (only for Bedrock mode):
  - An AWS account with **Amazon Bedrock** model access in your region
  - An **S3 bucket** with your documents
  - A **Bedrock Knowledge Base** pointing at that bucket (Titan Text Embeddings V2 as the embedding model)
  - AWS credentials configured locally (`aws configure`) or as environment variables

---

## Setup

```bash
git clone https://github.com/Vivek-Nemani/Enterprise-Knowledge-Base-Q-A-System-Using-Amazon-Bedrock-Knowledge-Bases.git
cd Enterprise-Knowledge-Base-Q-A-System-Using-Amazon-Bedrock-Knowledge-Bases

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Configuration

Create a `.env` in the project root (or copy `.env.example`):

```env
AWS_REGION=ap-south-1
KNOWLEDGE_BASE_ID=your-bedrock-knowledge-base-id
GEMINI_API_KEY=your-gemini-api-key
```

Make sure `.env` is git-ignored (it already is in this repo) and never committed.

### Streamlit Community Cloud / other hosts

Use **Secrets** instead of a `.env` file. In Streamlit Cloud → **Settings → Secrets**:

```toml
AWS_REGION = "ap-south-1"
KNOWLEDGE_BASE_ID = "your-bedrock-knowledge-base-id"
GEMINI_API_KEY = "your-gemini-api-key"
```

On other platforms (Render, Railway, Heroku, EC2, etc.), set the same keys as **environment variables**. The app reads `st.secrets` first, then falls back to environment variables.

---

## Run

```bash
streamlit run rag_bedrock_retreive.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

---

## Usage

1. **Context source** — pick **Amazon Bedrock Knowledge Base** or **Local document**.
2. For local mode, upload a `.pdf`, `.txt`, or `.md`.
3. Type your question and click **Ask**.
4. The answer and the retrieved / used context are shown.

---

## Project structure

```
.
├── rag_bedrock_retreive.py   # Streamlit app
├── requirements.txt          # Python dependencies
├── .env.example              # Template for local config
├── .gitignore
└── README.md
```

---

## Requirements

See `requirements.txt`:

- `streamlit`
- `boto3`
- `python-dotenv`
- `google-genai`
- `pypdf`

---

## Troubleshooting

- **`Set GEMINI_API_KEY ...` banner** — your Gemini key isn't set. Add it to `.env` locally or to Secrets / env vars on the host.
- **`ModuleNotFoundError: No module named 'boto3'`** — run `pip install -r requirements.txt` using the **same** Python that Streamlit uses (e.g. `/opt/anaconda3/bin/python -m pip install -r requirements.txt`).
- **Bedrock KB sync fails with HTTP 429 (“Too many requests”)** — throttling on the Titan embedding model. Wait and retry Sync; for sustained workloads, request higher on-demand quota for Titan Text Embeddings V2 or use Provisioned Throughput.
- **`Unable to locate credentials` from AWS CLI / boto3** — run `aws configure` and ensure the IAM user/role has `s3:GetObject` (and sync permissions for the KB) on your bucket.

---

## Acknowledgements

Built during my internship at **Innomatics Research Labs**.

## License

For learning and demonstration purposes. Use at your own discretion.
