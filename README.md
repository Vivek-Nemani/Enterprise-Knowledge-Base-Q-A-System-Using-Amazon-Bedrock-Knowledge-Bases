# RAG Bedrock Retrieve — Streamlit chat

Streamlit app that queries an [Amazon Bedrock Knowledge Base](https://aws.amazon.com/bedrock/) and uses Gemini for the response.

## Setup

1. Clone the repo and create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Copy environment variables:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your AWS region, Knowledge Base ID, and Gemini API key. Configure AWS credentials (`aws configure` or environment variables) so `boto3` can call Bedrock.

3. Run:

   ```bash
   streamlit run rag_bedrock_retreive.py
   ```

## License

Use at your own discretion for learning or projects.
