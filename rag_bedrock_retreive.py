import io
import os
import boto3
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader


load_dotenv()

st.set_page_config(page_title="Simple Bedrock Knowledge Base Chat", page_icon="💬", layout="centered")

st.markdown(
    """
    <style>
        .block-container {
            max-width: 760px;
            padding-top: 2.5rem;
            padding-bottom: 2rem;
        }
        .subtitle {
            color: #c7c9d1;
            margin-bottom: 1rem;
        }
        .stButton button {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Bedrock Knowledge Base Chat")
st.markdown(
    '<p class="subtitle">Use your Bedrock Knowledge Base, or upload a local document when AWS is unavailable.</p>',
    unsafe_allow_html=True,
)

MAX_LOCAL_CONTEXT_CHARS = 120_000

def _get_secret(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            value = st.secrets[key]
            if value:
                return str(value)
    except Exception:
        pass
    return os.getenv(key, default)


aws_region = _get_secret("AWS_REGION", "ap-south-1")
knowledge_base_id = _get_secret("KNOWLEDGE_BASE_ID", "DDSGHZ5O1L")
gemini_api_key = _get_secret("GEMINI_API_KEY") or _get_secret("gemini_key")
if gemini_api_key:
    os.environ["GEMINI_API_KEY"] = gemini_api_key


def extract_text_from_upload(uploaded_file) -> str:
    name = (uploaded_file.name or "").lower()
    data = uploaded_file.getvalue()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n\n".join(parts).strip()
    return data.decode("utf-8", errors="replace").strip()


def truncate_for_context(text: str):
    if len(text) <= MAX_LOCAL_CONTEXT_CHARS:
        return (text, False)
    return (text[:MAX_LOCAL_CONTEXT_CHARS], True)


def retrieve_context(question_text: str) -> str:
    bedrock = boto3.client("bedrock-agent-runtime", region_name=aws_region)
    kb_response = bedrock.retrieve(
        knowledgeBaseId=knowledge_base_id,
        retrievalQuery={"text": question_text},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 5}},
    )

    chunks = []
    for item in kb_response.get("retrievalResults", []):
        text = item.get("content", {}).get("text", "")
        if text:
            chunks.append(text)
    return "\n\n".join(chunks)


def generate_answer(question_text: str, context_text: str) -> str:
    prompt = f"""
You are a helpful assistant.
Answer the user's question using only the context below.
If the answer is not found in the context, say:
"I could not find that in the document context."

Context:
{context_text}

Question:
{question_text}
"""
    client = genai.Client()
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text or "No answer was generated."


context_source = st.radio(
    "Context source",
    ["Amazon Bedrock Knowledge Base", "Local document"],
    horizontal=True,
    help="Choose Local document when AWS/Bedrock is unavailable; answers still use Gemini with your file as context.",
)

extracted_local = ""
uploaded_file = None
if context_source == "Local document":
    uploaded_file = st.file_uploader(
        "Upload a document (PDF, TXT, or Markdown)",
        type=["pdf", "txt", "md"],
        help="Text is sent to Gemini as context (large files are truncated).",
    )
    if uploaded_file is not None:
        try:
            extracted_local = extract_text_from_upload(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
    if uploaded_file is not None and not extracted_local.strip():
        st.warning("No text could be extracted from this file. Try another PDF or a .txt/.md file.")

question = st.text_input("Enter your question", placeholder="Type your question here...")

if st.button("Ask", type="primary"):
    if not gemini_api_key:
        st.error("Set `GEMINI_API_KEY` (or `gemini_key`) in your environment variables.")
    elif not question.strip():
        st.error("Please enter a question.")
    elif context_source == "Local document":
        if not extracted_local.strip():
            st.error("Upload a document with readable text first.")
        else:
            try:
                context, was_truncated = truncate_for_context(extracted_local)
                if was_truncated:
                    st.caption(
                        f"Using the first {MAX_LOCAL_CONTEXT_CHARS:,} characters of the document for context."
                    )
                with st.spinner("Generating answer with Gemini..."):
                    answer = generate_answer(question, context)
                st.subheader("Answer")
                st.write(answer)
                with st.expander("Context sent to the model"):
                    st.write(context)
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
    else:
        if not knowledge_base_id:
            st.error("Set `KNOWLEDGE_BASE_ID` in your environment variables.")
        else:
            try:
                with st.spinner("Retrieving knowledge base context..."):
                    context = retrieve_context(question)

                if not context.strip():
                    st.warning("No context returned from Bedrock Knowledge Base.")
                else:
                    with st.spinner("Generating answer with Gemini..."):
                        answer = generate_answer(question, context)
                    st.subheader("Answer")
                    st.write(answer)

                with st.expander("Retrieved Context"):
                    st.write(context if context else "No context returned.")
            except (ClientError, BotoCoreError) as exc:
                st.error(f"AWS Bedrock error: {exc}")
            except Exception as exc:  # Broad catch for API and runtime issues.
                st.error(f"Something went wrong: {exc}")
