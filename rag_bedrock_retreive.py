import os
import boto3
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from google import genai

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

st.title("Simple Bedrock Knowledge Base Chat")
st.markdown('<p class="subtitle">Ask a question to your Amazon Bedrock Knowledge Base.</p>', unsafe_allow_html=True)

aws_region = os.getenv("AWS_REGION", "eu-north-1")
knowledge_base_id = os.getenv("KNOWLEDGE_BASE_ID", "DDSGHZ5O1L")
gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("gemini_key")
if gemini_api_key:
    os.environ["GEMINI_API_KEY"] = gemini_api_key


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
"I could not find that in the knowledge base context."

Context:
{context_text}

Question:
{question_text}
"""
    client = genai.Client()
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text or "No answer was generated."


question = st.text_input("Enter your question", placeholder="Type your question here...")

if st.button("Ask", type="primary"):
    if not knowledge_base_id:
        st.error("Set `KNOWLEDGE_BASE_ID` in your environment variables.")
    elif not gemini_api_key:
        st.error("Set `GEMINI_API_KEY` (or `gemini_key`) in your environment variables.")
    elif not question.strip():
        st.error("Please enter a question.")
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