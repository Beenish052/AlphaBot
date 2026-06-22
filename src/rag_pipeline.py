"""
rag_pipeline.py
----------------
Core RAG logic: loads the FAISS vector store, retrieves relevant
context for a user query, and generates a response using Groq
LLM with conversation history and prompt engineering.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")

BOT_NAME = "ALFA"

SYSTEM_PROMPT = """You are {bot_name} (NLP-Powered Oracle for Versatile
Answering), a personal AI assistant created by Beenish, a Software
Engineering student, for her NLP semester project (CC438) at UMT Lahore.
You answer questions about Beenish's academic profile, skills, projects,
and NLP course knowledge using ONLY the context provided below, retrieved
from her personal documents.

Rules:
- Answer naturally and conversationally, refer to her as "Beenish" or "she".
- Use ONLY the information in the CONTEXT section below. Do not invent facts,
  grades, dates, or numbers that are not present in the context.
- If the answer isn't in the context, say so honestly — e.g. "I don't have
  that information in my knowledge base" — instead of guessing.
- Keep answers concise and relevant to the question asked.
- Always introduce yourself as ALFA when asked who you are.
- Use the conversation history to understand follow-up questions.

CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

CURRENT QUESTION:
{question}

Answer as {bot_name}:"""


class RAGPipeline:
    def __init__(self, api_key: str, model_name: str = "llama-3.1-8b-instant"):
        if not api_key:
            raise ValueError("A Groq API key is required.")

        self.client = Groq(api_key=api_key)
        self.model_name = model_name

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        if not os.path.exists(VECTORSTORE_DIR):
            raise FileNotFoundError(
                "Vector store not found. Run `python src/ingest.py` first."
            )

        self.vectorstore = FAISS.load_local(
            VECTORSTORE_DIR,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def retrieve_context(self, query: str, k: int = 6) -> str:
        """Retrieve top-k relevant chunks for the query."""
        docs = self.vectorstore.similarity_search(query, k=k)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        return context

    def format_history(self, history: list, max_turns: int = 4) -> str:
        """Format the last few conversation turns as plain text."""
        if not history:
            return "No previous conversation."

        recent = history[-max_turns:]
        formatted = []
        for turn in recent:
            formatted.append(f"User: {turn['user']}")
            formatted.append(f"{BOT_NAME}: {turn['bot']}")
        return "\n".join(formatted)

    def generate_response(self, query: str, history: list = None) -> dict:
        """Run the full RAG pipeline: retrieve -> prompt -> generate."""
        history = history or []

        context = self.retrieve_context(query)
        formatted_history = self.format_history(history)

        prompt = SYSTEM_PROMPT.format(
            bot_name=BOT_NAME,
            context=context,
            history=formatted_history,
            question=query,
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.3,
        )

        answer = response.choices[0].message.content.strip()

        return {
            "answer": answer,
            "context_used": context,
        }
