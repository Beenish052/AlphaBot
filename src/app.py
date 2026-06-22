"""
app.py
------
Streamlit web interface for ALFA, Beenish's personal RAG-based chatbot.

Run locally with:
    streamlit run src/app.py
"""

import os
import streamlit as st
from rag_pipeline import RAGPipeline, BOT_NAME

st.markdown("""
<div style="
    background: linear-gradient(135deg, #1E1E2E, #2A2A40);
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #3A3A5A;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    margin-bottom: 15px;
">
    <h2 style="margin:0; color:#FFFFFF;">
        🤖 ALFA
    </h2>
    <p style="color:#D1D5DB; font-size:16px; line-height:1.6; margin-top:10px;">
        <b>ALFA</b> (NLP-Powered Oracle for Versatile Answering) is a
        <b>RAG-based chatbot</b> that provides information about
        <b>Beenish's Profile, Skills, Projects, Experience, and Education</b>.
        It was developed as a semester project for the
        <b>Natural Language Processing (CC438)</b> course.
    </p>
</div>
""", unsafe_allow_html=True)




# ---------- API key handling ----------
def get_api_key():
    """Look for the Groq API key in Streamlit secrets, then env vars."""
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY")


# ---------- Session state ----------
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"user": ..., "bot": ...}

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None


# ---------- Sidebar ----------
with st.sidebar:
    st.title(f"🤖 {BOT_NAME}")
    st.markdown(
    "**🤖 ALFA**\n\n"
    "ALFA (NLP-Powered Oracle for Versatile Answering) is a RAG-based chatbot "
    "that provides information about Beenish's profile, skills, projects, "
    "experience, and education. It was developed as a semester project for "
    "the Natural Language Processing course (CC438)."
)
    st.divider()

    api_key_input = st.text_input(
        "Groq API Key",
        type="password",
        help="Get a free key at https://console.groq.com/keys",
        value=get_api_key() or "",
    )

    st.divider()
    st.markdown("**Suggested Questions:**")
    st.markdown(
    "- Who is Beenish Bhatti?\n"
    "- What is Beenish studying?\n"
    "- Tell me about the ALFA chatbot project\n"
    "- What are Beenish's technical skills?\n"
    "- What projects has Beenish worked on?"
)
    

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.history = []
        st.rerun()


# ---------- Main UI ----------
st.title(f"💬 {BOT_NAME}")
st.caption("Ask anything about Beenish")

if not api_key_input:
    st.warning(
        "Please enter a Groq API key in the sidebar to start chatting. "
        "Get a free key at https://console.groq.com/keys"
    )
    st.stop()

# Initialize (or re-initialize) the pipeline if needed
if st.session_state.pipeline is None:
    try:
        with st.spinner("Loading ALFA's knowledge base..."):
            st.session_state.pipeline = RAGPipeline(api_key=api_key_input)
    except FileNotFoundError as e:
        st.error(
            f"{e}\n\nRun `python src/ingest.py` first to build the vector store."
        )
        st.stop()
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {e}")
        st.stop()

# Render past conversation
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["user"])
    with st.chat_message("assistant"):
        st.markdown(turn["bot"])

# Chat input
user_query = st.chat_input("Ask ALFA something about Beenish...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.pipeline.generate_response(
                    query=user_query,
                    history=st.session_state.history,
                )
                answer = result["answer"]
            except Exception as e:
                answer = f"Sorry, I ran into an error: {e}"

        st.markdown(answer)

    st.session_state.history.append({"user": user_query, "bot": answer})
