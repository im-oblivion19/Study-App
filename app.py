import os
import json
import streamlit as st
import fitz  # PyMuPDF
from pydantic import BaseModel, Field
from typing import List

# OpenAI (newer SDK style)
from openai import OpenAI

# ----------------------------
# Models for structured output
# ----------------------------
class Flashcard(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)

class QuizQuestion(BaseModel):
    question: str = Field(..., min_length=1)
    options: List[str] = Field(..., min_items=4, max_items=4)
    correct_index: int = Field(..., ge=0, le=3)

class AIResult(BaseModel):
    summary: str
    flashcards: List[Flashcard] = Field(..., min_items=10, max_items=10)
    quiz: List[QuizQuestion] = Field(..., min_items=5, max_items=5)

# ----------------------------
# Helpers
# ----------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from uploaded PDF using PyMuPDF."""
    data = uploaded_file.read()
    doc = fitz.open(stream=data, filetype="pdf")
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    doc.close()
    return "\n".join(parts).strip()

def chunk_text(text: str, max_chars: int = 12000) -> str:
    """
    Keep it simple: cap input length so requests don't blow up.
    You can replace this later with real chunking + map-reduce summaries.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars]

def get_openai_client() -> OpenAI:
    key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
    key = key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set env var OPENAI_API_KEY or add it to .streamlit/secrets.toml"
        )
    return OpenAI(api_key=key)

def generate_study_material(text: str) -> AIResult:
    """
    Calls OpenAI and forces a strict JSON output compatible with AIResult schema.
    """
    client = get_openai_client()

    system = (
        "You are a study assistant. Given extracted text from a PDF, you must produce:\n"
        "1) A concise but high-signal summary.\n"
        "2) Exactly 10 flashcards (question/answer).\n"
        "3) Exactly 5 multiple-choice quiz questions with exactly 4 options each.\n"
        "Return ONLY valid JSON. No markdown, no extra commentary.\n"
        "Quiz correct_index must be 0-3.\n"
        "Flashcards and quiz must be grounded in the provided text."
    )

    user = f"PDF TEXT:\n{text}"

    # Choose a good general model name; you can change later.
    # If your account uses a different model, update it here.
    model_name = "gpt-4.1-mini"

    resp = client.chat.completions.create(
        model=model_name,
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    raw = resp.choices[0].message.content
    data = json.loads(raw)
    return AIResult.model_validate(data)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="PDF → Summary / Flashcards / Quiz", layout="wide")
st.title("📄➡️🧠 PDF Study Helper (Streamlit + PyMuPDF + OpenAI)")

with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    st.divider()
    st.caption("Tip: If the PDF is scanned images, this won't extract text well (needs OCR).")

if "ai_result" not in st.session_state:
    st.session_state.ai_result = None

if uploaded:
    st.subheader("1) Extracted Text Preview")
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded)

    if not pdf_text:
        st.error("No text found. This PDF might be scanned images. You'll need OCR for that.")
        st.stop()

    st.text_area("Extracted text (preview)", pdf_text[:4000], height=220)

    st.subheader("2) Generate Study Materials")
    colA, colB = st.columns([1, 2])
    with colA:
        if st.button("Generate Summary + Flashcards + Quiz", type="primary"):
            with st.spinner("Calling the AI..."):
                safe_text = chunk_text(pdf_text, max_chars=12000)
                try:
                    st.session_state.ai_result = generate_study_material(safe_text)
                    st.success("Done!")
                except Exception as e:
                    st.session_state.ai_result = None
                    st.error(f"Failed to generate. Error: {e}")

if st.session_state.ai_result:
    result: AIResult = st.session_state.ai_result

    st.divider()
    st.subheader("✅ Summary")
    st.info(result.summary)

    st.subheader("🧾 Flashcards")
    for i, fc in enumerate(result.flashcards, start=1):
        with st.expander(f"Flashcard {i}: {fc.question}"):
            st.write(fc.answer)

    st.subheader("📝 Quiz (5 Questions)")
    st.caption("Choose answers and hit Submit to see your score.")

    # Store user answers in session state so UI doesn't reset on rerun
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = [None] * len(result.quiz)
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False

    for qi, q in enumerate(result.quiz):
        st.markdown(f"**Q{qi+1}. {q.question}**")
        choice = st.radio(
            label=f"q_{qi}",
            options=list(range(4)),
            format_func=lambda idx: q.options[idx],
            index=st.session_state.quiz_answers[qi] if st.session_state.quiz_answers[qi] is not None else 0,
            key=f"radio_{qi}",
        )
        st.session_state.quiz_answers[qi] = choice

    if st.button("Submit Quiz"):
        st.session_state.quiz_submitted = True

    if st.session_state.quiz_submitted:
        correct = 0
        for qi, q in enumerate(result.quiz):
            if st.session_state.quiz_answers[qi] == q.correct_index:
                correct += 1

        st.success(f"Score: {correct} / {len(result.quiz)}")

        with st.expander("Review Answers"):
            for qi, q in enumerate(result.quiz):
                user_idx = st.session_state.quiz_answers[qi]
                st.write(f"**Q{qi+1}. {q.question}**")
                st.write(f"Your answer: {q.options[user_idx]}")
                st.write(f"Correct answer: {q.options[q.correct_index]}")
                st.divider()
else:
    st.caption("Upload a PDF to begin.")
