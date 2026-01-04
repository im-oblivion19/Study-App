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
    quiz: List[QuizQuestion] = Field(..., min_items=10, max_items=10)

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

import random

def shuffle_quiz_in_place(ai_result, seed: int | None = None):
    """
    Randomizes answer option order for each quiz question while preserving correctness.
    Works even if the model always puts the correct answer at options[0].
    """
    rng = random.Random(seed)

    for q in ai_result.quiz:
        # Save the correct answer text before shuffling
        correct_text = q.options[q.correct_index]

        # Shuffle a copy of options
        shuffled = list(q.options)
        rng.shuffle(shuffled)

        # Update question options + correct index
        q.options = shuffled
        q.correct_index = shuffled.index(correct_text)

    return ai_result


def generate_study_material(text: str) -> AIResult:
    """
    Calls OpenAI and forces a strict JSON output compatible with AIResult schema.
    """
    client = get_openai_client()    

    system = (
        "You are a study assistant. Given extracted text from a PDF, you must produce:\n"
        "1) A concise but high-signal summary. It should contain all important keywords. \n"
        "2) Exactly 10 flashcards (question/answer).\n"
        "3) Exactly 10 multiple-choice quiz questions with exactly 4 options each.\n"
        "Return ONLY valid JSON. No markdown, no extra commentary.\n"
        "Quiz correct_index must be 0-3. Make sure the correct index is a random number between 0 and 3 for each question\n"
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

    result = AIResult.model_validate(data)
    result = shuffle_quiz_in_place(result)

    return result



# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="StudyPixel", layout="wide")
st.title("StudyPixel📃➡️🧠\n Upload a PDF and instantly get a summary, flashcards, and a quiz!")


with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    st.divider()
    st.caption("Tip: If the PDF is scanned images, this won't extract text well (needs OCR).")

# ----------------------------
# Session state initialization
# ----------------------------
if "ai_result" not in st.session_state:
    st.session_state.ai_result = None
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = None
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = None
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

# ----------------------------
# PDF extraction (do once)
# ----------------------------
if uploaded:
    st.subheader("1) Extracted Text Preview")
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded)

    if not pdf_text:
        st.error("No text found. This PDF might be scanned images. You'll need OCR for that.")
        st.stop()

    st.session_state.pdf_text = pdf_text
    st.text_area("Extracted text (preview)", pdf_text[:4000], height=220)

    # ----------------------------
    # Generate (single source of truth)
    # ----------------------------
    st.subheader("2) Generate Study Materials")
    colA, colB = st.columns([1, 2])
    with colA:
        if st.button("Generate Summary + Flashcards + Quiz", type="primary"):
            with st.spinner("Calling the AI..."):
                safe_text = chunk_text(st.session_state.pdf_text, max_chars=12000)
                try:
                    st.session_state.ai_result = generate_study_material(safe_text)
                    st.success("Done!")
                    # Reset quiz state for new results
                    st.session_state.quiz_answers = [None] * len(st.session_state.ai_result.quiz)
                    st.session_state.quiz_submitted = False
                except Exception as e:
                    st.session_state.ai_result = None
                    st.error(f"Failed to generate. Error: {e}")

# ----------------------------
# Render results (only if exists)
# ----------------------------
if st.session_state.ai_result:
    result: AIResult = st.session_state.ai_result

    st.divider()
    st.subheader("✅ Summary")
    st.info(result.summary)

    st.subheader("🧾 Flashcards")
    for i, fc in enumerate(result.flashcards, start=1):
        with st.expander(f"Flashcard {i}: {fc.question}"):
            st.write(fc.answer)

    st.subheader("📝 Quiz")
    st.caption("Choose the appropriate answer for each question and hit Submit to see your score. Good luck!")

    # Ensure quiz state is initialized (covers refresh/rerun cases)
    if st.session_state.quiz_answers is None or len(st.session_state.quiz_answers) != len(result.quiz):
        st.session_state.quiz_answers = [None] * len(result.quiz)
        st.session_state.quiz_submitted = False

    for qi, q in enumerate(result.quiz):
        st.markdown(f"**Q{qi+1}. {q.question}**")

        # IMPORTANT: don't default to 0, because that makes it look like option 1 is "always chosen"
        current = st.session_state.quiz_answers[qi]
        index = 0 if current is None else current

        choice = st.radio(
            label="",
            options=list(range(4)),
            format_func=lambda idx: q.options[idx],
            index=index,
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

                if user_idx is None:
                    st.write("Your answer: (no answer selected)")
                else:
                    st.write(f"Your answer: {q.options[user_idx]}")

                st.write(f"Correct answer: {q.options[q.correct_index]}")
                st.divider()
else:
    st.caption("Upload a PDF to begin.")
