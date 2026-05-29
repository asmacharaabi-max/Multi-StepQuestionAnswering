from flask import Flask, render_template, request
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
import os

from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# =========================
# LLM
# =========================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

parser = StrOutputParser()

# =========================
# STEP 1:
# Decompose Question
# =========================

decompose_prompt = PromptTemplate.from_template(
    """
You are an expert assistant.

Break this question into maximum 3 sub-questions.

Question:
{question}

Return ONLY:

1. ...
2. ...
3. ...
"""
)

decompose_chain = ( decompose_prompt | llm | parser )

# =========================
# Parse subquestions
# =========================

def parse_subquestions(text):
    lines = text.strip().split("\n")

    questions = []

    for line in lines:
        line = line.strip()

        if line.startswith("1.") or line.startswith("2.") or line.startswith("3."):
            questions.append(line[2:].strip())

    return questions


# =========================
# STEP 2:
# Answer Sub-question
# =========================

answer_prompt = PromptTemplate.from_template(
    """
Answer the following sub-question briefly.

Sub-question:
{subquestion}

Return:
- Short answer
- 2 bullet reasoning points
"""
)

answer_chain = ( answer_prompt | llm | parser)

# =========================
# STEP 3:
# Final Synthesis
# =========================

final_prompt = PromptTemplate.from_template(
    """
You are an expert summarizer.

Original Question:
{question}

Sub-question Answers:
{answers}

Provide a short final synthesized answer.
"""
)

final_chain = (final_prompt | llm | parser)

# =========================
# Flask Routes
# =========================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        question = request.form["question"]

        # -----------------------------
        # STEP 1: Decompose
        # -----------------------------

        subquestions_text = decompose_chain.invoke({
            "question": question
        })

        subquestions = parse_subquestions(subquestions_text)

        # -----------------------------
        # STEP 2: Parallel Answers
        # -----------------------------

        parallel_chain = RunnableParallel({
            f"q{i+1}": (
                RunnableLambda(
                    lambda x, sq=sq: {"subquestion": sq}
                )
                | answer_chain
            )
            for i, sq in enumerate(subquestions)
        })

        answers = parallel_chain.invoke({})

        # Format answers
        formatted_answers = ""

        for key, value in answers.items():
            formatted_answers += f"{key}:\\n{value}\\n\\n"

        # -----------------------------
        # STEP 3: Final Synthesis
        # -----------------------------

        final_answer = final_chain.invoke({
            "question": question,
            "answers": formatted_answers
        })

        result = {
            "question": question,
            "subquestions": subquestions,
            "answers": answers,
            "final_answer": final_answer
        }

    return render_template("test.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)

