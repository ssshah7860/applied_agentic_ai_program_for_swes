
import os
from pathlib import Path

import streamlit as st
import pandas as pd

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from docx import Document
from pypdf import PdfReader


# ---------------------------------------------------
# 1. LOAD ENVIRONMENT
# ---------------------------------------------------

root_env = Path(__file__).resolve().parents[2] / ".env"

load_dotenv(root_env)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is missing")


# ---------------------------------------------------
# 2. READ UPLOADED FILE
# ---------------------------------------------------

def read_uploaded_file(file):
    """Read CSV, Excel, Word, PDF, or TXT and return text."""

    extension = file.name.lower().split(".")[-1]

    # CSV
    if extension == "csv":
        df = pd.read_csv(file)
        return df.to_csv(index=False)

    # EXCEL
    elif extension == "xlsx":
        sheets = pd.read_excel(file, sheet_name=None)

        return "\n".join(
            f"Sheet: {name}\n{df.to_csv(index=False)}"
            for name, df in sheets.items()
        )

    # WORD
    elif extension == "docx":
        doc = Document(file)

        paragraphs = [
            p.text for p in doc.paragraphs
            if p.text.strip()
        ]

        tables = []
        for table in doc.tables:
            for row in table.rows:
                tables.append(
                    " | ".join(cell.text for cell in row.cells)
                )

        return "\n".join(paragraphs + tables)

    # PDF
    elif extension == "pdf":
        reader = PdfReader(file)

        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    # TEXT
    elif extension == "txt":
        return file.getvalue().decode("utf-8")

    else:
        return ""


# ---------------------------------------------------
# 3. CREATE LLM
# ---------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# ---------------------------------------------------
# 4. STREAMLIT USER INTERFACE
# ---------------------------------------------------

st.title("Hello Agent - Document FAQ Agent")

st.write(
    "Upload CSV, Excel, Word, PDF, or TXT files "
    "and ask questions about their contents."
)

uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["csv", "xlsx", "docx", "pdf", "txt"],
    accept_multiple_files=True
)


# ---------------------------------------------------
# 5. EXTRACT TEXT FROM ALL UPLOADED FILES
# ---------------------------------------------------

document_contents = []

for file in uploaded_files:

    st.subheader(file.name)

    try:
        text = read_uploaded_file(file)

        if text.strip():
            document_contents.append(
                f"FILE: {file.name}\n{text}"
            )

            st.success("File read successfully")

            # Display a small preview
            with st.expander("View extracted content"):
                st.text(text[:2000])

        else:
            st.warning("No readable text found in this file.")

    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")


# ---------------------------------------------------
# 6. ASK QUESTION
# ---------------------------------------------------

question = st.text_input(
    "Ask a question about your uploaded documents"
)


# ---------------------------------------------------
# 7. GENERATE ANSWER USING DOCUMENT CONTENT
# ---------------------------------------------------

if st.button("Submit"):

    if not uploaded_files:
        st.warning("Please upload a document first.")

    elif not document_contents:
        st.warning("No readable document content is available.")

    elif not question.strip():
        st.warning("Please enter a question.")

    else:

        # Combine all uploaded document contents
        context = "\n\n".join(document_contents)

        # Create prompt using document data
        prompt = f"""
You are a Document FAQ Assistant.

Answer the user's question using ONLY the uploaded
document content provided below.

Do not use outside knowledge.

If the answer is not available in the documents,
respond exactly:

"I could not find the answer in the uploaded documents."

Mention the source filename when possible.

DOCUMENT CONTENT:
{context}

USER QUESTION:cd 
{question}
"""

        # Send document content + question to LLM
        with st.spinner("Searching uploaded documents..."):
            response = llm.invoke(prompt)

        st.write("Your question:", question)

        st.subheader("AI Answer")
        st.write(response.content)
