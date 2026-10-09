
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Application title
st.title("Hello Agent - CSV FAQ Agent")
st.write("Upload CSV files and ask questions about your data.")

# Allow uploading multiple CSV files
uploaded_files = st.file_uploader(
    "Upload your CSV files",
    type=["csv"],
    accept_multiple_files=True
)

# Read and display each uploaded CSV file
for file in uploaded_files:
    df = pd.read_csv(file)

    st.subheader(file.name)
    st.dataframe(df.head())

    # Ask the user a question
question = st.text_input("Ask a question about your CSV data")

# Submit button
if st.button("Submit"):
    if question:
        st.write("Your question:", question)
    else:
        st.warning("Please enter a question.")



# Read environment variables from .env
load_dotenv()

# Retrieve the API key
api_key = os.getenv("OPENAI_APIKEY")

# Check whether the key exists
if api_key:
    print("OpenAI API key loaded successfully!!!")
else:
    print("OpenAI API key not found.")        

# Create an OpenAI model object

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

response = llm.invoke(question)
st.write("AI Answer:", response.content)

