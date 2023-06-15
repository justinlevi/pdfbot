from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import tiktoken
import logging

hide_st_style = """
  <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
  </style>
"""

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize session state
def initialize_session_state():
    if 'amt' not in st.session_state:
        st.session_state.amt = 0

perTokenCost = 0.0200 / 1000

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def calculate_cost(tokens: int) -> float:
    """Calculates the cost based on the number of tokens."""
    return tokens * perTokenCost

def process_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def run_qa_chain(docs, question):
    llm = OpenAI(model_name="text-davinci-003", temperature=0, max_tokens=512)
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=question)
        return response, cb.total_cost

def main():
    # Initialize session state
    initialize_session_state()

    load_dotenv()
    st.set_page_config(page_title="Ask your PDF", page_icon="💬", layout="centered")
    st.header("Ask your PDF 💬")

    # PDF processing
    def process_uploaded_pdf(file):
        try:
            text = process_pdf(file)
            num_tokens = num_tokens_from_string(text)
            st.session_state.amt = calculate_cost(num_tokens)
            return text
        except Exception as e:
            st.error("PDF processing error: " + str(e))
            logging.error("PDF processing error", exc_info=True)
            return None

    pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if pdf is not None:
        text = process_uploaded_pdf(pdf)

        if text:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=20,
                length_function=len,
            )
            docs = text_splitter.create_documents([text])
            embeddings = OpenAIEmbeddings()
            db = FAISS.from_documents(docs, embeddings)

            question = st.text_input("Ask your question about the pdf", key="question")
            if question:
                try:
                    docs = db.similarity_search(question, k=1)
                    response, total_cost = run_qa_chain(docs, question)
                    st.write(response)
                    st.session_state.amt += total_cost
                    st.write("💸 💰: $", st.session_state.amt)
                except Exception as e:
                    st.error("Question answering error: " + str(e))
                    logging.error("Question answering error", exc_info=True)

    st.markdown(hide_st_style, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
