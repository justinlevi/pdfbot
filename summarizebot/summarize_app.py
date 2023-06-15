from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.self_hosted import Self
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.docstore.document import Document
from typing import List
import tiktoken
import logging

hide_st_style = """
  <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
  </style>
"""

embeddings = OpenAIEmbeddings()
llm_summarize = OpenAI(model_name="text-davinci-003", temperature=0, max_tokens=1000)
llm_qa = OpenAI(model_name="text-davinci-003", temperature=0, max_tokens=512)

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

def get_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def run_summarize_chain(docs: List[Document]):
    chain = load_summarize_chain(llm_summarize, chain_type="map_reduce")
    with get_openai_callback() as cb:
        response = chain.run(docs)
        return response, cb.total_cost

def run_qa_chain(docs: List[Document], question):
    chain = load_qa_chain(llm_qa, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=question, )
        return response, cb.total_cost

def process_uploaded_pdf(file):
    logging.info("Processing PDF...")
    try:
        with st.spinner('Processing...'):
            text = get_text_from_pdf(file)
            num_tokens = num_tokens_from_string(text)
            cost = calculate_cost(num_tokens)

            if text:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=50,
                    length_function=len,
                )
                docs = text_splitter.create_documents([text])
                summary, cost = run_summarize_chain(docs)
                db = Chroma.from_documents(docs, embeddings)
            return summary, db, cost
    except Exception as e:
        logging.error("PDF processing error", exc_info=True)
        raise e

def process_question(db: Chroma, question: str):
    logging.info("Processing Question...")
    try:
        with st.spinner('Processing...'):
            docs = db.similarity_search(question, k=1)
            response, total_cost = run_qa_chain(docs, question)
        return response, total_cost
    except Exception as e:
        logging.error("Summarizer error", exc_info=True)
        raise e

def main():
    # Initialize session state
    initialize_session_state()

    load_dotenv()
    st.set_page_config(page_title="Summarize your PDF", page_icon="📖", layout="centered")
    st.header("Summarize your PDF 📖")

    pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if pdf is not None:
        try:
            summary, db, cost = process_uploaded_pdf(pdf)
            st.session_state.amt += cost

            st.write(summary)
            st.write("💸 💰: $", st.session_state.amt)

            question = st.text_input("Ask a question about the pdf", key="question")
            if question:
                try:
                    response, total_cost = process_question(db, question)
                    st.write(response)
                    st.session_state.amt += total_cost
                    st.write("💸 💰: $", st.session_state.amt)
                except Exception as e:
                    st.error("Summarizer error: " + str(e))
                    logging.error("Summarizer error", exc_info=True)

        except Exception as e:
            st.error("PDF processing error: " + str(e))

    st.markdown(hide_st_style, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
