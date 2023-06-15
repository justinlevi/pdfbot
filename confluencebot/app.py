from dotenv import load_dotenv
import streamlit as st
from confluence_qa import ConfluenceQA
import logging
import os

hide_st_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""



def main():

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    load_dotenv()
    
    CONFLUENCE_API_KEY = os.getenv('CONFLUENCE_API_KEY')

    config = {
        "persist_directory":"./chroma_db/",
        "confluence_url":"https://amazeeio.atlassian.net/wiki/",
        "username":"cara.solt@amazee.io",
        "api_key":CONFLUENCE_API_KEY,
        "space_key":"IH"
    }

    confluenceQA = ConfluenceQA(config=config)
    confluenceQA.init_embeddings()
    confluenceQA.init_models()
    confluenceQA.vector_db_confluence_docs()
    confluenceQA.retreival_qa_chain()

    st.set_page_config(page_title="Ask your Internal Handbook", page_icon="💬", layout="centered")
    st.header("Ask your Internal Handbook 💬")

    question = st.text_input("Ask your question", key="question")
    if question:
        try:
            response = confluenceQA.answer_confluence(question)
            st.write(response)
        except Exception as e:
            st.error("Question answering error: " + str(e))
            logging.error("Question answering error", exc_info=True)

    st.markdown(hide_st_style, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
