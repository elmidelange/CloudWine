import streamlit as st
import pandas as pd
import numpy as np

from cloudwine.inference import inference_tfidf, inference_docvec, inference_bert
from cloudwine.utils import show_pca

data_file = 'cloudwine/data/data.csv'
model_dir = 'cloudwine/models/'

# Start execution
def main():
    # Render mardown text
    f = open("cloudwine/resources/intro.md", 'r')
    readme_text = st.markdown(f.read())
    f.close()

    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Run the app", "Model deep dive"])
    if app_mode == "Run the app":
        # readme_text.empty()
        run_app()
    elif app_mode == "Model deep dive":
        readme_text.empty()
        run_analysis()


# Main app
def run_app():
    # Input description from user
    user_input = st.text_area("Describe your favourite wine here")

    if user_input:
        # Display recommendations as table
        st.table(inference_bert(data_file, model_dir, user_input))


# Analysis app
def run_analysis():
    embed_model = st.sidebar.selectbox("Embedding model",
        ["TF-IDF", "Doc2Vec", "BERT"])

    show_pca(embed_model, data_file, model_dir)


if __name__ == "__main__":
    # execute only if run as a script
    main()
