import streamlit as st
import pandas as pd
import numpy as np

from inference_model import inference


# Start execution
def main():

    # Render mardown text
    f = open("./resources/intro.md", 'r')
    readme_text = st.markdown(f.read())
    f.close()

    run_app()

# Main app
def run_app():

    # Input description from user
    user_input = st.text_area("Describe your favourite wine here")

    if user_input:
        # Display recommendations as table
        st.table(inference('/Users/elmi/Projects/CloudWine/data/raw/sample.csv', user_input))


if __name__ == "__main__":
    # execute only if run as a script
    main()
