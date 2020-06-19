import streamlit as st
from PIL import Image
import numpy as np

from cloudwine.inference import inference_tfidf, inference_docvec, inference_bert
from cloudwine.utils import show_metrics_graph, download_data, update_embedding, init_data, logger

# Start execution
def main():
    # Download data from GCP bucket
    download_data()
    # Initialise the data module for the app
    data_module = init_data()
    # Determine app mode to run
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Run the app", "Data Exploration", "Model deep dive"])
    if app_mode == "Run the app":
        run_app(data_module)
    elif app_mode == "Data Exploration":
        run_data_analysis(data_module)
    elif app_mode == "Model deep dive":
        run_model_analysis()


# Main app
def run_app(data_module):
    # Render mardown text
    f = open("cloudwine/resources/intro.md", 'r')
    readme_text = st.markdown(f.read())
    f.close()
    # Filters
    # Select model to use
    embed_model = st.sidebar.selectbox("Embedding model",
        ["BERT", "Doc2Vec", "TF-IDF"])
    update_embedding(data_module, embed_model)
    st.sidebar.header("Filters")
    # Select varieties
    df = data_module.data
    variety_options = df.variety.value_counts().index.tolist()
    # dataframe['name'].value_counts()[:n].index.tolist()
    varieties = st.sidebar.multiselect( 'Wine Variety',
            variety_options)
    price_range = st.sidebar.slider("Price Range ($)", 0, int(df.price.max()), (0, int(df.price.max())), 5)
    # Apply filters
    df_subset = apply_filters(df, varieties, price_range)
    data_module.data_filtered = df_subset
    # Main page
    # Input description from user
    user_input = st.text_area("Describe your favourite wine here")
    if user_input:
        st.table(perform_inference(data_module, user_input, embed_model))
    else:
        if varieties or (price_range != (0,df.price.max())):
            st.table(df_subset[['title', 'description', 'variety', 'price']])


# Analysis app
def run_data_analysis(data_module):
    df = data_module.data
    st.image(load_image('cloudwine/resources/wine_reviews_image.jpg'), use_column_width=True)
    st.title('"Wine Reviews" Dataset Analysis')
    st.write('One this page we explore the Kaggle 2017 "Wine Reviews" dataset.')
    # Dataframe samples
    st.subheader("Sample of raw dataset")
    st.write(df.head())
    st.subheader("Features used in training")
    st.table(df[['description', 'variety', 'region_1']].head(3))
    # Description length histogram
    st.subheader("Text Length")
    hist_values = np.histogram(
    df['description'].str.len().tolist(), bins=24)[0]
    st.bar_chart(hist_values)


# Model analysis app
def run_model_analysis():
    # f = open("cloudwine/resources/model_analysis.md", 'r')
    # readme_text = st.markdown(f.read())
    # f.close()
    st.title('Model Evaluation')
    st.markdown("This project explored three different Natural Language Processing (NLP) text vectorisation techniques:")
    st.markdown("1. [Term Frequency-Inverse Document Frequency (TF-IDF)] (https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089)")
    st.markdown("2. [Doc2Vec] (https://medium.com/wisio/a-gentle-introduction-to-doc2vec-db3e8c0cce5e)")
    st.markdown("3. [Bidirectional Encoder Representations from Transformers (BERT)] (https://towardsdatascience.com/word-embedding-using-bert-in-python-dd5a86c00342)")
    st.subheader("Metric for sucess")
    st.markdown("""So how do we determine which model gives the best vector representation?
        First step is to cluster the text vectors by creating a joint label of 'variety' and 'region', as these are the biggest influencers of taste.
        As the embedding model improves and incorporates more semantic relationships between text, the the intra-cluster cosine similarity will increase
        (see [diversity metric] (https://gab41.lab41.org/recommender-systems-its-not-all-about-the-accuracy-562c7dceeaff#.5exl13wqv).""")
    st.image(load_image('cloudwine/resources/metric_figure.png'), use_column_width=True)
    st.subheader('Experimental Results')
    st.write("""The BERT embedding outperformed the baseline TF-IDF model by over 100%. To see the different models in action, go to 'Run the App'
        in the sidebar and select a model type in the dropdown.""")
    show_metrics_graph()

# Returns dataframe subset with filters applied
def apply_filters(df, varieties, price_range):
    df_subset = df.copy()
    # Varieties selection
    if varieties:
        df_subset = df_subset[df_subset['variety'].isin(varieties)]
    # Price range selection
    df_subset = df_subset[(df_subset['price']>price_range[0]) & (df_subset['price']<price_range[1])]
    return df_subset


# @st.cache
def perform_inference(data_module, user_input, embed_model):
    # Display recommendations as table
    if embed_model == 'BERT':
        df_recommend = inference_bert(data_module, user_input)
    elif embed_model == "Doc2Vec":
        df_recommend = inference_docvec(data_module, user_input)
    elif embed_model == "TF-IDF":
        df_recommend = inference_tfidf(data_module, user_input)
    return df_recommend[['title', 'description', 'variety', 'price', 'similarity']]


@st.cache
def load_image(path):
	im =Image.open(path)
	return im



if __name__ == "__main__":
    # execute only if run as a script
    main()
