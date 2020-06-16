import streamlit as st

from cloudwine.inference import inference_tfidf, inference_docvec, inference_bert
from cloudwine.utils import show_metrics_graph, download_data, update_embedding, init_data

# Start execution
def main():
    # Download data from GCP bucket
    data_module = download_data()
    # Determine app mode to run
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Run the app", "Model deep dive"])
    if app_mode == "Run the app":
        run_app()
    elif app_mode == "Model deep dive":
        run_analysis()


# Main app
def run_app():
    # Initialise the data module for the app
    data_module = init_data()
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
        # Display recommendations as table
        if embed_model == 'BERT':
            df_recommend = inference_bert(data_module, user_input)
        elif embed_model == "Doc2Vec":
            df_recommend = inference_docvec(data_module, user_input)
        elif embed_model == "TF-IDF":
            df_recommend = inference_tfidf(data_module, user_input)
        st.table(df_recommend[['title', 'description', 'variety', 'price', 'similarity']])
    else:
        if varieties or (price_range != (0,df.price.max())):
            st.table(df_subset[['title', 'description', 'variety', 'price']])


# Analysis app
def run_analysis():
    st.title('Model Evaluation')
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


if __name__ == "__main__":
    # execute only if run as a script
    main()
