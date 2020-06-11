import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize

import streamlit as st


# Loads TF-IDF model and inferences input string
def inference_tfidf(df, text):
    print('TF-IDF Inference')
    # Load trained model pickles
    tf = pickle.load(open("cloudwine/models/tfidf_model.pkl", 'rb'))
    x = pickle.load(open("cloudwine/models/tfidf_vectors.pkl", 'rb'))
    # Get filtered embeddings
    x = x[df.index.tolist()]

    # Create new tfidfVectorizer with trained vocabulary
    tf_new = TfidfVectorizer(analyzer='word', ngram_range=(1,2), stop_words = "english", lowercase = True,
                          max_features = 500000, vocabulary = tf.vocabulary_)

    # Convert text to vector representation
    x_new = tf_new.fit_transform([text])

    # Calculate cosine similarities to trained text
    cosine_similarities = cosine_similarity(x_new, x).flatten()

    # Get the index to the top 3 similar texts
    related_docs_indices = cosine_similarities.argsort()[:-6:-1]

    df_subset = df.reset_index().loc[related_docs_indices].reset_index(drop=True)
    df_subset['similarity'] = cosine_similarities[related_docs_indices]
    return df_subset


# Loads TF-IDF model and inferences input string
def inference_docvec(df, text):
    print('Doc2Vec Inference')
    # Load trained model pickles
    model = pickle.load(open("cloudwine/models/doc2vec_model.pkl", 'rb'))
    x = pickle.load(open("cloudwine/models/doc2vec_vectors.pkl", 'rb'))
    # Get filtered embeddings
    x = x[df.index.tolist()]

    words=word_tokenize(text.lower())
    x_new = model.infer_vector(words)

    # Calculate cosine similarities to trained text
    cosine_similarities = cosine_similarity([x_new], x).flatten()

    # Get the index to the top 3 similar texts
    related_docs_indices = cosine_similarities.argsort()[:-6:-1]

    df_subset = df.reset_index().loc[related_docs_indices].reset_index(drop=True)
    df_subset['similarity'] = cosine_similarities[related_docs_indices]
    return df_subset


# Loads BERT embeddings and inferences input string
def inference_bert(df, text):
    print('BERT Inference')
    # Load trained model pickles
    model = pickle.load(open("cloudwine/models/bert_model.pkl", 'rb'))
    x = pickle.load(open("cloudwine/models/bert_vectors.pkl", 'rb'))
    # Get filtered embeddings
    x = x[df.index.tolist()]
    # Inference new text
    x_new = model.encode([text])

    # Calculate cosine similarities to trained text
    # cosine_similarities = linear_kernel(x_new, x).flatten()
    cosine_similarities = cosine_similarity(x_new, x).flatten()

    # Get the index to the top 3 similar texts
    related_docs_indices = cosine_similarities.argsort()[:-6:-1]

    df_subset = df.reset_index().loc[related_docs_indices].reset_index(drop=True)
    df_subset['similarity'] = cosine_similarities[related_docs_indices]
    return df_subset
