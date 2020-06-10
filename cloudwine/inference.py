import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize

import streamlit as st


# Loads TF-IDF model and inferences input string
def inference_tfidf(data_dir, model_dir, text):
    print('TF-IDF Inference')
    # Load trained model pickles
    tf = pickle.load(open(model_dir + "tfidf_model.pkl", 'rb'))
    x = pickle.load(open(model_dir + "tfidf_vectors.pkl", 'rb'))

    # Create new tfidfVectorizer with trained vocabulary
    tf_new = TfidfVectorizer(analyzer='word', ngram_range=(1,2), stop_words = "english", lowercase = True,
                          max_features = 500000, vocabulary = tf.vocabulary_)

    # Convert text to vector representation
    x_new = tf_new.fit_transform([text])

    # Calculate cosine similarities to trained text
    cosine_similarities = cosine_similarity(x_new, x).flatten()

    # Get the index to the top 3 similar texts
    related_docs_indices = cosine_similarities.argsort()[:-6:-1]

    # Find the wine titles for given index
    df = pd.read_csv(data_dir)

    print('\n')
    print('Original Text: ')
    print('text')
    print('\n')

    for i in range(len(related_docs_indices)):
        print('Recommendation ', i+1)
        print(df.iloc[related_docs_indices[i]]['description'])
        print('\n')

    df_subset = df.loc[related_docs_indices].reset_index(drop=True)
    return df_subset[['title', 'description', 'variety']]


# Loads TF-IDF model and inferences input string
def inference_docvec(data_dir, model_dir, text):
    print('Doc2Vec Inference')
    # Load trained model pickles
    model = pickle.load(open(model_dir + "doc2vec_model.pkl", 'rb'))
    x = pickle.load(open(model_dir + "doc2vec_vectors.pkl", 'rb'))

    words=word_tokenize(text.lower())
    x_new = model.infer_vector(words)

    # Calculate cosine similarities to trained text
    cosine_similarities = cosine_similarity([x_new], x).flatten()

    # Get the index to the top 3 similar texts
    related_docs_indices = cosine_similarities.argsort()[:-6:-1]

    # Find the wine titles for given index
    df = pd.read_csv(data_dir)

    print('\n')
    print('Original Text: ')
    print('text')
    print('\n')

    for i in range(len(related_docs_indices)):
        print('Recommendation ', i+1)
        print(df.iloc[related_docs_indices[i]]['description'])
        print('\n')

    df_subset = df.loc[related_docs_indices].reset_index(drop=True)
    df_subset['similarity'] = cosine_similarities[related_docs_indices]
    return df_subset[['title', 'description', 'variety', 'similarity']]


# Loads BERT embeddings and inferences input string
def inference_bert(data_dir, model_dir, text):
    print('BERT Inference')
    # Load trained model pickles
    model = pickle.load(open(model_dir + "bert_model.pkl", 'rb'))
    x = pickle.load(open(model_dir + "bert_vectors.pkl", 'rb'))
    # Inference new text
    x_new = model.encode([text])

    # Calculate cosine similarities to trained text
    # cosine_similarities = linear_kernel(x_new, x).flatten()
    cosine_similarities = cosine_similarity(x_new, x).flatten()

    # Get the index to the top 3 similar texts
    related_docs_indices = cosine_similarities.argsort()[:-6:-1]

    # Find the wine titles for given index
    df = pd.read_csv(data_dir)

    print('\n')
    print('Original Text: ')
    print('text')
    print('\n')

    for i in range(len(related_docs_indices)):
        print('Recommendation ', i+1)
        print(df.iloc[related_docs_indices[i]]['description'])
        print('\n')

    df_subset = df.loc[related_docs_indices].reset_index(drop=True)
    df_subset['similarity'] = cosine_similarities[related_docs_indices]
    return df_subset[['title', 'description', 'variety', 'similarity']]
