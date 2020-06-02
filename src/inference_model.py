import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def main(args):
    print(args)
    tf = pickle.load(open("/Users/elmi/Projects/CloudWine/src/models/tfidf_transform.pkl", 'rb'))
    x = pickle.load(open("/Users/elmi/Projects/CloudWine/src/models/tfidf_matrix.pkl", 'rb'))

    # Create new tfidfVectorizer with trained vocabulary
    tf_new = TfidfVectorizer(analyzer='word', ngram_range=(1,2), stop_words = "english", lowercase = True,
                          max_features = 500000, vocabulary = tf.vocabulary_)

    x_new = tf_new.fit_transform([args['text']])

    cosine_similarities = linear_kernel(x_new, x).flatten()

    related_docs_indices = cosine_similarities.argsort()[:-4:-1]

    df = pd.read_csv(args['data_dir'])

    print('\n')
    print('Original Text: ')
    print(args['text'])
    print('\n')

    for i in range(len(related_docs_indices)):
        print('Recommendation ', i+1)
        print(df.iloc[related_docs_indices[i]]['description'])
        print('\n')



def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Inference a TD-IDF model"
    )
    parser.add_argument(
        "-d", "--data_dir", help="File path to the training data",
            default='/Users/elmi/Projects/CloudWine/data/raw/sample.csv'
    )
    parser.add_argument(
        "-t", "--text", help="Text to inference"
    )
    return parser


if __name__ == "__main__":
    # execute only if run as a script
    parser = init_argparse()
    args = parser.parse_args()
    main(vars(args))
