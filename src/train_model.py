import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def main(args):
    print(args)
    df = pd.read_csv(args['data_dir'])
    corpus = df['description'].tolist()

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), stop_words = "english", lowercase = True, max_features = 500000)
    tf_transformer = tf.fit(corpus)

    with open('/Users/elmi/Projects/CloudWine/src/models/tfidf_transform.pkl', "wb") as pickleFile:
        pickle.dump(tf_transformer, pickleFile)

    X = tf.fit_transform(corpus)
    with open('/Users/elmi/Projects/CloudWine/src/models/tfidf_matrix.pkl', "wb") as pickleFile:
        pickle.dump(X, pickleFile)

    print('Saved Pickle')
    

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Train a TD-IDF model"
    )
    parser.add_argument(
        "-d", "--data_dir", help="File path to the training data",
            default='/Users/elmi/Projects/CloudWine/data/raw/sample.csv'
    )
    return parser


if __name__ == "__main__":
    # execute only if run as a script
    parser = init_argparse()
    args = parser.parse_args()
    main(vars(args))
