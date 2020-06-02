import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Main function
def main(args):
    # Run inference and return dataframe with ordered recommendations
    df = inference(args['data_dir'], args['text'])


# Loads TF-IDF model and inferences input string
def inference(data_dir, text):
    # Load trained model pickles
    tf = pickle.load(open("./models/tfidf_transform.pkl", 'rb'))
    x = pickle.load(open("./models/tfidf_matrix.pkl", 'rb'))

    # Create new tfidfVectorizer with trained vocabulary
    tf_new = TfidfVectorizer(analyzer='word', ngram_range=(1,2), stop_words = "english", lowercase = True,
                          max_features = 500000, vocabulary = tf.vocabulary_)

    # Convert text to vector representation
    x_new = tf_new.fit_transform([text])

    # Calculate cosine similarities to trained text
    cosine_similarities = linear_kernel(x_new, x).flatten()

    # Get the index to the top 3 similar texts
    related_docs_indices = cosine_similarities.argsort()[:-4:-1]

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
    return df_subset[['title', 'description']]



# Returns argument parser
def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Inference a TD-IDF model"
    )
    parser.add_argument(
        "-d", "--data_dir", help="File path to the training data",
            default='../data/raw/sample.csv'
    )
    parser.add_argument(
        "-t", "--text", help="Text to inference"
    )
    return parser


if __name__ == "__main__":
    # execute only if run as a script
    parser = init_argparse()
    argsNamespace = parser.parse_args()
    main(vars(argsNamespace))
