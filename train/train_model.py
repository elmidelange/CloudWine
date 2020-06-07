# Imports
import os
import argparse

from utils import Dataset, Validation
from models import TfidfTrainer


# Main function
def main(args):
    # Read training data
    dataset = Dataset(args['data_path'])
    corpus = dataset.get_corpus()

    # Train TF-IDF model
    model = TfidfTrainer()
    model.train(corpus)

    # Save run
    model.save(args['model_dir'])
    dataset.save(args['model_dir'])

    # Validate
    valid = Validation()
    x = model.get_vectors()
    df = dataset.get_df()
    valid.plot_pca(x, df['variety_region'])
    print(valid.cluster_similarities(x, df))


# Checks for valid directory
def check_dir_exists(path):
    if (os.path.isdir(path)):
        return path
    else:
        raise ValueError(path)


# Checks for valid file
def check_file_exists(path):
    if (os.path.isfile(path)):
        return path
    else:
        raise ValueError(path)


# Returns argument parser
def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [DATA_PATH] [MODE_DIR]",
        description="Train a TD-IDF model"
    )
    parser.add_argument(
        "-d", "--data_path", help="Training data file path",
            default='./data/raw/sample.csv', type=check_file_exists
    )
    parser.add_argument(
        "-m", "--model_dir", help="Model directory to save into",
            default='./model', type=check_dir_exists
    )
    # parser.add_argument(
    #     "-v", "--validation", help="Run validation",
    #         store=True
    # )
    return parser


if __name__ == "__main__":
    # execute only if run as a script
    parser = init_argparse()
    args = parser.parse_args()
    main(vars(args))
