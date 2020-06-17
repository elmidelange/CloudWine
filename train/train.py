# Imports
import os
import sys
import argparse
import yaml

from utils import Dataset, Validation, logger
from models import TfidfTrainer, DocVecTrainer, BertTrainer

from datetime import datetime

import pickle


# Main function
def main(config):
    # Initialise the model type and arguments
    model, args = init_trainer(config)
    logger.info(args)
    # Read training data
    dataset = Dataset(args['data_path'], args)
    corpus = dataset.get_corpus()
    # Train model
    model.train(corpus)
    # Save model
    if args['save_model'] == True:
        # Save run
        logger.info('Saving Model')
        model.save(args['model_dir'])
        dataset.save(args['data_path'])
    # Perform validation
    valid = Validation()
    x = model.get_vectors()
    df = dataset.get_df()
    # valid.plot_pca(x, df['variety_region'])
    results = valid.cluster_similarities(x, df)
    logger.info(results)
    if args['save_validation'] == True:
        logger.info('Saving Validation')
        config['output'] = results['similarity']
        with open(args['validation_dir'] +  '{}.pkl'.format(datetime.now()), "wb") as pickleFile:
            pickle.dump(config, pickleFile)


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


# Checks for valid arguments
def check_config(path):
    # Check file exists
    if not (os.path.isfile(path)):
        ValueError(path)
    # Read file into json
    try:
        with open(path) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        logger.info(e)
        logger.error('Invalid parameters yaml ' + path)
        raise ValueError(path)
    # Check for model and args keys
    if not all(key in config for key in ('model', 'args')):
        logger.error("Missing 'model' or 'args' parameter in config yaml " + path)
        raise ValueError(path)
    model = config['model']
    args = config['args']
    # Check for general args
    if not all(key in args for key in ('data_path', 'save_model', 'model_dir', 'save_validation', 'validation_dir')):
        logger.error("Config yaml missing model argument(s) " + path)
        raise Exception("Config yaml missing model arguments")
    # Check for NLP processing args
    if not all(key in args for key in ('lowercase', 'remove_punctuation', 'remove_stopwords', 'lemmatize')):
        logger.error("Missing NLP processing parameters " + path)
        raise Exception("Config yaml missing NLP processing parameter(s)")
    return config


# Returns a model trainer and arguments
def init_trainer(config):
    model_type = config['model']
    args = config['args']
    if model_type == 'tfidf':
        model = TfidfTrainer()
    elif model_type == 'doc2vec':
        model = DocVecTrainer()
    elif model_type == 'bert':
        model = BertTrainer()
    else:
        raise ValueError(model_type)

    return model, args


# Returns argument parser
def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [CONFIG_YAML]",
        description="Train a text embedding model"
    )
    parser.add_argument(
        "-y", "--config_yaml", help="Training config yaml file",
            type=check_config
    )
    return parser


if __name__ == "__main__":
    # execute only if run as a script
    parser = init_argparse()
    config_ns = parser.parse_args()
    config = vars(config_ns)['config_yaml']
    main(config)
