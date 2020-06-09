# Imports
import os
import argparse
import yaml

from utils import Dataset, Validation, logger
from models import TfidfTrainer, DocVecTrainer


# Main function
def main(config):

    # Initialise the model type and arguments
    model, args = init_trainer(config)

    print(args)

    # Read training data
    dataset = Dataset(args['data_path'])
    corpus = dataset.get_corpus()

    # Train TF-IDF model
    # model = TfidfTrainer()
    model.train(corpus)

    if args['save'] == True:
        # Save run
        model.save(args['model_dir'])
        dataset.save(args['model_dir'])

    if args['validation'] == True:
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
        exit()
    # Check for parameters
    if not all(key in config for key in ('model', 'args')):
        logger.error("Missing 'model' or 'args' parameter in config yaml " + path)
        raise ValueError(path)
        exit()
    model = config['model']
    args = config['args']
    if not all(key in args for key in ('data_path', 'model_dir', 'validation', 'save')):
        logger.error("Missing 'data_path' or 'model_dir' args in config yaml " + path)
        raise Exception("Config yaml missing one of 'data_path', 'model_dir', 'validation', 'save'")
        sys.exit(1)
    return config


# Returns a model trainer and arguments
def init_trainer(config):
    model_type = config['model']
    args = config['args']
    if model_type == 'tfidf':
        model = TfidfTrainer()
    elif model_type == 'doc2vec':
        model = DocVecTrainer(args)
        pass
    elif model_type == 'bert':
        # TODO: Add transformers model check_dir_exists
        pass
    else:
        raise ValueError(model_type)

    return model, args


# Returns argument parser
def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [CONFIG_YAML]",
        description="Train a TD-IDF model"
    )
    parser.add_argument(
        "-y", "--config_yaml", help="Training config yaml file",
            default='./args.yaml', type=check_config
    )
    return parser


if __name__ == "__main__":
    # execute only if run as a script
    parser = init_argparse()
    config_ns = parser.parse_args()
    config = vars(config_ns)['config_yaml']
    main(config)
