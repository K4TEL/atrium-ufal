import argparse
import os
import configparser
from classifier import *
import time
from pathlib import Path

if __name__ == "__main__":
    # Read the configuration file
    config = configparser.ConfigParser()
    config.read('config.txt')

    # Read values from the config file
    base_dir = config.get('SETUP', 'base_dir')
    max_categ = config.getint('SETUP', 'max_categ')
    w_class = config.getint('SETUP', 'w_class')
    weight_options = ["balance", "size-prior", "priority", "no"]
    trees = config.getint('SETUP', 'trees')
    top_N = config.getint('SETUP', 'top_N')
    force = config.getboolean('SETUP', 'force')

    category_map = {
        1: {"PHOTO": config.get('CATEGORIES', 'photo').split(',')},
        2: {"DRAW": config.get('CATEGORIES', 'draw').split(',')},
        3: {"LINE": config.get('CATEGORIES', 'line').split(',')},
        4: {"TEXT": config.get('CATEGORIES', 'text').split(',')}
    }

    train_folder = config.get('INPUT', 'train_folder')
    test_file = config.get('INPUT', 'test_file')

    time_stamp = time.strftime("%Y%m%d-%H%M")

    parser = argparse.ArgumentParser(description='Page sorter based on RFC')
    parser.add_argument('-f', "--file", type=str, default=None, help="Single PNG page path")
    parser.add_argument('-d', "--directory", type=str, default=None, help="Path to folder with PNG pages")
    parser.add_argument('-tn', "--topn", type=int, default=top_N, help="Number of top result categories to consider")
    parser.add_argument('-w', "--weight", type=int, default=w_class, help='Weighting scheme by index (e.g., "balance", "size-prior", "priority", "no")')
    parser.add_argument('-t', "--tree", type=int, default=trees, help="Number of trees in the Random Forest classifier")
    parser.add_argument("--dir", help="Process whole directory", action="store_true")
    parser.add_argument("--train", help="Train the model", default=force, action="store_true")

    args = parser.parse_args()

    # Prepare directories
    cur = Path(__file__).resolve().parent
    output_dir = Path(config.get('OUTPUT', 'folder_results'))
    page_images_folder = Path(config.get('INPUT', 'folder_pages'))
    input_dir = Path(config.get('INPUT', 'folder_input')) if args.directory is None else Path(args.directory)

    if not args.train:
        data_dir = None
    else:
        data_dir = str(page_images_folder / train_folder)

    Dataset = DataWrapper(base_dir, data_dir, max_categ, category_map)
    if args.train:
        Dataset.process_page_directory()
        Dataset.load_features_dataset()
    else:
        Dataset.load_features_dataset()

    RandomForest_classifier = RFC(Dataset, base_dir, max_categ, args.tree, weight_options[args.weight], args.topn, str(output_dir))

    RandomForest_classifier.train(args.train)
    RandomForest_classifier.test("final")

    if args.file is not None:
        c, pred = RandomForest_classifier.predict_single(args.file)
        print(RandomForest_classifier.categories)
        print(c, pred)

    if args.dir:
        directory_result_output = str(
            Path(output_dir / f'tables/raw_result_{args.topn}n_{weight_options[args.weight]}_{max_categ}c_{args.tree}t.csv'))
        RandomForest_classifier.predict_directory(str(input_dir), raw=True, out_table=directory_result_output, general=True)
