import argparse
import os
from dotenv import load_dotenv
from pathlib import Path
import json
import pandas as pd

from classifier import *


if __name__ == "__main__":
    base_dir = "/lnet/work/people/lutsai/pythonProject"

    top_N = 3
    trees = 100

    max_categ = 1350

    input_dir = f'pages/train_{"final" if max_categ % 50 == 0 else "balanced"}'

    large_dataset = f"{base_dir}/dataset/data_tf_1350c.h5" if max_categ % 100 == 0 else f"{base_dir}/dataset/data_tb_1301c.h5"
    large_dataset_labels = f"{base_dir}/dataset/labels_tf_1350c.h5" if max_categ % 100 == 0 else f"{base_dir}/dataset/labels_tb_1301c.h5"
    large_dataset_name = "dataset_1350c" if max_categ % 100 == 0 else "dataset_1301c"

    # weighting scheme for the Random Forest classifier
    w_class = 0
    weight_options = ["balance", "size-prior", "priority", "no"]

    # weighting priorities
    category_map = {
        1: {
            "PHOTO": ["PHOTO", "PHOTO_L"]
        },
        2: {
            "DRAW": ["DRAW", "DRAW_L"]
        },
        3: {
            "LINE": ["LINE_HW", "LINE_P", "LINE_T"]
        },
        4: {
            "TEXT": ["TEXT", "TEXT_HW", "TEXT_P", "TEXT_T"]
        }
    }

    force = False  # training

    test_dir = 'test_pages'
    test_file = "MTX202210023c_page_21.png"

    parser = argparse.ArgumentParser(description='Page sorter based on RFC')
    parser.add_argument('-f', "--file", type=str, default=test_file, help="Single PNG page path")
    parser.add_argument('-d', "--directory", type=str, default=input_dir, help="Path to folder with PNG pages")
    parser.add_argument('-tn', "--topn", type=int, default=top_N, help="Number of top result categories to consider")
    parser.add_argument('-w', "--weight", type=int, default=w_class, help='By index from "balance"(D), "size-prior", "priority", and "no" options')
    parser.add_argument('-t', "--tree", type=int, default=trees, help="Number of trees in the Random Forest classifier")
    parser.add_argument("--dir", help="Process whole directory", action="store_true")
    parser.add_argument("--train", help="Process PDF files into layouts", default=force, action="store_true")

    args = parser.parse_args()

    load_dotenv()

    cur = Path.cwd() #  directory with this script
    # locally creating new directory pathes instead of .env variables loaded with mistakes
    page_images_folder = Path(os.environ.get('FOLDER_PAGES', cur / "pages"))  # weights a lot, contains grayscale page images, detected layouts and line plots
    output_dir = Path(os.environ.get('FOLDER_RESULTS', cur / "results"))  # weights a little, page and pdf level summaries with classified categories in csv format

    Dataset = DataWrapper(base_dir,
                          args.directory,
                          max_categ,
                          category_map)
    Dataset.process_page_directory()
    Dataset.load_features_dataset()
    # Dataset.load_features_dataset(large_dataset, large_dataset_labels, large_dataset_name)

    RandomForest_classifier = RFC(Dataset,
                                  base_dir,
                                  max_categ,
                                  args.tree,
                                  weight_options[args.weight],
                                  args.topn,
                                  str(Path(cur / output_dir)))

    RandomForest_classifier.train(args.train)
    RandomForest_classifier.test("final")

    rfc_params = RandomForest_classifier.model.get_params()

    print(rfc_params)

    directory_result_output = Path(cur / f'{output_dir}/tables/raw_result_{args.topn}n_{args.weight[0]}_{max_categ}c_{args.tree}t.csv')

    c, pred = RandomForest_classifier.predict_single(str(Path(cur / f"{test_dir}/{args.file}")))
    print(c, pred)

    if args.dir:
        RandomForest_classifier.predict_directory(os.path.join(base_dir, args.input_dir), 5, str(directory_result_output), True)



