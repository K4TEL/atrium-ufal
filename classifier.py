import os
import pickle
from collections import Counter
import random
from pathlib import Path
import argparse
# from dotenv import load_dotenv

import time

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler

import multiprocessing as mp

import sys
import torch
import clip
from PIL import Image
import csv
import os
import string


# get list of all files in the folder and nested folders by file format
def directory_scraper(folder_path: Path, file_format: str = "png", file_list: list = None) -> list[str]:
    if file_list is None:
        file_list = []
    file_list += list(folder_path.rglob(f"*.{file_format}"))
    print(f"[ {file_format.upper()} ] \tFrom directory {folder_path} collected {len(file_list)} {file_format} files")
    return file_list


def append_to_csv(df, filepath):
    """Append DataFrame to CSV file, or create a new file if it doesn't exist."""
    if not os.path.exists(filepath):
        df.to_csv(filepath, index=False, sep=",")
    else:
        df.to_csv(filepath, mode="a", header=False, index=False, sep=",")


class CLIP:
    def __init__(self, base_path: str, max_category_samples: int,
                 top_N: int, model_name: str, device, out: str = None):
        self.model_dir = f"{base_path}/model"
        self.data_dir = f"{base_path}/dataset"
        self.upper_category_limit = max_category_samples
        self.top_N = top_N

        self.seed = 42

        # self.model_name = f"model_{dataset.folder_abbr}_{self.weighted_classes[0]}_{self.upper_category_limit}c_{self.N_trees}t.pkl"

        # self.categories = sorted(os.listdir(self.page_dir))
        # self.categories = dataset.categories
        # self.label_general = dataset.label_general
        # self.general_categories = dataset.general_categories

        # self.categories =['DRAW', 'DRAW_L', 'LINE_HW', 'LINE_P', 'LINE_T', 'PHOTO', 'PHOTO_L', 'TEXT', 'TEXT_HW', 'TEXT_P', 'TEXT_T']
        self.categories = ['DRAW', 'PHOTO', 'TEXT', 'LINE']
        print(f"Category loaded from memory: {self.categories}")

        self.label_priority = {i: 0 for i, v in enumerate(self.categories)}
        self.label_general = {i: v for i, v in enumerate(self.categories)}
        self.general_categories = []
        for proir, label_dict in category_map.items():
            for gen_label, labels_list in label_dict.items():
                self.general_categories.append(gen_label)
                for i, label in enumerate(labels_list):
                    label_id = self.categories.index(label)
                    # self.label_priority[label_id] = (5 - proir) * 5 + i  # higher prior values for more important categs
                    self.label_priority[label_id] = proir + i / 10
                    self.label_general[label_id] = gen_label

        print(f"Category mapping: {self.label_general}")
        print(f"Category priority: {self.label_priority}")

        # print(f"Category input directories found: {self.categories}")

        # self.model = None
        # self.scaler = MinMaxScaler(feature_range=(0, 1))

        # self.data_train = dataset.data["train"]
        # self.data_test = dataset.data["test"]

        self.output_dir = "results" if out is None else out

        self.download_root = '/lnet/work/projects/atrium/cache/clip'

        self.model, self.preprocess = clip.load(model_name, device=device,
                                       download_root=self.download_root)  # 86 million parameters

        # Load categories from external TSV file
        # categories_tsv = "page_categories.tsv"  # Replace with your TSV file path
        categories_tsv = "page_general_categories.tsv"  # Replace with your TSV file path
        categories = load_categories(categories_tsv)

        # Prepare text descriptions for the CLIP model
        self.text_inputs = torch.cat([clip.tokenize(description) for _, description in categories]).to(device)
        self.model_name = model_name
        self.device = device


    def generalize(self, image_scores: np.array):
        general_scores = np.zeros(len(self.general_categories))
        for i, cat in enumerate(image_scores):
            general_label = self.label_general[i]
            general_scores[self.general_categories.index(general_label)] += cat
        return general_scores


    def top_N_prediction(self, image_data: torch.Tensor, N: int):

        # print(image_data)

        image_data = image_data.to(self.device)


        # Perform inference
        with torch.no_grad():
            image_features = self.model.encode_image(image_data)
            text_features = self.model.encode_text(self.text_inputs)

            # Compute similarity scores
            logits_per_image, _ = self.model(image_data, self.text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # Display results
        # print(f"\nResults for image: {image_path}")
        # for label, prob in zip(self.categories, probs[0]):
        #     print(f"{label}: {prob * 100:.2f}%")

        pred_scores = probs[0]
        best_n = np.argsort(pred_scores)[-N:]
        best_n_scores = np.sort(pred_scores)[-N:]

        best_n_scores_normal = best_n_scores / sum(best_n_scores)

        score_var = np.var(best_n_scores_normal)

        return -np.sort(-best_n_scores_normal), best_n, pred_scores, np.round(score_var, 5), self.generalize(pred_scores)


    def prediction(self, image_data: torch.Tensor) -> (np.array, int):

        image_data = image_data.to(self.device)

        # Perform inference
        with torch.no_grad():
            image_features = self.model.encode_image(image_data)
            text_features = self.model.encode_text(self.text_inputs)

            # Compute similarity scores
            logits_per_image, _ = self.model(image_data, self.text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # Display results
        # print(f"\nResults for image: {image_path}")
        # for label, prob in zip(self.categories, probs[0]):
        #     print(f"{label}: {prob * 100:.2f}%")

        pred_scores = probs[0]
        return pred_scores, np.argmax(pred_scores)

    def predict_single(self, image_file: str, N: int = None):
        if self.model is None:
            # load
            with open(f"{self.model_dir}/{self.model_name}", 'rb') as f:
                self.model = pickle.load(f)

            print(f"Model loaded from {self.model_name}")

        # feature = img_to_feature(image_file)
        image = Image.open(image_file)
        image_input = self.preprocess(image).unsqueeze(0).to(device)

        category_distrib = self.prediction(image_input)
        return category_distrib, self.categories[np.argmax(category_distrib)]

    def predict_directory(self, folder_path: str, batch_size: int = None, out_table: str = None, raw: bool = False):
        # if self.model is None:
        #     # load
        #     with open(f"{self.model_dir}/{self.model_name}", 'rb') as f:
        #         self.model = pickle.load(f)
        #
        #     print(f"Model loaded from {self.model_name}")

        images = directory_scraper(Path(folder_path), "png")

        print(f"Predicting {len(images)} images from {folder_path}")

        # res_table, raw_table = [], []

        batch_size = mp.cpu_count() if batch_size is None else batch_size

        for batch_start in range(0, len(images), batch_size):
            batch_images = images[batch_start:batch_start + batch_size]

            # data_ids = np.arange(batch_start, batch_start + batch_size)

            res_table, raw_table = [], []

            best_n_scores_normal, best_n, raw_scores, vars, gens = [], [], [], [], []
            data_ids = []
            for img in batch_images:
                image = Image.open(img)
                # print(image)
                image_input = self.preprocess(image).unsqueeze(0).to(device)

                image = Image.open(img)
                image_input = self.preprocess(image).unsqueeze(0).to(device)

                # if image_input is not None:
                #     pred_scores, pred_label = self.prediction(image_input)
                #     # predictions.append(pred_label)
                #
                #     score_norm, img_best_n, pred_scores, varian, gen = self.top_N_prediction(image_input, self.top_N)
                #     best_n_scores_normal.append(score_norm)
                #     best_n.append(img_best_n)
                #     raw_scores.append(pred_scores)
                #     vars.append(varian)
                #     gens.append(gen)
                #
                #     data_ids.append(Path(img).stem)

                try:
                    image = Image.open(img)
                    image_input = self.preprocess(image).unsqueeze(0).to(device)

                    if image_input is not None:
                        pred_scores, pred_label = self.prediction(image_input)
                        # predictions.append(pred_label)

                        score_norm, img_best_n, pred_scores, varian, gen = self.top_N_prediction(image_input, self.top_N)
                        best_n_scores_normal.append(score_norm)
                        best_n.append(img_best_n)
                        raw_scores.append(pred_scores)
                        vars.append(varian)
                        gens.append(gen)

                        data_ids.append(Path(img).stem)

                except Exception as e:
                    print(f"Error processing file {img}: {e}")

            for i, image_f in enumerate(batch_images):
                categ_labels = [self.categories[c] for c in best_n[i]]
                categ_scores = np.round(best_n_scores_normal[i], 3)
                # image_filename = data_ids[i].replace("_", "-").replace("-page-", "-")
                img_file, img_page = data_ids[i].split("-")
                print(img_file, img_page)
                res_table.append([img_file, img_page] + categ_labels + categ_scores.tolist() + [vars[i]] + np.round(gens[i], 3).tolist())

                if raw:
                    raw_row = list(raw_scores[i]) + [img_file, img_page] + np.round(gens[i], 3).tolist()
                    raw_table.append(raw_row)

            # print(res_table)

            # Append to top-N output
            if out_table is not None:
                columns = ["FILE", "PAGE"] + \
                          [f"CLASS-{i + 1}" for i in range(self.top_N)] + \
                          [f"SCORE-{i + 1}" for i in range(self.top_N)] + ["CERTAIN"] + self.general_categories
                top_n_df = pd.DataFrame(res_table, columns=columns)

                out_table = f'{self.output_dir}/tables/result_{self.top_N}n_{self.upper_category_limit}c.csv'
                append_to_csv(top_n_df, out_table)

            # Append to raw output
            if raw:
                raw_columns = self.categories + ["FILE", "PAGE"] + self.general_categories
                raw_df = pd.DataFrame(raw_table, columns=raw_columns)
                raw_output_path = f'{self.output_dir}/tables/raw_result_{self.top_N}n_{self.upper_category_limit}c.csv'
                append_to_csv(raw_df, raw_output_path)

        print("Processing completed.")


# Function to load categories from a TSV file
def load_categories(tsv_file):
    categories = []
    try:
        with open(tsv_file, "r") as file:
            reader = csv.DictReader(file, delimiter="\t")
            for row in reader:
                categories.append((row["label"], row["description"]))
    except Exception as e:
        print(f"Error reading categories file: {e}")
        sys.exit(1)

    print(categories)
    return categories


if __name__ == "__main__":
    # Automatically select the device based on availability
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use CUDA-enabled GPU on systems with NVIDIA GPUs
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Use MPS on macOS with Apple Silicon
    else:
        device = torch.device("cpu")  # Fallback to CPU if no GPU is available

    print(f"Using device: {device}")

    # base_dir = "/lnet/work/projects/atrium/clip"

    # model_name = "ViT-B/32"  # Vision Transformer (ViT) model with 32x32 patches
    model_name = "ViT-L/14@336px"  # Vision Transformer (ViT) model with 14 layers and 336x336 patches

    top_N = 3
    trees = 333

    max_categ = 100

    # weighting priorities
    # category_map = {
    #     1: {
    #         "PHOTO": ["PHOTO", "PHOTO_L"]
    #     },
    #     2: {
    #         "DRAW": ["DRAW", "DRAW_L"]
    #     },
    #     3: {
    #         "LINE": ["LINE_HW", "LINE_P", "LINE_T"]
    #     },
    #     4: {
    #         "TEXT": ["TEXT", "TEXT_HW", "TEXT_P", "TEXT_T"]
    #     }
    # }

    category_map = {
        1: {
            "PHOTO": ["PHOTO"]
        },
        2: {
            "DRAW": ["DRAW"]
        },
        3: {
            "LINE": ["LINE"]
        },
        4: {
            "TEXT": ["TEXT"]
        }
    }

    force = False  # training

    test_dir = 'testing'
    test_file = "T_MTX195602489-12.png"

    parser = argparse.ArgumentParser(description='Page sorter based on RFC')
    parser.add_argument('-f', "--file", type=str, default=None, help="Single PNG page path")
    parser.add_argument('-d', "--directory", type=str, default=None, help="Path to folder with PNG pages")
    parser.add_argument('-tn', "--topn", type=int, default=top_N, help="Number of top result categories to consider")
    # parser.add_argument('-w', "--weight", type=int, default=w_class, help='By index from "balance"(D), "size-prior", "priority", and "no" options')
    # parser.add_argument('-t', "--tree", type=int, default=trees, help="Number of trees in the Random Forest classifier")
    parser.add_argument("--dir", help="Process whole directory", action="store_true")
    parser.add_argument("--train", help="Process PDF files into layouts", default=force, action="store_true")

    args = parser.parse_args()

    # load_dotenv()

    cur = Path.cwd() #  directory with this script
    # locally creating new directory pathes instead of .env variables loaded with mistakes
    output_dir = Path(os.environ.get('FOLDER_RESULTS', cur / "results"))
    page_images_folder = Path(os.environ.get('FOLDER_PAGES', cur / "test-images/pages"))
    input_dir = Path(os.environ.get('FOLDER_INPUT', page_images_folder / test_dir)) if args.directory is None else Path(args.directory)

    clip_instance = CLIP(str(cur),
                           max_categ,
                           args.topn,
                           model_name,
                           device,
                           str(output_dir))

    if args.file is not None:
        c, pred = clip_instance.predict_single(args.file)
        print(clip_instance.categories)
        print(c, pred)

    if args.dir:
        directory_result_output = str(
            Path(cur / f'{output_dir}/tables/raw_result_{args.topn}n_{max_categ}c.csv'))
        clip_instance.predict_directory(str(input_dir), raw=True, out_table=directory_result_output)

