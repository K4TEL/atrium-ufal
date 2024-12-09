import os
import pickle
from pyexpat import features

import cv2
import mahotas
import h5py
import random
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier


max_categ = 500
trees = 100
base_dir = "/lnet/work/people/lutsai/pythonProject"
input_dir = 'pages/train_data'
test_dir = 'pages/test_data'

output_dir = "results"


category_map = {
    1: {
        "PHOTO": ["PHOTO", "PHOTO_L"]
    },
    2: {
        "DRAW": ["DRAW", "DRAW_L"]
    },
    3: {
        "LINE": ["LINE_HW", "LINE_P",  "LINE_T"]
    },
    4: {
        "TEXT": ["TEXT", "TEXT_HW", "TEXT_P", "TEXT_T"]
    }
}

w_class = True


# get list of all files in the folder and nested folders by file format
def directory_scraper(folder_path: Path, file_format: str = "png", file_list: list = None) -> list[str]:
    if file_list is None:
        file_list = []
    file_list += list(folder_path.rglob(f"*.{file_format}"))
    print(f"[ {file_format.upper()} ] \tFrom directory {folder_path} collected {len(file_list)} {file_format} files")
    return file_list


# feature-descriptor-1: Hu Moments
def fd_hu_moments(image, bin=False):
    feature = cv2.HuMoments(cv2.moments(image, bin)).flatten()
    return feature


# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    haralick = mahotas.features.haralick(image).mean(axis=0)
    return haralick


# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def img_to_feature(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    fv_hu_moments, fv_haralick, fv_histogram = fd_hu_moments(img), fd_haralick(img), fd_histogram(img)

    threshold = mahotas.otsu(img)
    binarized_image = np.where(img > threshold, 255, 0)
    iw, ih = binarized_image.shape

    bi_fv_hu_moments, bi_fv_haralick = fd_hu_moments(binarized_image, True), fd_haralick(
        binarized_image)

    black, white = 0, 0
    for r in range(iw):
        for c in range(ih):
            if binarized_image[r, c] > 0:
                white += 1
            else:
                black += 1

    area = iw * ih
    bi_fv_histogram = [black / area, white / area]

    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments,
                                bi_fv_haralick, bi_fv_hu_moments, bi_fv_histogram])

    return global_feature


class RFC:
    def __init__(self, base_path: str, page_folder: str, max_category_samples: int, tree_N: int, weighted: bool):
        self.page_dir = f"{base_path}/{page_folder}"
        self.model_dir = f"{base_path}/model"
        self.data_dir = f"{base_path}/dataset"
        self.upper_category_limit = max_category_samples
        self.weighted_classes = weighted

        self.N_trees = tree_N
        self.test_size = 0.1
        self.seed = 42

        suffix = f"{self.upper_category_limit}c"

        self.h5f_data = f'data_{suffix}.h5'
        self.h5f_labels = f'labels_{suffix}.h5'

        self.dataset_name = f"dataset_{suffix}"
        self.model_name = f"model_{suffix}_{self.N_trees}t.pkl"

        self.categories = os.listdir(self.page_dir)
        print(f"Category input directories found: {self.categories}")

        self.data = {
            "train": {
                "X": None,
                "Y": None
            },
            "test": {
                "X": None,
                "Y": None
            }
        }

        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        self.label_priority = {i: 0 for i, v in enumerate(self.categories)}
        self.label_general = {i: v for i, v in enumerate(self.categories)}
        self.general_categories = []
        for proir, label_dict in category_map.items():
            for gen_label, labels_list in label_dict.items():
                self.general_categories.append(gen_label)
                for i, label in enumerate(labels_list):
                    label_id = self.categories.index(label)
                    self.label_priority[label_id] = (5 - proir) * 5 + i  # higher prior values for more important categs
                    self.label_general[label_id] = gen_label

        print(f"Category mapping: {self.label_general}")
        print(f"Category priority: {self.label_priority}")

    def process_page_directory(self):

        if Path(f"{self.data_dir}/{self.h5f_data}").is_file() and Path(f"{self.data_dir}/{self.h5f_labels}").is_file():
            print(f"Dataset {self.dataset_name} has been already processed")
            return
        else:
            print(f"{self.data_dir}/{self.h5f_data}", f"{self.data_dir}/{self.h5f_labels}")

        features_data, labels = [], []

        for category_idx, category in enumerate(self.categories):
            all_category_files = os.listdir(os.path.join(self.page_dir, category))
            if len(all_category_files) > self.upper_category_limit:
                random.shuffle(all_category_files)
                all_category_files = all_category_files[:self.upper_category_limit]

            for i, file in enumerate(all_category_files):
                print(i, file, category)
                img_path = os.path.join(self.page_dir, category, file)
                global_feature = img_to_feature(img_path)

                features_data.append(global_feature)
                labels.append(category_idx)

        data = np.asarray(features_data)
        labels = np.asarray(labels)

        # print(data.shape, labels.shape)

        rescaled_features = self.scaler.fit_transform(data)

        # save the feature vector using HDF5
        h5f_data = h5py.File(self.h5f_data, 'w')
        h5f_data.create_dataset(self.dataset_name, data=np.array(rescaled_features))

        h5f_label = h5py.File(self.h5f_labels, 'w')
        h5f_label.create_dataset(self.dataset_name, data=labels)

        h5f_data.close()
        h5f_label.close()

        print(f"Data {data.shape} and labels {labels.shape} saved to f{self.dataset_name}")

    def load_features_dataset(self):
        h5f_data = h5py.File(f"{self.data_dir}/{self.h5f_data}", 'r+')
        h5f_label = h5py.File(f"{self.data_dir}/{self.h5f_labels}", 'r+')

        #h5f_data[self.dataset_name] = h5f_data["dataset_v2_500"]
        #h5f_label[self.dataset_name] = h5f_label["dataset_v2_500"]
        #del h5f_label["dataset_v2_500"]
        #del h5f_data["dataset_v2_500"]

        global_features_string = h5f_data[self.dataset_name]
        global_labels_string = h5f_label[self.dataset_name]

        global_features = np.array(global_features_string)
        global_labels = np.array(global_labels_string)

        h5f_data.close()
        h5f_label.close()

        label, count = np.unique(global_labels, return_counts=True)

        print(f"From dataset file {self.dataset_name} loaded:")

        for label_id, label_count in zip(label, count):
            print(f"{self.categories[label_id]}:\t{label_count}\t{round(label_count / len(global_labels) * 100, 2)}%")

        (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                                  np.array(global_labels),
                                                                                                  test_size=self.test_size,
                                                                                                  random_state=self.seed,
                                                                                                  stratify=np.array(
                                                                                                      global_labels))

        self.data["train"]["X"] = trainDataGlobal
        self.data["test"]["X"] = testDataGlobal
        self.data["train"]["Y"] = trainLabelsGlobal
        self.data["test"]["Y"] = testLabelsGlobal

        print("[STATUS] splitted train and test data...")
        print("Train data  : {}".format(trainDataGlobal.shape))
        print("Test data   : {}".format(testDataGlobal.shape))
        print("Train labels: {}".format(trainLabelsGlobal.shape))
        print("Test labels : {}".format(testLabelsGlobal.shape))

        for category_idx, category in enumerate(self.categories):
            all_category_files = os.listdir(os.path.join(self.page_dir, category))
            if len(all_category_files) > self.upper_category_limit:
                self.label_priority[category_idx] *= 1
            else:
                self.label_priority[category_idx] *= 2

    def train(self):

        model_file = f"{self.model_dir}/{'w_' if self.weighted_classes else ''}{self.model_name}"

        if Path(model_file).is_file():
            print(f"RFC model with current parameters already exists")
        else:
            print(f"Training RFC of {self.N_trees} trees")
            print(f"Category weights: {self.label_priority}")
            if self.weighted_classes:
                clf = RandomForestClassifier(n_estimators=self.N_trees, random_state=self.seed, class_weight=self.label_priority)
            else:
                clf = RandomForestClassifier(n_estimators=self.N_trees, random_state=self.seed)

            clf.fit(self.data["train"]["X"], self.data["train"]["Y"])

            with open(model_file, 'wb') as f:
                pickle.dump(clf, f)
            print(f"Model saved to {self.model_name}")

        # load
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)

        # fast eval
        acc = self.model.score(self.data["test"]["X"], self.data["test"]["Y"])
        print(f"Model accuracy:\t{round(acc * 100, 2)}%")

    def test(self):
        if self.model is None:
            # load
            with open(f"{self.model_dir}/{self.model_name}", 'rb') as f:
                self.model = pickle.load(f)

            print(f"Model loaded from {self.model_name}")

        predictions = self.model.predict(self.data["test"]["X"])
        print('Percentage correct: ',
              round(100 * np.sum(predictions == self.data["test"]["Y"]) / len(self.data["test"]["Y"]), 2))

        disp = ConfusionMatrixDisplay.from_predictions(self.data["test"]["Y"], predictions,
                                                       normalize="true", display_labels=np.array(self.categories))

        print(f"\t{' '.join(disp.display_labels)}")
        for ir, row in enumerate(disp.confusion_matrix):
            print(f"{disp.display_labels[ir]}\t{'   '.join([str(val) if val > 0 else ' -- ' for val in np.round(row, 2)])}")

        plt.savefig(f'{"w_" if self.weighted_classes else ""}conf_{self.upper_category_limit}c_{self.N_trees}t.png', bbox_inches='tight')
        plt.close()

        general_prediction = np.array([self.general_categories.index(self.label_general[l]) for l in predictions])
        gegeral_truth = np.array([self.general_categories.index(self.label_general[l]) for l in self.data["test"]["Y"]])

        print('General percentage correct: ', round(100 * np.sum(general_prediction == gegeral_truth) / len(gegeral_truth), 2))

        disp_gen = ConfusionMatrixDisplay.from_predictions(gegeral_truth, general_prediction,
                                                       normalize="true", display_labels=np.array(self.general_categories))

        print(f"\t{' '.join(disp_gen.display_labels)}")
        for ir, row in enumerate(disp_gen.confusion_matrix):
            print(f"{disp_gen.display_labels[ir]}\t{'   '.join([str(val) if val > 0 else ' -- ' for val in np.round(row, 2)])}")

        plt.savefig(f'{"w_" if self.weighted_classes else ""}gen_conf_{self.upper_category_limit}c_{self.N_trees}t.png', bbox_inches='tight')
        plt.close()

    def predict_single(self, image_file: str):
        feature = img_to_feature(image_file)

        category_distrib = self.model.predict_proba(feature.reshape(1, -1))[0]
        return category_distrib, self.categories[np.argmax(category_distrib)]

    def predict_directory(self, folder_path: str, n: int, out_table: str = None):
        images = directory_scraper(Path(folder_path), "png")
        data_features, data_ids = [], []

        random.shuffle(images)
        images = images[:n]

        for image_file in images:
            img_path = os.path.join(folder_path, image_file)
            feature = img_to_feature(img_path)
            data_features.append(feature)
            data_ids.append(Path(img_path).stem)

        category_distrib = self.model.predict_proba(data_features)

        categories = np.argmax(category_distrib, axis=1)

        if out_table is not None:
            labels = [self.categories[c] for c in categories]
            scores = np.max(category_distrib, axis=1)
            filenames = [d.split("-")[0] for d in data_ids]
            page_nums = [d.split("-")[1] for d in data_ids]

            dict = {'FILE': filenames, 'PAGE': page_nums, 'CLASS': labels, 'SCORE': scores}

            df = pd.DataFrame(dict)
            df.to_csv(out_table, sep=",")

        return category_distrib, [self.categories[c] for c in categories]


RandomForest_classifier = RFC(base_dir, input_dir, max_categ, trees, w_class)

RandomForest_classifier.process_page_directory()
RandomForest_classifier.load_features_dataset()
RandomForest_classifier.train()
RandomForest_classifier.test()

c, pred = RandomForest_classifier.predict_single(os.path.join(base_dir, test_dir, "CTX194604301-15.png"))
print(c, pred)
#cs, preds = RandomForest_classifier.predict_directory(os.path.join(base_dir, test_dir), 5, os.path.join(base_dir, output_dir, "res.csv"))
#print(cs, preds)

