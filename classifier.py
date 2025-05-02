import os
import time
import pickle
from collections import Counter
import random
from pathlib import Path

import cv2
import mahotas
import h5py
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing as mp

from skimage.filters import gabor
from skimage.feature import local_binary_pattern



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


# def sift(image):
#     # Initialize SIFT detector
#     sift = cv2.SIFT_create()
#     # Detect keypoints and descriptors
#     keypoints, descriptors = sift.detectAndCompute(image, None)
#     return keypoints, descriptors
#
# def orb(image):
#     # Detect keypoints and descriptors
#     orb_model = cv2.ORB_create()
#     keypoints_orb, descriptors_orb = orb_model.detectAndCompute(image, None)
#     return keypoints_orb, descriptors_orb
#
# def gabor_filter(image_gray):
#     # Apply Gabor filter
#     frequency = 0.2
#     gabor_image, gabor_response = gabor(image_gray, frequency=frequency)
#     return gabor_image, gabor_response
#
# def lbp(image_gray):
#     # Parameters for LBP
#     radius = 5  # Radius of the circular neighborhood
#     n_points = 5 * radius  # Number of sampling points
#     # Apply LBP to the grayscale image
#     lbp_result = local_binary_pattern(image_gray, n_points, radius, method='uniform')
#     # Histogram of LBP
#     hist, _ = np.histogram(lbp_result.ravel(), bins=np.arange(3, n_points + 3), range=(5, n_points + 2))
#     # Normalize the histogram
#     hist = hist.astype('float')
#     hist /= hist.sum()
#
#     return lbp_result, hist


def img_to_feature(img_path):
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(img_path)

    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    threshold = mahotas.otsu(img)
    binarized_image = np.where(img_gray > threshold, 255, 0)
    iw, ih = binarized_image.shape

    fv_hu_moments, fv_haralick, fv_histogram = fd_hu_moments(img_gray), fd_haralick(img_gray), fd_histogram(img_gray)

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


def batch_process_images(image_batch, folder_path):
    """Process a batch of images into features and IDs."""
    data_features, data_ids = [], []
    for image_file in image_batch:
        img_path = os.path.join(folder_path, image_file)
        feature = img_to_feature(img_path)
        data_features.append(feature)
        data_ids.append(Path(img_path).stem)
    # print(f"Processed {len(data_features)} images into features {len(data_features)}")
    return data_features, data_ids


def batch_prepare_images(image_batch):
    """Process a batch of images into features and IDs."""
    data_features = []
    for img_path in image_batch:
        feature = img_to_feature(img_path)
        data_features.append(feature)
    return data_features


def append_to_csv(df, filepath):
    """Append DataFrame to CSV file, or create a new file if it doesn't exist."""
    if not os.path.exists(filepath):
        df.to_csv(filepath, index=False, sep=",")
    else:
        df.to_csv(filepath, mode="a", header=False, index=False, sep=",")


class DataWrapper:
    def __init__(self, base_path: str, page_folder: str, max_category_samples: int, category_map: dict):
        self.page_dir = None
        self.data_dir = f"{base_path}/dataset"
        self.upper_category_limit = max_category_samples

        self.seed = 42
        self.test_size = 0.1

        suffix = f"{self.upper_category_limit}c"
        self.dataset_name = f"dataset_{suffix}"

        if page_folder is not None:
            self.page_dir = f"{page_folder}"

            self.folder_abbr = "".join([w[0] for w in page_folder.split("/")[1].split("_")])

            # self.categories = sorted(os.listdir(self.page_dir))
            self.categories = os.listdir(self.page_dir)
            print(f"Category input directories found: {self.categories}")
        else:
            self.categories =['DRAW', 'DRAW_L', 'LINE_HW', 'LINE_P', 'LINE_T', 'PHOTO', 'PHOTO_L', 'TEXT', 'TEXT_HW', 'TEXT_P', 'TEXT_T']
            print(f"Category loaded from memory: {self.categories}")

            self.folder_abbr = "tf"  # train final

        self.h5f_data = f'data_{self.folder_abbr}_{suffix}.h5'
        self.h5f_labels = f'labels_{self.folder_abbr}_{suffix}.h5'


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

        self.scaler = MinMaxScaler(feature_range=(0, 1))

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

    def process_page_directory(self, batch_size: int = None):

        data_path = f"{self.data_dir}/{self.h5f_data}"
        label_path = f"{self.data_dir}/{self.h5f_labels}"

        if Path(data_path).is_file() and Path(label_path).is_file():
            print(f"Dataset {self.dataset_name} has been already processed")
            return
        else:
            if self.page_dir is None:
                return
            print(f"{self.data_dir}/{self.h5f_data}", f"{self.data_dir}/{self.h5f_labels}")

        features_data, labels = [], []

        random.seed(self.seed)

        total_files = []
        total_labels = []
        for category_idx, category in enumerate(self.categories):
            all_category_files = os.listdir(os.path.join(self.page_dir, category))
            if len(all_category_files) > self.upper_category_limit:
                random.shuffle(all_category_files)
                all_category_files = all_category_files[:self.upper_category_limit]

            total_files += [os.path.join(self.page_dir, category, file) for file in all_category_files]
            total_labels += [category_idx] * len(all_category_files)

        print(len(total_labels), len(total_files))

        batch_size = mp.cpu_count() if batch_size is None else batch_size

        for batch_start in range(0, len(total_files), batch_size):
            batch_files = total_files[batch_start:batch_start + batch_size]
            batch_labels = total_labels[batch_start:batch_start + batch_size]

            # print(len(batch_files))

            with Pool(ncpus=mp.cpu_count()) as pool:
                results = pool.map(batch_prepare_images, [batch_files])

            for ind, image_features in enumerate(results[0]):
                if image_features is not None:
                    features_data.append(image_features)
                    labels.append(batch_labels[ind])

            print(f"Processed {len(labels)} images into features")


        data = np.asarray(features_data)
        labels = np.asarray(labels)
        rescaled_features = self.scaler.fit_transform(data)

        # save the feature vector using HDF5
        h5f_data = h5py.File(data_path, 'w')
        h5f_data.create_dataset(self.dataset_name, data=np.array(rescaled_features))

        h5f_label = h5py.File(label_path, 'w')
        h5f_label.create_dataset(self.dataset_name, data=labels)

        h5f_data.close()
        h5f_label.close()

        print(f"Data {data.shape} and labels {labels.shape} saved to {self.dataset_name}")

    def load_features_dataset(self, data_path: str = None, labels_path: str = None, data_name: str = None, general: bool = False):
        if data_path is None and labels_path is None:
            h5f_data = h5py.File(f"{self.data_dir}/{self.h5f_data}", 'r+')
            h5f_label = h5py.File(f"{self.data_dir}/{self.h5f_labels}", 'r+')
        else:
            h5f_data = h5py.File(data_path, 'r+')
            h5f_label = h5py.File(labels_path, 'r+')

        #h5f_data[self.dataset_name] = h5f_data["dataset_v2_500"]
        #h5f_label[self.dataset_name] = h5f_label["dataset_v2_500"]
        #del h5f_label["dataset_v2_500"]
        #del h5f_data["dataset_v2_500"]

        global_features_string = h5f_data[self.dataset_name if data_name is None else data_name]
        global_labels_string = h5f_label[self.dataset_name if data_name is None else data_name]

        global_features = np.array(global_features_string)
        global_labels = np.array(global_labels_string)

        ordered_categories = sorted(self.categories)
        # print(self.categories)
        print(f"Ordered categories: {[(category, label) for label, category in enumerate(ordered_categories)]}")

        h5f_data.close()
        h5f_label.close()

        label, count = np.unique(global_labels, return_counts=True)

        data_ids = np.arange(0, global_features.shape[0])

        print(f"From dataset file {self.dataset_name if data_name is None else data_name} loaded:")

        ordered_labels = []
        for l, i in zip(global_labels, data_ids):
            # print(l, i, self.categories[l], ordered_categories.index(self.categories[l]))
            if not general:
                ordered_labels.append(ordered_categories.index(self.categories[l]))
            else:
                ordered_labels.append(self.general_categories.index(self.label_general[l]))

        # print(global_labels)
        # print(ordered_labels)
        global_labels = np.array(ordered_labels)

        label, count = np.unique(global_labels, return_counts=True)


        self.categories = ordered_categories if not general else self.general_categories
        if general:
            new_prior = {}
            for k, v in self.label_priority.items():
                new_prior[self.general_categories.index(self.label_general[k])] = v

            self.label_priority = new_prior


        for label_id, label_count in zip(label, count):
            print(f"{self.categories[label_id]}:\t{label_count}\t{round(label_count / len(global_labels) * 100, 2)}%")

        (trainDataGlobal, testDataGlobal,
         trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                          np.array(global_labels),
                                                          test_size=self.test_size,
                                                          random_state=self.seed,
                                                          stratify=np.array(
                                                              global_labels))

        self.data["test"]["X"] = testDataGlobal
        self.data["test"]["Y"] = testLabelsGlobal

        print("[STATUS] splitted train and test data...")
        print("Train data  : {}".format(trainDataGlobal.shape))
        print("Test data   : {}".format(testDataGlobal.shape))
        print("Train labels: {}".format(trainLabelsGlobal.shape))
        print("Test labels : {}".format(testLabelsGlobal.shape))

        train_label_counts = Counter(trainLabelsGlobal)
        train_label_ids = np.arange(0, len(trainLabelsGlobal))

        to_be_removed = []
        for label, counts in train_label_counts.items():
            if counts > self.upper_category_limit:
                print(label, "oversized")
                category_ids = []
                for i in train_label_ids:
                    if trainLabelsGlobal[i] == label:
                        category_ids.append(i)
                print(f"Found {len(category_ids)} IDs")
                # random.shuffle(category_ids)
                # new_category_ids = category_ids[:self.upper_category_limit]
                cropped_category_ids = category_ids[self.upper_category_limit:]

                print(f"{len(cropped_category_ids)} samples of {label} will be removed")

                to_be_removed += cropped_category_ids

        print(f"Removing a total of {len(to_be_removed)} samples from oversized categories")

        trainDataGlobal = np.delete(trainDataGlobal, to_be_removed, 0)
        trainLabelsGlobal = np.delete(trainLabelsGlobal, to_be_removed, 0)

        # for i in to_be_removed:
        #     del trainDataGlobal[i]
        #     del trainLabelsGlobal[i]

        self.data["train"]["X"] = trainDataGlobal
        self.data["train"]["Y"] = trainLabelsGlobal

        print("[STATUS] splitted train and test data...")
        print("Train data  : {}".format(trainDataGlobal.shape))
        print("Test data   : {}".format(testDataGlobal.shape))
        print("Train labels: {}".format(trainLabelsGlobal.shape))
        print("Test labels : {}".format(testLabelsGlobal.shape))





class RFC:
    def __init__(self, dataset: DataWrapper, base_path: str, max_category_samples: int,
                 tree_N: int, weight_opt: str, top_N: int, out: str = None):
        self.model_dir = f"{base_path}/model"
        self.data_dir = f"{base_path}/dataset"
        self.upper_category_limit = max_category_samples
        self.weighted_classes = weight_opt
        self.top_N = top_N

        self.N_trees = tree_N
        self.seed = 42

        self.model_name = f"model_{dataset.folder_abbr}_{self.weighted_classes[0]}_{self.upper_category_limit}c_{self.N_trees}t.pkl"

        # self.categories = sorted(os.listdir(self.page_dir))
        self.categories = dataset.categories
        self.label_priority = dataset.label_priority
        self.label_general = dataset.label_general
        self.general_categories = dataset.general_categories

        print(f"Category input directories found: {self.categories}")

        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        self.data_train = dataset.data["train"]
        self.data_test = dataset.data["test"]

        self.output_dir = "results" if out is None else out



    def train(self, force: bool, general: bool = False):
        model_file = f"{self.model_dir}/{self.model_name}"

        if Path(model_file).is_file() and not force:
            print(f"RFC model with current parameters already exists")
        else:
            print(f"Training RFC of {self.N_trees} trees")
            print(f"Category weights:", {self.categories[k]: v for k, v in self.label_priority.items()})

            print(f"{self.weighted_classes} weights applied")

            label_counts = Counter(self.data_train["Y"])
            label_counts = {k.item() : v for k,v in label_counts.items()}
            total_samples = self.data_train["Y"].shape[0]

            print(label_counts)
            print(total_samples)

            if self.weighted_classes == "no":
                clf = RandomForestClassifier(n_estimators=self.N_trees, random_state=self.seed)
            elif self.weighted_classes == "balance":
                category_size_weights = {
                    cat: total_samples / (len(self.categories) * min(self.upper_category_limit, cat_size)) for
                    cat, cat_size in label_counts.items()}
                print({self.categories[k] : v for k, v in category_size_weights.items()}, "weights")
                clf = RandomForestClassifier(n_estimators=self.N_trees, random_state=self.seed, class_weight="balanced")
            elif self.weighted_classes == "priority":
                reverse_proir = {k: 1/v for k, v in self.label_priority.items()}
                print({self.categories[k] : v for k, v in reverse_proir.items()}, "weights")
                clf = RandomForestClassifier(n_estimators=self.N_trees, random_state=self.seed, class_weight=reverse_proir)
            elif self.weighted_classes == "size-prior":
                category_size_weights = {self.categories.index(cat): total_samples / (len(self.categories) * min(self.upper_category_limit, cat_size))
                         - self.label_priority[self.categories.index(cat)] / 10 for cat, cat_size in label_counts.items()}
                print({self.categories[k] : v for k, v in category_size_weights.items()}, "weights")
                clf = RandomForestClassifier(n_estimators=self.N_trees, random_state=self.seed, class_weight=category_size_weights)

            clf.fit(self.data_train["X"], self.data_train["Y"])

            with open(model_file, 'wb') as f:
                pickle.dump(clf, f)
            print(f"Model saved to {self.model_name}")

        # load
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)

        # fast eval
        acc = self.model.score(self.data_test["X"], self.data_test["Y"])
        print(f"Model accuracy:\t{round(acc * 100, 2)}%")


    def generalize(self, scores: np.array):
        general_scores = np.zeros((scores.shape[0], len(self.general_categories)))
        for i, sample in enumerate(scores):
            for j, cat in enumerate(sample):
                general_label = self.label_general[j]
                general_scores[i, self.general_categories.index(general_label)] += cat
        return general_scores


    def top_N_prediction(self, data_array: np.array, N: int, general: bool = False):
        pred_scores = self.model.predict_proba(data_array)
        best_n = np.argsort(pred_scores, axis=1)[:, -N:]
        best_n_scores = np.sort(pred_scores, axis=1)[:, -N:]

        row_sums = best_n_scores.sum(axis=1)
        best_n_scores_normal = best_n_scores / row_sums[:, np.newaxis]

        score_var = np.var(best_n_scores_normal, axis=1)

        gen_scores = self.generalize(pred_scores) if not general else None

        return -np.sort(-best_n_scores_normal), best_n, pred_scores, np.round(score_var, 5), gen_scores

    def test(self, input_dir: str, general: bool = False):
        if self.model is None:
            # load
            with open(f"{self.model_dir}/{self.model_name}", 'rb') as f:
                self.model = pickle.load(f)

            print(f"Model loaded from {self.model_name}")

        plot_image = f'conf_{self.top_N}n_{self.weighted_classes[0]}_{self.upper_category_limit}c_{self.N_trees}t.png'
        gen_plot_image = "gen_" + plot_image

        predictions = self.model.predict(self.data_test["X"])
        best_n_scores_normal, best_n, raw, var, gen_scores = self.top_N_prediction(self.data_test["X"], self.top_N)

        test_df = pd.DataFrame(best_n, columns=[f"top-{i+1}" for i in range(self.top_N)])

        test_df[[f"score-{i+1}" for i in range(self.top_N)]] = best_n_scores_normal
        for i in range(self.top_N):
            test_df[f"pred-{i+1}"] = test_df[f"top-{i+1}"].apply(lambda x: self.categories[x])

        test_df["true"] = self.data_test["Y"]
        test_df["true_cat"] = test_df["true"].apply(lambda x: self.categories[x])

        test_df.drop(columns=[f"top-{i+1}" for i in range(self.top_N)] + ["true"], inplace=True)

        test_df["certain"] = var

        print(test_df)

        test_df.to_csv(f'{self.output_dir}/tables/test_{self.top_N}n_{self.weighted_classes[0]}_{self.upper_category_limit}c_{self.N_trees}t.csv',
                       index=False, sep=",")

        print('Percentage correct: ',
              round(100 * np.sum(predictions == self.data_test["Y"]) / len(self.data_test["Y"]), 2))
        #
        # disp = ConfusionMatrixDisplay.from_predictions(self.data_test["Y"], predictions,
        #                                                normalize="true", display_labels=np.array(self.categories))
        #
        # print(f"\t{' '.join(disp.display_labels)}")
        # for ir, row in enumerate(disp.confusion_matrix):
        #     print(f"{disp.display_labels[ir]}\t{'   '.join([str(val) if val > 0 else ' -- ' for val in np.round(row, 2)])}")
        #

        # Calculate the percentage correct for top-N predictions
        correct_count = sum(true_label in best_n[i] for i, true_label in enumerate(self.data_test["Y"]))
        percentage_correct = 100 * correct_count / len(self.data_test["Y"])

        print(f'Percentage correct (top-{self.top_N}): ', round(percentage_correct, 2))

        correct_index = []
        for i, true_label in enumerate(self.data_test["Y"]):
            best_n_i = best_n[i]

            sample_rec = False
            for g in range(self.top_N):
                if best_n_i[g] == true_label:
                    correct_index.append(g)
                    sample_rec = True
            if not sample_rec:
                correct_index.append(np.argmax(best_n_i))

        # correct_index = [self.data_test["Y"][i] in best_n[i] for i in range(len(self.data_test["Y"]))]
        # print(best_n)

        # Flatten best_n predictions for confusion matrix
        predictions = np.array(
            [best_n[i, correct_index[i]] for i in range(best_n.shape[0])])  # Using the top prediction for confusion matrix

        # Confusion matrix display and normalized output
        disp = ConfusionMatrixDisplay.from_predictions(
            self.data_test["Y"], predictions,
            normalize="true", display_labels=np.array(self.categories)
        )

        print(f"\t{' '.join(disp.display_labels)}")
        for ir, row in enumerate(disp.confusion_matrix):
            print(
                f"{disp.display_labels[ir]}\t{'   '.join([str(val) if val > 0 else ' -- ' for val in np.round(row, 2)])}")

        disp.ax_.set_title(f"TOP {self.top_N} {self.weighted_classes[0]} {self.upper_category_limit}c {self.N_trees}t  Full CM folder: {input_dir}")
        plt.savefig(f"{self.output_dir}/plots/{plot_image}", bbox_inches='tight')
        plt.close()

        if not general:
            general_prediction = np.array([self.general_categories.index(self.label_general[l]) for l in predictions])
            gegeral_truth = np.array([self.general_categories.index(self.label_general[l]) for l in self.data_test["Y"]])

            print(f'General percentage correct (Top-{self.top_N}): ', round(100 * np.sum(general_prediction == gegeral_truth) / len(gegeral_truth), 2))

            disp_gen = ConfusionMatrixDisplay.from_predictions(gegeral_truth, general_prediction,
                                                           normalize="true", display_labels=np.array(self.general_categories))

            print(f"\t{' '.join(disp_gen.display_labels)}")
            for ir, row in enumerate(disp_gen.confusion_matrix):
                print(f"{disp_gen.display_labels[ir]}\t{'   '.join([str(val) if val > 0 else ' -- ' for val in np.round(row, 2)])}")

            disp_gen.ax_.set_title(f"TOP {self.top_N} {self.weighted_classes[0]} {self.upper_category_limit}c {self.N_trees}t  General CM folder: {input_dir}")
            plt.savefig(f"{self.output_dir}/plots/{gen_plot_image}", bbox_inches='tight')
            plt.close()

    def predict_single(self, image_file: str, N: int = None):
        if self.model is None:
            # load
            with open(f"{self.model_dir}/{self.model_name}", 'rb') as f:
                self.model = pickle.load(f)

            print(f"Model loaded from {self.model_name}")

        feature = img_to_feature(image_file)

        category_distrib = self.model.predict_proba(feature.reshape(1, -1))[self.top_N if N is None else N]
        return category_distrib, self.categories[np.argmax(category_distrib, axis=0)]

    def predict_directory(self, folder_path: str, batch_size: int = None, out_table: str = None, raw: bool = False, general: bool = False):
        if self.model is None:
            # load
            with open(f"{self.model_dir}/{self.model_name}", 'rb') as f:
                self.model = pickle.load(f)

            print(f"Model loaded from {self.model_name}")

        images = directory_scraper(Path(folder_path), "png")

        print(f"Predicting {len(images)} images from {folder_path}")

        time_stamp = time.strftime("%Y%m%d-%H%M")

        batch_size = mp.cpu_count() if batch_size is None else batch_size

        for batch_start in range(0, len(images), batch_size):
            batch_images = images[batch_start:batch_start + batch_size]

            with Pool(ncpus=mp.cpu_count()) as pool:
                results = pool.map(batch_process_images, [batch_images], [folder_path])

            res_table, raw_table = [], []

            for data_features, data_ids in results:
                print(data_features[0].shape, data_ids)
                best_n_scores_normal, best_n, raw_scores, vars, gen_scores = self.top_N_prediction(data_features, self.top_N, general)

                for i, image_f in enumerate(batch_images):
                    categ_labels = [self.categories[c] for c in best_n[i]]
                    categ_scores = np.round(best_n_scores_normal[i], 3)
                    general_scores = np.round(gen_scores[i], 3) if not general else np.array([])
                    # image_filename = data_ids[i].replace("_", "-").replace("-page-", "-")
                    img_file, img_page = data_ids[i].split("-")
                    print(img_file, img_page)
                    res_table.append([img_file, img_page] + categ_labels + categ_scores.tolist() + [vars[i]] + general_scores.tolist())

                    if raw:
                        raw_row = list(raw_scores[i]) + [img_file, img_page] + general_scores.tolist()
                        raw_table.append(raw_row)

                # print(res_table)

                # Append to top-N output
                if out_table is not None:
                    columns = ["FILE", "PAGE"] + \
                              [f"CLASS-{i + 1}" for i in range(self.top_N)] + \
                              [f"SCORE-{i + 1}" for i in range(self.top_N)] + ["CERTAIN"]
                    if not general:
                        columns += self.general_categories
                    top_n_df = pd.DataFrame(res_table, columns=columns)

                    out_table = f'{self.output_dir}/tables/result_{time_stamp}_{self.top_N}n_{self.weighted_classes[0]}_{self.upper_category_limit}c_{self.N_trees}t.csv'
                    append_to_csv(top_n_df, out_table)

                # Append to raw output
                if raw:
                    raw_columns = self.categories + ["FILE", "PAGE"]
                    if not general:
                        raw_columns += self.general_categories
                    raw_df = pd.DataFrame(raw_table, columns=raw_columns)
                    raw_output_path = f'{self.output_dir}/tables/raw_result_{time_stamp}_{self.top_N}n_{self.weighted_classes[0]}_{self.upper_category_limit}c_{self.N_trees}t.csv'
                    append_to_csv(raw_df, raw_output_path)

        print("Processing completed.")
