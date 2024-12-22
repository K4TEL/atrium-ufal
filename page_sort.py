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


max_categ = 701
trees = 100

w_class = 0
weight_options = ["balance", "size-prior", "priority", "no"]


base_dir = "/lnet/work/people/lutsai/pythonProject"
input_dir = f'pages/train_{"data" if max_categ % 100 == 0 else "balanced"}'
test_dir = 'pages/test_data'

large_dataset = f"{base_dir}/dataset/data_tb_1301c.h5"
large_dataset_labels = f"{base_dir}/dataset/labels_tb_1301c.h5"
large_dataset_name = "dataset_1301c"

# large_dataset = f"{base_dir}/dataset/data_td_900c.h5" if max_categ % 100 == 0 else f"{base_dir}/dataset/data_tb_1301c.h5"
# large_dataset_labels = f"{base_dir}/dataset/labels_td_900c.h5" if max_categ % 100 == 0 else f"{base_dir}/dataset/labels_tb_1301c.h5"
# large_dataset_name = "dataset_900c" if max_categ % 100 == 0 else "dataset_1301c"


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

    if img is None:
        return None

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



class DataWrapper:
    def __init__(self, base_path: str, page_folder: str, max_category_samples: int):
        self.page_dir = f"{base_path}/{page_folder}"
        self.data_dir = f"{base_path}/dataset"
        self.upper_category_limit = max_category_samples

        self.seed = 42
        self.test_size = 0.1

        suffix = f"{self.upper_category_limit}c"
        self.dataset_name = f"dataset_{suffix}"

        self.folder_abbr = "".join([w[0] for w in page_folder.split("/")[1].split("_")])
        self.h5f_data = f'data_{self.folder_abbr}_{suffix}.h5'
        self.h5f_labels = f'labels_{self.folder_abbr}_{suffix}.h5'

        # self.categories = sorted(os.listdir(self.page_dir))
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

    def process_page_directory(self):

        data_path = f"{self.data_dir}/{self.h5f_data}"
        label_path = f"{self.data_dir}/{self.h5f_labels}"

        if Path(data_path).is_file() and Path(label_path).is_file():
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

                if global_feature is not None:
                    features_data.append(global_feature)
                    labels.append(category_idx)

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

    def load_features_dataset(self, data_path: str = None, labels_path: str = None, data_name: str = None):
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
        reordered_labels = []
        for l in global_labels:
            reordered_labels.append(ordered_categories.index(self.categories[l]))

        reordered_labels = np.asarray(reordered_labels)
        del h5f_label[self.dataset_name if data_name is None else data_name]
        h5f_label.create_dataset(self.dataset_name if data_name is None else data_name, data=reordered_labels)

        self.categories = ordered_categories
        global_labels = reordered_labels

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

        # category_size = {category: len(os.listdir(os.path.join(self.page_dir, category))) for category in self.categories}
        # largest_size = min(self.upper_category_limit, max(category_size.values()))
        # total_samples = sum([min(self.upper_category_limit, categ_size) for categ_size in category_size.values()])
        #
        # category_size_weights = {cat: total_samples / (len(self.categories) * min(self.upper_category_limit, cat_size)) for cat, cat_size in category_size.items()}
        #
        # print(category_size_weights)
        #
        # print(category_size)
        # print(largest_size)
        # print(total_samples)
        #
        # raise NotImplementedError
        #
        # for category_idx, category in enumerate(self.categories):
        #     all_category_files = os.listdir(os.path.join(self.page_dir, category))
        #     if len(all_category_files) > self.upper_category_limit:
        #         self.label_priority[category_idx] *= 1
        #     else:
        #         self.label_priority[category_idx] *= 2


class RFC:
    def __init__(self, dataset: DataWrapper, base_path: str, max_category_samples: int, tree_N: int, weight_opt: int):
        self.model_dir = f"{base_path}/model"
        self.data_dir = f"{base_path}/dataset"
        self.upper_category_limit = max_category_samples
        self.weighted_classes = weight_options[weight_opt]

        self.N_trees = tree_N
        self.seed = 42

        suffix = f"{self.upper_category_limit}c"
        self.dataset_name = f"dataset_{suffix}"

        # folder_abbr = "".join([w[0] for w in page_folder.split("/")[1].split("_")])
        # self.h5f_data = f'data_{folder_abbr}_{suffix}.h5'
        # self.h5f_labels = f'labels_{folder_abbr}_{suffix}.h5'
        self.model_name = f"model_{dataset.folder_abbr}_{self.weighted_classes[0]}_{suffix}_{self.N_trees}t.pkl"

        # self.categories = sorted(os.listdir(self.page_dir))
        self.categories = dataset.categories
        self.label_priority = dataset.label_priority
        self.label_general = dataset.label_general
        self.general_categories = dataset.general_categories
        self.page_dir = dataset.page_dir

        print(f"Category input directories found: {self.categories}")

        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        self.data_train = dataset.data["train"]
        self.data_test = dataset.data["test"]



    def train(self):
        model_file = f"{self.model_dir}/{self.model_name}"

        if Path(model_file).is_file():
            print(f"RFC model with current parameters already exists")
        else:
            print(f"Training RFC of {self.N_trees} trees")
            print(f"Category weights:", {self.categories[k]: v for k, v in self.label_priority.items()})

            print(f"{self.weighted_classes} weights applied")

            if self.weighted_classes == "no":
                clf = RandomForestClassifier(n_estimators=self.N_trees, random_state=self.seed)
            elif self.weighted_classes == "balance":
                category_size = {category: len(os.listdir(os.path.join(self.page_dir, category))) for category in
                                 self.categories}
                total_samples = sum(
                    [min(self.upper_category_limit, categ_size) for categ_size in category_size.values()])
                category_size_weights = {
                    cat: total_samples / (len(self.categories) * min(self.upper_category_limit, cat_size)) for
                    cat, cat_size in category_size.items()}
                print({self.categories.index(k) : v for k, v in category_size_weights.items()}, "weights")
                clf = RandomForestClassifier(n_estimators=self.N_trees, random_state=self.seed, class_weight="balanced")
            elif self.weighted_classes == "priority":
                reverse_proir = {k: 1/v for k, v in self.label_priority.items()}
                print({self.categories[k] : v for k, v in reverse_proir.items()}, "weights")
                clf = RandomForestClassifier(n_estimators=self.N_trees, random_state=self.seed, class_weight=reverse_proir)
            elif self.weighted_classes == "size-prior":
                category_size = {category: len(os.listdir(os.path.join(self.page_dir, category))) for category in self.categories}
                total_samples = sum([min(self.upper_category_limit, cs) for cs in category_size.values()])

                category_size_weights = {self.categories.index(cat): total_samples / (len(self.categories) * min(self.upper_category_limit, cat_size))
                         - self.label_priority[self.categories.index(cat)] / 10 for cat, cat_size in category_size.items()}
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

    def test(self):
        if self.model is None:
            # load
            with open(f"{self.model_dir}/{self.model_name}", 'rb') as f:
                self.model = pickle.load(f)

            print(f"Model loaded from {self.model_name}")

        predictions = self.model.predict(self.data_test["X"])
        print('Percentage correct: ',
              round(100 * np.sum(predictions == self.data_test["Y"]) / len(self.data_test["Y"]), 2))

        disp = ConfusionMatrixDisplay.from_predictions(self.data_test["Y"], predictions,
                                                       normalize="true", display_labels=np.array(self.categories))

        print(f"\t{' '.join(disp.display_labels)}")
        for ir, row in enumerate(disp.confusion_matrix):
            print(f"{disp.display_labels[ir]}\t{'   '.join([str(val) if val > 0 else ' -- ' for val in np.round(row, 2)])}")

        disp.ax_.set_title(f"{self.weighted_classes[0]} {self.upper_category_limit}c {self.N_trees}t  Full CM folder: {input_dir}")
        plt.savefig(f'conf_{self.weighted_classes[0]}_{self.upper_category_limit}c_{self.N_trees}t.png', bbox_inches='tight')
        plt.close()




        general_prediction = np.array([self.general_categories.index(self.label_general[l]) for l in predictions])
        gegeral_truth = np.array([self.general_categories.index(self.label_general[l]) for l in self.data_test["Y"]])

        print('General percentage correct: ', round(100 * np.sum(general_prediction == gegeral_truth) / len(gegeral_truth), 2))

        disp_gen = ConfusionMatrixDisplay.from_predictions(gegeral_truth, general_prediction,
                                                       normalize="true", display_labels=np.array(self.general_categories))

        print(f"\t{' '.join(disp_gen.display_labels)}")
        for ir, row in enumerate(disp_gen.confusion_matrix):
            print(f"{disp_gen.display_labels[ir]}\t{'   '.join([str(val) if val > 0 else ' -- ' for val in np.round(row, 2)])}")

        disp_gen.ax_.set_title(f"{self.weighted_classes[0]} {self.upper_category_limit}c {self.N_trees}t  General CM folder: {input_dir}")
        plt.savefig(f'gen_conf_{self.weighted_classes[0]}_{self.upper_category_limit}c_{self.N_trees}t.png', bbox_inches='tight')
        plt.close()

    def predict_single(self, image_file: str):
        if self.model is None:
            # load
            with open(f"{self.model_dir}/{self.model_name}", 'rb') as f:
                self.model = pickle.load(f)

            print(f"Model loaded from {self.model_name}")

        feature = img_to_feature(image_file)

        category_distrib = self.model.predict_proba(feature.reshape(1, -1))[0]
        return category_distrib, self.categories[np.argmax(category_distrib)]

    def predict_directory(self, folder_path: str, n: int, out_table: str = None, rename: bool = False):
        if self.model is None:
            # load
            with open(f"{self.model_dir}/{self.model_name}", 'rb') as f:
                self.model = pickle.load(f)

            print(f"Model loaded from {self.model_name}")

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
        labels = [self.categories[c] for c in categories]

        if out_table is not None:
            scores = np.max(category_distrib, axis=1)
            filenames = [d.split("-")[0] for d in data_ids]
            page_nums = [d.split("-")[1] for d in data_ids]

            dict = {'FILE': filenames, 'PAGE': page_nums, 'CLASS': labels, 'SCORE': scores}

            df = pd.DataFrame(dict)
            df.to_csv(out_table, sep=",")

        if rename:
            for image_name, image_label in zip(data_ids, labels):
                original = f"{folder_path}/{image_name}.png"

                short_label = image_label[0]
                if "_" in image_label:
                    short_label += image_label[image_label.index("_")+1]

                new_name = f"{folder_path}/{short_label}_{image_name}.png"

                os.rename(original, new_name)

        return category_distrib, [self.categories[c] for c in categories]



class PDO:
    def __init__(self, swarm_size: int):
        self.swarn_N = swarm_size
        self.iter = 100

        self.w_inertia = 0.2
        self.acc_personal = 1
        self.acc_social = 2

        self.lower_limit = 0.0
        self.upper_limit = 1.0


    def gen_init(self, dim_N: int):
        return np.random.uniform(self.lower_limit, self.upper_limit, (self.swarn_N, dim_N))




from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold




Dataset = DataWrapper(base_dir, input_dir, max_categ)
Dataset.process_page_directory()
Dataset.load_features_dataset()

d = Dataset.data["train"]
seed = 42

print(d)
# print(d.shape)
# print(np.max(d))
# print(np.min(d))

# def models_train(X, Y):
#     # create all the machine learning models
#     models = []
#     models.append(('LR', LogisticRegression(random_state=seed)))
#     models.append(('LDA', LinearDiscriminantAnalysis()))
#     models.append(('KNN', KNeighborsClassifier()))
#     models.append(('CART', DecisionTreeClassifier(random_state=seed)))
#     models.append(('RF', RandomForestClassifier(n_estimators=100, random_state=seed)))
#     models.append(('NB', GaussianNB()))
#     models.append(('SVM', SGDClassifier(random_state=seed)))
#     models.append(('SGD', SVC(random_state=seed)))
#
#     # variables to hold the results and names
#     results = []
#     names = []
#     # 10-fold cross validation
#     for name, model in models:
#         kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
#         cv_results = cross_val_score(model, X, Y, cv=kfold, scoring="accuracy")
#         results.append(cv_results)
#         names.append(name)
#         msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#         print(msg)
#
#     # boxplot algorithm comparison
#     fig = plt.figure()
#     fig.suptitle('Machine Learning algorithm comparison')
#     ax = fig.add_subplot(111)
#     plt.boxplot(results)
#     ax.set_xticklabels(names)
#     plt.show()

# models_train(d["X"], d["Y"])

RandomForest_classifier = RFC(Dataset, base_dir, max_categ, trees, w_class)



# RandomForest_classifier.load_features_dataset(large_dataset, large_dataset_labels, large_dataset_name)

# RandomForest_classifier.train()
RandomForest_classifier.test()

rfc_params = RandomForest_classifier.model.get_params()

print(rfc_params)

# c, pred = RandomForest_classifier.predict_single(os.path.join(base_dir, test_dir, "CTX194604301-15.png"))
# print(c, pred)
# cs, preds = RandomForest_classifier.predict_directory(os.path.join(base_dir, test_dir), 1120, os.path.join(base_dir, output_dir, "res.csv"), True)
# print(cs, preds)

