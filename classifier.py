import os
import pickle
import cv2
import mahotas
from pylab import imshow, gray, show
import h5py
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt
from pathlib import Path

from skimage.io import imread
from skimage.transform import resize, rescale
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from skimage.feature import hog

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
# from sklearn.externals import joblib


from wand.image import Image
from wand.display import display

upper_category_limit = 500

suffix = f"v2_{upper_category_limit}"

# prepare data
input_dir = '/lnet/work/people/lutsai/pythonProject/pages/train_data'
categories = os.listdir(input_dir)
print(categories)
w, h = 256, 256  # scaled weight and height
hw, hh = 1024, 1024  # hog scaled weight and height
margin = 40  # border to be erased

seed = 42
num_trees = 100
test_size = 0.1

datasets_directory = "/lnet/work/people/lutsai/pythonProject/dataset"
models_directory = "/lnet/work/people/lutsai/pythonProject/model"


h5_images = f"images_{suffix}.h5"
h5_data = f'data_{suffix}.h5'
h5_labels = f'labels_{suffix}.h5'
h5_hog = f"hog_{suffix}.h5"

dataset_name = f"dataset_{suffix}"
model_name = f"model_{suffix}.pkl"

scaler = MinMaxScaler(feature_range=(0, 1))



class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """

    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)

        try:  # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])

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


def image_fill_edges_square(image, margin):
    iw, ih = image.shape

    image_core = image[margin:-margin, margin:-margin]
    # print(image_core.shape, image.shape)
    icw, ich = image.shape

    pad_top, pad_right, pad_bottom, pad_left = margin, margin, margin, margin
    if iw > ih:
        diff = icw - ich
        pad_top += int(diff/2)
        pad_bottom += diff - int(diff / 2)
    elif ih > iw:
        diff = ich - icw
        pad_left += int(diff / 2)
        pad_right += diff - int(diff / 2)
    else:
        return image

    # print(pad_top, pad_right, pad_bottom, pad_left)
    result = np.pad(image_core, ((pad_left, pad_right), (pad_top, pad_bottom)), mode='maximum')
    return result


def process_directory_to_dataset(data_folder, data_categories, out_feature, out_image, out_label,
                                 dataset_name, image_preprocess=False):
    scaled_images, features_data, labels = [], [], []

    f_data, f_image, f_label = f"{datasets_directory}/{out_feature}", f"{datasets_directory}/{out_image}", f"{datasets_directory}/{out_label}"

    for category_idx, category in enumerate(data_categories):
        for i, file in enumerate(os.listdir(os.path.join(data_folder, category))):
            print(i, file, category)
            img_path = os.path.join(data_folder, category, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            fv_hu_moments, fv_haralick, fv_histogram = fd_hu_moments(img), fd_haralick(img), fd_histogram(img)

            if image_preprocess:
                threshold = mahotas.otsu(img)
                timg = np.where(img > threshold, 255, 0)
                padded = image_fill_edges_square(timg, margin)
                # img_path = os.path.join(input_dir, category, f"pad-{file}")
                cv2.imwrite(img_path, padded)
                print(img.shape, "->", padded.shape)
                small_img = resize(padded, (w, h))
            else:
                small_img = resize(img, (w, h))

            global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

            scaled_images.append(small_img.flatten())
            features_data.append(global_feature)
            labels.append(category_idx)

    scaled_images = np.asarray(scaled_images)
    data = np.asarray(features_data)
    labels = np.asarray(labels)

    print(scaled_images.shape, data.shape, labels.shape)

    rescaled_features = scaler.fit_transform(data)

    # save the feature vector using HDF5
    h5f_data = h5py.File(f_data, 'w')
    h5f_data.create_dataset(dataset_name, data=np.array(rescaled_features))

    h5f_label = h5py.File(f_label, 'w')
    h5f_label.create_dataset(dataset_name, data=labels)

    h5f_image = h5py.File(f_image, 'w')
    h5f_image.create_dataset(dataset_name, data=scaled_images)

    h5f_data.close()
    h5f_label.close()
    h5f_image.close()


def process_dir_to_dataset(data_folder, data_categories, out_feature, out_label,
                                 dataset_name, image_preprocess=True):
    features_data, labels = [], []

    f_data, f_label = f"{datasets_directory}/{out_feature}", f"{datasets_directory}/{out_label}"

    for category_idx, category in enumerate(data_categories):
        all_category_files = os.listdir(os.path.join(data_folder, category))
        if len(all_category_files) > upper_category_limit:
            random.shuffle(all_category_files)
            all_category_files = all_category_files[:upper_category_limit]

        for i, file in enumerate(all_category_files):
            print(i, file, category)
            img_path = os.path.join(data_folder, category, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # print(img.shape)

            fv_hu_moments, fv_haralick, fv_histogram = fd_hu_moments(img), fd_haralick(img), fd_histogram(img)

            if image_preprocess:
                threshold = mahotas.otsu(img)
                binarized_image = np.where(img > threshold, 255, 0)
                iw, ih = binarized_image.shape

                # print(binarized_image.shape)

                bi_fv_hu_moments, bi_fv_haralick = fd_hu_moments(binarized_image, True), fd_haralick(binarized_image)

                black, white = 0, 0
                for r in range(iw):
                    for c in range(ih):
                        if binarized_image[r, c] > 0:
                            white += 1
                        else:
                            black += 1

                area = iw * ih
                bi_fv_histogram = [black/area, white/area]

                global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments,
                                            bi_fv_haralick, bi_fv_hu_moments, bi_fv_histogram])
            else:
                global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

            features_data.append(global_feature)
            labels.append(category_idx)

    data = np.asarray(features_data)
    labels = np.asarray(labels)

    print(data.shape, labels.shape)

    rescaled_features = scaler.fit_transform(data)

    # save the feature vector using HDF5
    h5f_data = h5py.File(f_data, 'w')
    h5f_data.create_dataset(dataset_name, data=np.array(rescaled_features))

    h5f_label = h5py.File(f_label, 'w')
    h5f_label.create_dataset(dataset_name, data=labels)

    h5f_data.close()
    h5f_label.close()


def hog_dataset(data_directory, data_categories, out_hog, data_name, size=1024):
    images = []
    for category_idx, category in enumerate(data_categories):
        for i, file in enumerate(os.listdir(os.path.join(data_directory, category))):
            img_path = os.path.join(data_directory, category, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            print(i, file, category, img.shape)
            small_img = resize(img, (size, size))

            images.append(small_img)

    images = np.asarray(images)
    print(images.shape)
    hog_images = hogify.fit_transform(images)
    print(hog_images.shape)

    f_hog = f"{datasets_directory}/{out_hog}"
    h5f_hog = h5py.File(f_hog, 'w')
    h5f_hog.create_dataset(data_name, data=hog_images)

    h5f_hog.close()


def read_h5(in_feature, in_label, in_image, in_hog, data_name):
    f_data, f_image = f"{datasets_directory}/{in_feature}", f"{datasets_directory}/{in_image}"
    f_label, f_hog = f"{datasets_directory}/{in_label}", f"{datasets_directory}/{in_hog}"

    h5f_data = h5py.File(f_data, 'r')
    h5f_label = h5py.File(f_label, 'r')
    h5f_image = h5py.File(f_image, 'r')
    h5f_hog = h5py.File(f_hog, 'r')

    global_features_string = h5f_data[data_name]
    global_labels_string = h5f_label[data_name]
    global_image_string = h5f_image[data_name]
    global_hog_string = h5f_hog[data_name]

    global_features = np.array(global_features_string)
    global_labels = np.array(global_labels_string)

    global_images = np.array(global_image_string)
    global_images = np.reshape(global_images, (global_images.shape[0], w, h))

    global_hogs = np.array(global_hog_string)
    # global_hogs = np.reshape(global_hogs, (global_hogs.shape[0], 300, 300))

    h5f_data.close()
    h5f_label.close()
    h5f_image.close()
    h5f_hog.close()

    return global_features, global_labels, global_images, global_hogs


def read_train_h5(in_feature, in_label, data_name):
    f_data, f_label = f"{datasets_directory}/{in_feature}", f"{datasets_directory}/{in_label}"

    h5f_data = h5py.File(f_data, 'r')
    h5f_label = h5py.File(f_label, 'r')

    global_features_string = h5f_data[data_name]
    global_labels_string = h5f_label[data_name]

    global_features = np.array(global_features_string)
    global_labels = np.array(global_labels_string)

    h5f_data.close()
    h5f_label.close()

    label, count = np.unique(global_labels, return_counts=True)
    for label_id, label_count in zip(label, count):
        print(f"{categories[label_id]}:\t{label_count}\t{round(label_count/len(global_labels)*100, 2)}%")

    return global_features, global_labels


def models_train(X, Y):
    # create all the machine learning models
    models = []
    models.append(('LR', LogisticRegression(random_state=seed)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier(random_state=seed)))
    models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SGDClassifier(random_state=seed)))
    models.append(('SGD', SVC(random_state=seed)))

    # variables to hold the results and names
    results = []
    names = []

    # split the training and testing data
    (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(X),
                                                                                              np.array(Y),
                                                                                              test_size=test_size,
                                                                                              random_state=seed,
                                                                                              stratify=np.array(Y))

    print("[STATUS] splitted train and test data...")
    print("Train data  : {}".format(trainDataGlobal.shape))
    print("Test data   : {}".format(testDataGlobal.shape))
    print("Train labels: {}".format(trainLabelsGlobal.shape))
    print("Test labels : {}".format(testLabelsGlobal.shape))

    # 10-fold cross validation
    for name, model in models:
        kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
        cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring="accuracy")
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Machine Learning algorithm comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


def RFC_train(X,  Y, output_file):
    # create the model - Random Forests
    clf = RandomForestClassifier(n_estimators=num_trees, random_state=seed)

    # split the training and testing data
    (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(X),
                                                                                              np.array(Y),
                                                                                              test_size=test_size,
                                                                                              random_state=seed,
                                                                                              stratify=np.array(Y))

    print("[STATUS] splitted train and test data...")
    print("Train data  : {}".format(trainDataGlobal.shape))
    print("Test data   : {}".format(testDataGlobal.shape))
    print("Train labels: {}".format(trainLabelsGlobal.shape))
    # label, count = np.unique(trainLabelsGlobal, return_counts=True)
    # for label_id, label_count in zip(label, count):
    #     print(f"{categories[label_id]}:\t{label_count}\t{round(label_count / len(trainLabelsGlobal) * 100, 2)}%")
    print("Test labels : {}".format(testLabelsGlobal.shape))
    # label, count = np.unique(testLabelsGlobal, return_counts=True)
    # for label_id, label_count in zip(label, count):
    #     print(f"{categories[label_id]}:\t{label_count}\t{round(label_count / len(testLabelsGlobal) * 100, 2)}%")

    # fit the training data to the model
    clf.fit(trainDataGlobal, trainLabelsGlobal)

    # save
    with open(output_file, 'wb') as f:
        pickle.dump(clf, f)

    print(f"Model saved to {output_file}")

    # load
    with open(output_file, 'rb') as f:
        clf2 = pickle.load(f)

    acc = clf2.score(testDataGlobal, testLabelsGlobal)
    print(f"Model accuracy:\t{round(acc*100, 2)}%")




def RFC_test(X, Y, model_file, data_labels):
    with open(model_file, 'rb') as f:
        clf = pickle.load(f)

    print(f"Model loaded from {model_file}")

    # split the training and testing data
    (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(X),
                                                                                              np.array(Y),
                                                                                              test_size=test_size,
                                                                                              random_state=seed,
                                                                                              stratify=np.array(Y))

    print("[STATUS] splitted train and test data...")
    print("Train data  : {}".format(trainDataGlobal.shape))
    print("Test data   : {}".format(testDataGlobal.shape))
    print("Train labels: {}".format(trainLabelsGlobal.shape))
    print("Test labels : {}".format(testLabelsGlobal.shape))

    testLabelPrediction = clf.predict(testDataGlobal)

    # cmx_svm = confusion_matrix(testLabelsGlobal, testLabelPrediction)
    # plot_confusion_matrix(cmx_svm, vmax1=225, vmax2=100, vmax3=12)

    # disp = ConfusionMatrixDisplay(confusion_matrix=cmx_svm, display_labels=np.array(data_labels))
    print('Percentage correct: ', round(100 * np.sum(testLabelPrediction == testLabelsGlobal) / len(testLabelsGlobal), 2))

    disp = ConfusionMatrixDisplay.from_predictions(testLabelsGlobal, testLabelPrediction,
                                                   normalize="true", display_labels=np.array(data_labels))

    # disp.plot()
    plt.show()
    plt.savefig(f'conf_{upper_category_limit}.png', bbox_inches='tight')





hogify = HogTransformer(
    pixels_per_cell=(20, 20),
    cells_per_block=(2, 2),
    orientations=9,
    block_norm='L2-Hys'
)
scalify = StandardScaler()

# process_directory_to_dataset(input_dir, categories, h5_data, h5_images, h5_labels, dataset_name)
# hog_dataset(input_dir, categories, h5_hog, dataset_name)

process_dir_to_dataset(input_dir, categories, h5_data, h5_labels, dataset_name)
# hog_dataset(input_dir, categories, h5_hog, dataset_name)

features, labels = read_train_h5(h5_data, h5_labels, dataset_name)
print(features.shape, labels.shape)

model_out = f"{models_directory}/{model_name}"

RFC_train(features, labels, model_out)

RFC_test(features, labels, model_out, categories)

# models_train(features, labels)


# features, labels, images, hogs = read_h5(h5_data, h5_labels, h5_images, h5_hog, dataset_name)
# print(features.shape, labels.shape, images.shape, hogs.shape)

