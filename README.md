**Goal:** This project solves a task of page images classification

**Scope:** Processing of images into features, h5 dataset compilation, 
Random Forest Classifier training and evaluation using confusing matrices,
input file/directory processing, class (category) results of top N predictions output 
and summarizing into a tabular format 


**Categories:**

DRAW:	**782**	7.86% - drawings, maps, paintings 

DRAW_L:	**731**	7.35% - drawings, maps, paintings inside tabular layout

LINE_HW:	**813**	8.18% - hndwritten text lines inside tabular layout

LINE_P:	**691**	6.95% - printed text lines inside tabular layout

LINE_T:	**1182**	11.89% - typed text lines inside tabular layout

PHOTO:	**853**	8.58% - photos with text

PHOTO_L:	**603**	6.06% - photos inside tabular layout

TEXT:	**1015**	10.21% - mixed types, printed, and handwritten texts

TEXT_HW:	**1332**	13.4% - handwritten text

TEXT_P:	**596**	5.99% - printed text

TEXT_T:	**1346**	13.54% - typed text

**How to run:**

Open [.env](.env) environment file where all output folder paths are defined - please change all of them

Change paths to folders by replacing the beginnings of directory paths with your own **FULL** directory paths (to 
existing or not directories)

Use pip to install dependencies:

    pip install -r requirements.txt

Run the program from its starting point [run.py](run.py) with optional flags:

    python3 run.py -tn 3 -f '/full/path/to/file.png'
to run single PNG file classification with top 3 predictions

    python3 run.py -tn 3 --dir -d '/full/path/to/directory' 
to parse all PNG files in the directory (+ its subdirectories) and classify all pages (RECOMMENDED)

The results of PNG pages classification will be saved to related folders 

Tip: you can run the script in the second terminal window without ``-f`` flag to skip DD processing and run only 
category prediction for already collected page layout stat files

    python3 run.py -tn 3 -t 333 -w 0 -f '/full/path/to/file'
best file processing setup with 333 trees in the Random Forest Classifier, 
balanced (by size) class weights, 

    python3 run.py -tn 3 -t 333 -w 0 --dir -d '/full/path/to/directory'
best directory processing setup


**Step-by-step workflow explanations:**

Single PNG processing steps:

**1.1**     From the image extracting the following features: HuMomemts, Haralick Texture, and Color Histogram descriptors 

**1.2**     The source image undergoes Otsu binarization

**1.3**      From the grayscale binary version of the initial image extracting the same features: HuMomemts, Haralick Texture, and Black-White portion descriptors 

**1.4**      If it's a training mode, the features are saved to the h5 dataset, otherwise, the the features are sent to the model for prediction

For the training and testing mode:

**2.1**     After the dataset is compiled, category samples are split into training and testing sets according to the classes ratio, and max number of class samples provided 

**2.2**     The Random Forest Classifier is trained (and saved to pkl file) on the training set and the Top-1 accuracy is calculated on the testing set

**2.3**     For the test set, the confusion matrix is calculated and saved to the output folder 

**2.4**     For the test set predictions, the generalized labels are assigned and plotted on the confusion matrix that is also saved to the output folder 

**2.5**     The Top-N predictions, including class labels, normalized scores, certainty measure, and golden label are saved to the output folder in tabular format

For the inference mode:

**3.1**     The model is loaded from the pkl file 

**3.2**     For the single input file or collection scraped from the input directory features are extracted 

**3.3**     The features are sent to the model for prediction, raw class scores obtained

**3.4**     The Top-N predictions, including class labels, normalized scores, certainty measure, and filename info are saved to the output folder in tabular format

**Note!** it will take some time for OpenCV and mahotas to process all pages from the directory

Code of the algorithms can be found in the [classifier.py](classifier.py) file:

Code of the main function in the starting point [run.py](run.py) file can be edited. 
If [.env](.env) variables are not loaded - change filenames in the main function of [run.py](run.py)


**TIP**     You can set up default values of _topn_, _file_ and _directory_ values in the main function of
[run.py](run.py) and then run the script via:

    python3 run.py --dir 

which is for directory (and subdirectories) level processing

    python3 run.py 

which is for PDF file level processing
