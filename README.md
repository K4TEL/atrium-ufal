**Goal:** This project solves a task of predicted layouts analysis for the further page sorting

**Scope:** Parsing PDF files to extract layouts was implemented with DeepDoctection, Tesseract, and OpenCV, with 
manual post-processing of the page layout features and summarizing into a tabular format

**How to run:**

Open [.env](.env) environment file where all output folder paths are defined - please change all of them

**WARNING** It's necessary to change path to the preinstalled Tesseract binaries on your machine in the [.env](.env) 
environment file in order to use OCR methods of the ``--pro`` config version of [DeepDoctection](https://github.com/deepdoctection/deepdoctection)

Change paths to folders by replacing the beginnings of directory paths with your own **FULL** directory paths (to 
existing or not directories)

Use pip to install dependencies:

    pip install -r requirements.txt

Run the program from its starting point [script.py](script.py) with optional flags:

    python3 script.py --pdf -f '/full/path/to/file'
to run single PDF file parsing and following layout extraction from its pages

    python3 script.py --pdf --dir -d '/full/path/to/directory' 
to parse all PDF files in the directory (+ its subdirectories) and extract layouts from all pages (RECOMMENDED)

The results of PDF to CSV table parsing will be saved to related folders with page numbers and above flags added to the filenames 

``--pro`` flag to use [Tesseract](https://github.com/tesseract-ocr/tesseract)-based methods of OCR during page image processing - 
using Table extraction and PDF miner modules of DeepDoctection, NOTE! make sure you have installed the Tesseract binaries

``-c 0.1`` or ``--cred 0.1`` flag to setup the **lower threshold** value of prediction scores, default is 0.0

Tip: you can run the script in the second terminal window without ``--pdf`` flag to skip DD processing and run only 
category prediction for already collected page layout stat files

    python3 script.py -f '/full/path/to/file' --pro -c 0.0' 
best file processing setup

    python3 script.py --dir -d '/full/path/to/directory' --pro -c 0.0
best directory processing setup


**Step-by-step workflow explanations:**

Single PDF processing steps:

**1.1**     DD (DeepDoctection) layouts (in JSON format with encoded page images) of all pages in the PDF file (_layouts_dd_ folder) 

**1.2**     grayscale PNGs of all page images, hashed names with page numbers in the end (_pages/src/<filename>_ folder)

**1.3**     PNGs of all page images with all found  DD layout elements plotted ontop (_pages/<filename>_ folder)

it will take some time for DD to process all pages, don't worry if the program 
was interrupted - all progress is saved in the directories provided in [.env](.env) file, and then:

**2.1**     PDF's OCR-ed texts summary in single TXT file ([ocr_text](ocr_text)) 

**2.2**     JSON with parsed features of page layouts (_layout_stat_ folder)

**2.3**     TSV table summary of JSON page stats for the PDF file
 
**3.1**     for saved JSON stats detecting and counting Hough lines (_pages/<filename>_ folder)

**3.2**     predicting category based on JSON stats and lines 

and finally:

**4.1**     CSV format table with page-level details summary of the input PDF file / directory ([results](results))

**4.2**     CSV format table with PDF-level details summary of the input directory ([results](results))

NOTE! All .png, .txt, .json, and .tsv files in the output directories can be deleted after the table summary .csv file 
of the input file/directory have been created

You may need to **rerun** the program several times, all saved files count as a progress of the whole directory processing. 

Code of the algorithms can be found in the following files:

- [opencv_line_detector.py](opencv_line_detector.py) - hough line detection and its results post processing, PNG plots saving
- [pdf_parser.py](pdf_parser.py) - parsing PDF files to PNG sources, PNG layout plots, JSON page stats, TXT text summaries 
- [layout_sorter.py](layout_sorter.py) - compiling JSON stat files to CSV results tables with categories
- [common_utils.py](common_utils.py) - common functions including category prediction algorith

Code of the main function in the starting point [script.py](script.py) file can be edited. 
If [.env](.env) variables are not loaded - change filenames in the main function of [script.py](script.py)

The repository files include test documents [CTX199706756.pdf](CTX199706756.pdf), [CTX194604301.pdf](CTX194604301.pdf) 
referenced in [script.py](script.py)

**TIP**     You can set up default values of _credibility_, _file_ and _directory_ values in the main function of
[script.py](script.py) and then run the script via:

    python3 script.py --dir --pdf --pro

which is for directory (and subdirectories) level processing

    python3 script.py --pdf --pro

which is for PDF file level processing
