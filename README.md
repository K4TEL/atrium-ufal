**Goal:** This project solves a task of layouts analysis

**Scope:** Parsing PDF files to extract layouts was implemented with DeepDoctection, further sorting is manual

**How to run:**

Open [.env](.env) environment file where all output folder paths are defined.
NOTE that only the last folder path is going to contain the resulting files, while others - temporary files.

Change paths to folders by replacing the beginnings of directory paths with your own **FULL** directory paths (to existing or not folders)

Use pip to install dependencies:

``pip install -r requirements.txt``

Run the program from its starting point [script.py](script.py) with optional flags:

``python3 script.py --pdf -f '/full/path/to/file'`` to run single PDF file parsing and following layout extraction from its pages

``python3 script.py --pdf --dir -d '/full/path/to/directory'`` to parse all PDF files in the directory and extract layouts from all pages (RECOMMENDED)

The results of PDF to table parsing will be saved to related folders with page numbers added to PDF filename. 

``--pro`` flag to use Tesseract-based methods of OCR during page image processing - 
using Table extraction and PDF miner modules of DeepDoctection

``-c 0.1`` or ``--cred 0.1`` flag to setup threshold value of prediction scores

**Explanations:**

Single PDF processing steps:

1. DD (DeepDoctection) layout of all pages in the PDF (_layouts_dd_ folder)
 
2. JSON summaries of page layouts (_layout_stat_ folder)
 
3. table summary of the whole PDF in .tsv format (_layout_pdf_ folder)

and finally:

4. directory summary in .csv format (_results_ folder)

NOTE! All .json and .tsv Files in the output directories can be deleted after the table summary .csv file of the input directory have been created

You may need to **rerun** the program several times, all saved files count as a progress of the whole directory processing. 

Code of the algorithms can be found in [pdf_parser.py](pdf_parser.py) and [layout_sorter.py](layout_sorter.py) script

Code of the starting point [script.py](script.py) can be edited. 
If .env variables are not loaded - change filenames in the beginning of [script.py](script.py)

The repository files include a test document [CTX199706756.pdf](CTX199706756.pdf) referenced in [script.py](script.py)