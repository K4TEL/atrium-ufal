from pdf_parser import *
from layout_sorter import *
import argparse
import os
import pytesseract

if __name__ == "__main__":
    file_small = "/lnet/work/people/lutsai/pythonProject/OCR/ltp-ocr/OCR_demo/CTX199706756.pdf"
    file_big = "/lnet/work/people/lutsai/pythonProject/OCR/ltp-ocr/OCR_demo/CTX194604301.pdf"

    folder_brno = "/lnet/work/people/lutsai/atrium/Brno-20240119T165503Z-001/Brno"
    folder_arup = "/lnet/work/people/lutsai/atrium/ATRIUM_ARUP_vzorek/CTX"

    credibility = 0.0



    parser = argparse.ArgumentParser(description='OCR PDF/PNG parser')
    parser.add_argument('-f', "--file", type=str, default=file_small, help="Single PDF file path")
    parser.add_argument('-d', "--directory", type=str, default=folder_brno, help="Path to folder with PDF files")
    parser.add_argument('-c', "--cred", type=float, default=credibility, help="Minimal credibility of DeepDoctection predictions")
    parser.add_argument("--dir", help="Process whole directory", action="store_true")
    parser.add_argument("--pdf", help="Process PDF files into layouts", action="store_true")
    parser.add_argument("--pro", help="Use the best recognition model with OCR", action="store_true")
    parser.add_argument("--force", help="With --pdf to reform page stat and pdf summary files ", action="store_true")

    args = parser.parse_args()

    load_dotenv()

    cur = Path.cwd() #  directory with this script
    # locally creating new directory pathes instead of .env variables loaded with mistakes
    layout_dd_output_folder = Path(os.environ.get('FOLDER_LAYOUTS_DD', cur / "layouts_dd"))  # weights a lot, contains encoded page images
    page_stats_output_folder = Path(os.environ.get('FOLDER_STATS', cur / "layout_stat"))  # weights a little, selected features data
    page_images_folder = Path(os.environ.get('FOLDER_PAGES', cur / "pages"))  # weights a lot, contains grayscale page images, detected layouts and line plots
    pdf_summary_output_folder = Path(os.environ.get('FOLDER_SUMMARY', cur / "layouts_pdf"))  # weights a little, contains separate pdf files stat summaries
    results_table_folder = Path(os.environ.get('FOLDER_RESULTS', cur / "results"))  # weights a little, page and pdf level summaries with classified categories in csv format
    ocr_output_folder = Path(os.environ.get('FOLDER_TEXTS', cur / "ocr_text"))  # weights a little, files of recognized text for separate pdf files

    pytesseract.pytesseract.tesseract_cmd = os.environ['TESSERACT']  # should be preinstalled

    pdf_parser = PDF_parser(output_folder=layout_dd_output_folder,
                            credibility=args.cred,
                            pro_ocr=args.pro,
                            force=args.force)  # turns pdf to png, txt, and json page files, TODO image preprocessing
    if args.pdf:
        if args.dir:
            pdf_parser.folder_to_page_layouts(Path(args.directory))  # called on directory with pdf files or subdirectories containing pdf files
        else:
            pdf_parser.pdf_to_table(Path(args.file))   # called on pdf file to get its dd layout json, stats json, and tsv summary table

    tabler = Layout_sorter(output_folder=results_table_folder, credibility=args.cred,
                           pro_ocr=args.pro)  # turns stats to csv summary table
    if args.dir:
        tabler.folder_process_level()   # page level summary of all page stat jsons
        tabler.folder_process_level(page=False)    # pdf level summary of all pdf summary tsv files
    else:
        tabler.file_process_level(Path(args.file))  # page-only summary for single path processing

