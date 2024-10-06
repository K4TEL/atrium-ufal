from pdf_parser import *
from layout_sorter import *
import argparse
import os
import pytesseract

if __name__ == "__main__":
    file_small = "CTX199706756.pdf"
    file_big = "CTX194604301.pdf"

    folder_brno = "/lnet/work/people/lutsai/atrium/Brno-20240119T165503Z-001/Brno"
    folder_arup = "/lnet/work/people/lutsai/atrium/ATRIUM_ARUP_vzorek/CTX"

    credibility = 0.0

    pytesseract.pytesseract.tesseract_cmd = os.environ['TESSERACT']  # should be preinstalled

    parser = argparse.ArgumentParser(description='OCR PDF/PNG parser')
    parser.add_argument('-f', "--file", type=str, default=file_big, help="Single PDF file path")
    parser.add_argument('-d', "--directory", type=str, default=folder_brno, help="Path to folder with PDF files")
    parser.add_argument('-c', "--cred", type=float, default=credibility, help="Minimal credibility of DeepDoctection predictions")

    # parser.add_argument('-pf', "--pagefile", type=str, help="Single image file path")
    # parser.add_argument('-pd', "--pagedir", type=str, default=pages_output_folder, help="Path to folder with PNG images")

    parser.add_argument("--dir", help="Process whole directory", action="store_true")
    parser.add_argument("--pdf", help="Process PDF files", action="store_true")
    parser.add_argument("--pro", help="Use the best recognition model", action="store_true")

    args = parser.parse_args()

    load_dotenv()

    cur = Path.cwd()
    layout_dd_output_folder = Path(os.environ.get('FOLDER_LAYOUTS_DD', cur / "layouts_dd"))
    page_stats_output_folder = Path(os.environ.get('FOLDER_STATS', cur / "layout_stat"))
    pdf_summary_output_folder = Path(os.environ.get('FOLDER_SUMMARY', cur / "layouts_pdf"))
    results_table_folder = Path(os.environ.get('FOLDER_RESULTS', cur / "results"))
    ocr_output_folder = Path(os.environ.get('FOLDER_TEXTS', cur / "ocr_text"))

    pdf_parser = PDF_parser(output_folder=layout_dd_output_folder,
                            credibility=args.cred,
                            pro_ocr=args.pro)  # turns pdf to png, TODO image preprocessing
    if args.pdf:
        if args.dir:
            pdf_parser.folder_to_page_layouts(Path(args.directory))  # called on folder with pdf files and folders with pdf files
        else:
            pdf_parser.pdf_to_table(Path(args.file))   # called on pdf to get its dd layout, stats, and summary table

        tabler = Layout_sorter(output_folder=results_table_folder, credibility=args.cred,
                               pro_ocr=args.pro)  # turns stats to summary table
        if args.dir:
            tabler.folder_process_level(cred_files=True)   # page level
            tabler.folder_process_level(page=False, cred_files=True)    # pdf level
        else:
            tabler.file_process_level(Path(args.file))

