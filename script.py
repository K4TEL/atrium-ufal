from pdf_parser import *
from layout_sorter import *
import argparse
from dotenv import load_dotenv
import os

file_small = "CTX200903109.pdf"
file_big = "CTX200502635.pdf"

# load_dotenv()
# pages_output_folder = os.environ.get('FOLDER_PAGES', "/lnet/work/people/lutsai/pythonProject/pages")
# layout_dd_output_folder = os.environ.get('FOLDER_LAYOUTS_DD', "/lnet/work/people/lutsai/pythonProject/layouts_dd")
# page_stats_output_folder = os.environ.get('FOLDER_STATS', "/lnet/work/people/lutsai/pythonProject/layout_stat")
# pdf_summary_output_folder = os.environ.get('FOLDER_SUMMARY', "/lnet/work/people/lutsai/pythonProject/layout_pdf")
# results_table_folder = os.environ.get('FOLDER_RESULTS', "/lnet/work/people/lutsai/pythonProject/results")
# ocr_output_folder = os.environ.get('FOLDER_TEXTS', "/lnet/work/people/lutsai/pythonProject/ocr_text")

folder_brno = "/lnet/work/people/lutsai/atrium/Brno-20240119T165503Z-001/Brno"
folder_arup = "/lnet/work/people/lutsai/atrium/ATRIUM_ARUP_vzorek/CTX"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OCR PDF/PNG parser')
    parser.add_argument('-f', "--file", type=str, default=file_small, help="Single PDF file path")
    parser.add_argument('-d', "--directory", type=str, default=folder_arup, help="Path to folder with PDF files")

    # parser.add_argument('-pf', "--pagefile", type=str, help="Single image file path")
    # parser.add_argument('-pd', "--pagedir", type=str, default=pages_output_folder, help="Path to folder with PNG images")

    parser.add_argument("--dir", help="Process whole directory", action="store_true")
    parser.add_argument("--pdf", help="Process PDF files", action="store_true")

    args = parser.parse_args()

    load_dotenv()

    cur = Path.cwd()
    layout_dd_output_folder = Path(os.environ.get('FOLDER_LAYOUTS_DD', cur / "layouts_dd"))
    page_stats_output_folder = Path(os.environ.get('FOLDER_STATS', cur / "layout_stat"))
    pdf_summary_output_folder = Path(os.environ.get('FOLDER_SUMMARY', cur / "layouts_pdf"))
    results_table_folder = Path(os.environ.get('FOLDER_RESULTS', cur / "results"))
    ocr_output_folder = Path(os.environ.get('FOLDER_TEXTS', cur / "ocr_text"))

    # print(layout_dd_output_folder, ocr_output_folder_gcv, ocr_output_folder)

    pdf_parser = PDF_parser(output_folder=layout_dd_output_folder)  # turns pdf to png, TODO image preprocessing
    if args.pdf:
        if args.dir:
            pdf_parser.folder_to_page_layouts(Path(args.directory))  # called on folder with pdf files and folders with pdf files
        else:
            pdf_parser.pdf_to_table(Path(args.file))   # called on pdf to get its dd layout, stats, and summary table

        tabler = Layout_sorter(output_folder=results_table_folder)  # turns stats to summary table
        tabler.folder_process_level()   # page level
        tabler.folder_process_level(page=False)    # pdf level

