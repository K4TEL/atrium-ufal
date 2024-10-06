import os.path

import deepdoctection
# import torch

import time
import math
import fitz
import pandas
import pandas as pd
from common_utils import *
from opencv_line_detector import *

import deepdoctection as dd
# from IPython.core.display import HTML
from matplotlib import pyplot as plt
import json
import cv2 as cv
import numpy as np

from pdf2image import convert_from_path

dd_basic_config = [
    "LANGUAGE='ces'",
    "USE_OCR=False",
    "USE_LAYOUT=False",
    "USE_TABLE_SEGMENTATION=False",
    "USE_TABLE_REFINEMENT=False",
    "USE_PDF_MINER=False"
     ]
dd_pro_config = ["LANGUAGE='ces'",
                 "USE_OCR=True",
                 "USE_LAYOUT=True",
                 "USE_TABLE_SEGMENTATION=True",
                 "USE_TABLE_REFINEMENT=True",
                 "USE_PDF_MINER=True"
                 ]


load_dotenv()


# PDF files to png pages parser
class PDF_parser:
    def __init__(self, output_folder: Path = None, credibility: float = 0.1, pro_ocr: bool = False):
        self.layout_output_folder = Path(os.environ['FOLDER_LAYOUTS_DD']) if output_folder is None else output_folder
        self.stat_output_folder = Path(os.environ['FOLDER_STATS'])
        self.pdf_output_folder = Path(os.environ['FOLDER_SUMMARY'])
        self.page_output_folder = Path(os.environ['FOLDER_PAGES'])
        self.page_source_folder = Path(f"{os.environ['FOLDER_PAGES']}_src")

        self.pro_dd = pro_ocr
        self.credibility = credibility

        self.pdf_file_list = []

        self.detector = None

        # current file processing variables

        self.cur_file_name = ""
        self.cur_file_src_folder = self.page_source_folder / self.cur_file_name
        self.cur_stats_summary_file = Path(f"{'pro_' if self.pro_dd else ''}{self.cur_file_name}.tsv")
        self.cur_pdf_page_count = 0

        self.cur_pages = []
        self.cur_layouts = []
        self.cur_stats = []

        self.cur_page_src = []
        self.cur_pdf_src_paths = []

        # creating output directories

        if not self.layout_output_folder.is_dir():
            self.layout_output_folder.mkdir()

        if not self.stat_output_folder.is_dir():
            self.stat_output_folder.mkdir()

        if not self.pdf_output_folder.is_dir():
            self.pdf_output_folder.mkdir()

        if not self.page_output_folder.is_dir():
            self.page_output_folder.mkdir()

        if not self.page_source_folder.is_dir():
            self.page_source_folder.mkdir()

    # load from file sys current PDF's supplementary files
    def update_cur_vars(self, pdf_file: Path, next_file: bool = False) -> None:
        if next_file:
            self.cur_file_name = pdf_file.stem

            self.cur_file_src_folder = self.page_source_folder / self.cur_file_name

            if not self.cur_file_src_folder.is_dir():
                self.cur_file_src_folder.mkdir()

            self.cur_stats_summary_file = self.pdf_output_folder / f"{'pro_' if self.pro_dd else ''}{self.cur_file_name}_cred_{self.credibility}.tsv"
            self.cur_pdf_page_count = fitz.open(pdf_file).page_count

        self.cur_layouts = list(self.layout_output_folder.glob(f"{'pro_' if self.pro_dd else ''}{self.cur_file_name}_page*"))
        self.cur_stats = list(self.stat_output_folder.glob(f"{'pro_' if self.pro_dd else ''}{self.cur_file_name}_page*"))
        self.cur_pages = list(self.page_output_folder.glob(f"{'pro_' if self.pro_dd else ''}{self.cur_file_name}_page*"))
        self.cur_page_src = list(self.cur_file_src_folder.glob("*"))


    # load PDF, go through page layouts and save them as JSON
    def pdf_to_json(self, pdf_file: Path, save=True) -> None:
        self.update_cur_vars(pdf_file, self.cur_file_name == "")

        if len(self.cur_page_src) == self.cur_pdf_page_count:
            print(f"[ IMG ] \t{self.cur_file_name} has {len(self.cur_page_src)} source page(s)")
        else:
            print(f"{self.cur_file_name} to {self.cur_pdf_page_count} page source files")

            self.cur_pdf_src_paths = convert_from_path(str(pdf_file), output_folder=str(self.cur_file_src_folder),
                                                       grayscale=True, paths_only=True, fmt="png")
            print(f"[ +IMG ] \t{self.cur_file_name} saved as {len(self.cur_pdf_src_paths)} source page(s)")

        self.update_cur_vars(pdf_file)

        if len(self.cur_stats) == self.cur_pdf_page_count:
            print(f"[ JSON ] \t{self.cur_file_name} has {len(self.cur_stats)} page stat JSON(s)")
        elif len(self.cur_layouts) == self.cur_pdf_page_count and len(self.cur_stats) != self.cur_pdf_page_count:
            print(f"[ JSON ] \t{self.cur_file_name} has {len(self.cur_layouts)} layout JSON(s)")
        else:
            print(f"{self.cur_file_name} to {self.cur_pdf_page_count} page JSON stats and layout analysis...")


            if self.detector is None:
                print(f"Loading {'pro' if self.pro_dd else ''} DeepDoctection model...")
                self.detector = dd.get_dd_analyzer(config_overwrite = dd_basic_config if not self.pro_dd else dd_pro_config)

            results = self.detector.analyze(path=pdf_file)
            results.reset_state()

            pages = iter(results)
            for i, page in enumerate(pages):
                i += 1
                cur_page_layout_file = self.layout_output_folder / f"{'pro_' if self.pro_dd else ''}{self.cur_file_name}_page_{i}.json"
                cur_page_image_filename = self.page_output_folder / f"{'pro_' if self.pro_dd else ''}{self.cur_file_name}_page_{i}.png"

                if save and not cur_page_image_filename.is_file():
                    save_parsed_image(page, cur_page_image_filename)

                if save and not cur_page_layout_file.is_file():
                    page.save(image_to_json=True, path=cur_page_layout_file)
                    print(f"[ + JSON ] \t{i}/{self.cur_pdf_page_count} page layout {self.layout_output_folder}")

        pdf_text = ""

        self.update_cur_vars(pdf_file)
        if len(self.cur_layouts) == self.cur_pdf_page_count:
            page_numbers = [a for a in range(self.cur_pdf_page_count)]
            for pn in page_numbers:
                pn += 1
                ending = f"-0{pn}.png" if pn < 9 < self.cur_pdf_page_count else f"-{pn}.png"
                srcs = list(self.cur_file_src_folder.glob(f"*{ending}"))
                src_page_image_file = Path(srcs[0])
                cur_page_layout_file = self.layout_output_folder / f"{'pro_' if self.pro_dd else ''}{self.cur_file_name}_page_{pn}.json"
                if not cur_page_layout_file.is_file():
                    time.sleep(2.5 if not self.pro_dd else 5)

                print(f"IMG\t{pn + 1}/{self.cur_pdf_page_count} Page Layout Analysis...")

                page_text, page_json = page_layout_analysis(cur_page_layout_file, src_page_image_file)

                pdf_text += f" {page_text}"

                with open(self.stat_output_folder / f"{'pro_' if self.pro_dd else ''}{self.cur_file_name}_page_{pn}_cred_{self.credibility}.json", 'w') as f:
                    json.dump(page_json, f)
                print(f"[ + JSON ] \t{self.cur_file_name} {pn}/{self.cur_pdf_page_count} stats {self.stat_output_folder}")

        self.update_cur_vars(pdf_file)
        table_summary = merge_stats(self.cur_stats, self.stat_output_folder, self.credibility, self.pro_dd)
        table_summary.to_csv(self.cur_stats_summary_file, sep="\t", index=False)
        print(f"[ + TAB ] \t{self.cur_file_name} summary {self.pdf_output_folder}")

    # load PDF, go through pages and save them as PNG
    def pdf_to_table(self, pdf_file: Path) -> pandas.DataFrame:
        self.update_cur_vars(pdf_file, True)

        if not self.cur_stats_summary_file.is_file():
            self.pdf_to_json(pdf_file)

        if len(self.cur_stats) == self.cur_pdf_page_count and self.cur_stats_summary_file.is_file():
            # print(f"[ TSV ] \t{self.cur_file_name} has summary in the table format")
            return pandas.read_csv(self.cur_stats_summary_file, sep="\t")

        if len(self.cur_stats) == self.cur_pdf_page_count and not self.cur_stats_summary_file.is_file():
            print("[ JSON ] \tEnough page stats found and being recompiled into PDF's table summary...")
            table_summary = merge_stats(self.cur_stats, self.stat_output_folder, self.credibility, self.pro_dd)

            table_summary.to_csv(self.cur_stats_summary_file, sep="\t", index=False)
            print(f"[ + TAB ] \t{self.cur_file_name} summary {self.pdf_output_folder}")

            return table_summary

    # called to process directory path
    def folder_to_page_layouts(self, folder_path: Path) -> None:
        self.pdf_file_list = directory_scraper(Path(folder_path), "pdf")

        for file_path in self.pdf_file_list:
            self.pdf_to_table(Path(file_path))


# merge page stats to pdf summary table
def merge_stats(stats: list, stat_output_folder: Path, credibility: float = None, pro_dd: bool = False) -> pandas.DataFrame:
    pdf_pages = []
    for page_stat_name in stats:
        if credibility is not None and not str(page_stat_name.stem).endswith(f"cred_{credibility}"):
            continue

        if pro_dd and not str(page_stat_name.stem).startswith("pro_"):
            continue

        # print(page_stat_name)
        with open(stat_output_folder / page_stat_name, 'r') as f:
            json_data = json.load(f)

        json_data = json.loads(json_data)

        stat_filename = page_stat_name.stem.split("_")
        json_data['FIG']['FILE'] = stat_filename[1 if pro_dd else 0]
        json_data['FIG']['PAGE'] = stat_filename[3 if pro_dd else 2]

        pdf_pages.append(json_data["FIG"])

    table_summary = pd.DataFrame.from_records(pdf_pages)

    table_summary = table_summary.sort_values(by=['FILE', 'PAGE'])

    print(table_summary)
    return table_summary


# getting stats about particular element type from the layout elements
def el_type_parse(page: deepdoctection.Page,
                  layout: deepdoctection.Layout,
                  el_name: str,
                  el_out_dict: dict,
                  credibility: float = 0.0) -> dict:

    dd_name = None if not el_name else el_name
    dd_name = "title" if el_name == "HDR" else dd_name
    dd_name = "figure" if el_name == "IMG" else dd_name
    dd_name = "text" if el_name == "TXT" else dd_name

    if layout.category_name == dd_name or layout.category_name == "list" and dd_name == "TXT":

        if layout.score < credibility:
            return el_out_dict

        print(f"{layout.score}\t{el_name}: {layout.text}\t{layout.bounding_box.area}")
        el_out_dict[el_name]["AREA"] += layout.bounding_box.area
        el_out_dict[el_name]["COUNT"] += 1

    elif el_name == "TAB" and len(page.tables) > 0:
        for j in range(len(page.tables)):
            tab = page.tables[j]
            print(f"{tab.score}\tTAB: {tab.text}\t{tab.bounding_box.area}")
            print(tab.csv)

            if tab.score < credibility:
                return el_out_dict

            el_out_dict[el_name]["AREA"] += tab.bounding_box.area
            el_out_dict[el_name]["COUNT"] += 1

    # print({el_name}, el_out_dict[el_name])
    return el_out_dict


# saving page with recognized elements in color
def save_parsed_image(page: deepdoctection.Page, cur_page_image_filename: Path) -> None:
    image = page.viz()

    plt.figure(figsize=(16, 23))  # A4 paper ratio
    plt.axis('off')

    plt.imshow(image)
    plt.tight_layout()

    plt.savefig(cur_page_image_filename)
    print(f"[ + PNG ] \t{cur_page_image_filename.stem} layouts image {cur_page_image_filename.parent}")

    plt.close()


# create page stat json
def page_layout_analysis(layout_filename: Path, image_filename: Path, credibility: float = 0.0) :

    long_horiz, long_vert = page_visual_analysis(image_filename)
    print(long_horiz, long_vert)

    page = dd.Page.from_file(file_path=str(layout_filename))
    page_text = str(page.text)
    page_area = page.height * page.width

    found_el_stats = {
        "TXT": {},  # text
        "IMG": {},  # figures
        "TAB": {},  # tables
        "HDR": {}  # titles / headers
    }

    for el_type, el_stat_dict in found_el_stats.items():
        found_el_stats[el_type] = {
            "COUNT": 0,  # number of elements of this type on the page
            "AREA": 0,  # area taken by this type of elements
            "many": False,  # number of this type of elements if the nighest
            "large": False,  # area taken by this type of elements if more than 1/2 page
        }

    for layout in page.layouts:
        found_el_stats = el_type_parse(page, layout, "TAB", found_el_stats, credibility)
        found_el_stats = el_type_parse(page, layout, "HDR", found_el_stats, credibility)
        found_el_stats = el_type_parse(page, layout, "TXT", found_el_stats, credibility)
        found_el_stats = el_type_parse(page, layout, "IMG", found_el_stats, credibility)

    total_content_area = sum([el_stat["AREA"] for el_stat in found_el_stats.values()])

    for el_name, el_stat_dict in found_el_stats.items():
        found_el_stats[el_name]["large"] = el_stat_dict["AREA"] > (page_area / 2)
        found_el_stats[el_name]["many"] = el_stat_dict["COUNT"] > all([e["COUNT"] for name, e
                                                                       in found_el_stats.items()
                                                                       if name != el_name])

    j = json.dumps({'FIG': {"area": page_area,
                # rounded ratio of content to the total page area
                "content": round(total_content_area / page_area, 2),

                # rounded ratio of each element type comparing to the whole
                # (recognized) content area of this page
                "TAB": round(found_el_stats["TAB"]["AREA"] / total_content_area
                             if total_content_area > 0 else 0, 2),
                "TXT": round(found_el_stats["TXT"]["AREA"] / total_content_area
                             if total_content_area > 0 else 0, 2),
                "IMG": round(found_el_stats["IMG"]["AREA"] / total_content_area
                             if total_content_area > 0 else 0, 2),
                "HDR": round(found_el_stats["HDR"]["AREA"] / total_content_area
                             if total_content_area > 0 else 0, 2),

                "TABs": found_el_stats["TAB"]["many"],  # if number of elements is higher than
                "TXTs": found_el_stats["TXT"]["many"],  # number of elements in other categories
                "IMGs": found_el_stats["IMG"]["many"],
                "HDRs": found_el_stats["HDR"]["many"],

                "TAB_N": found_el_stats["TAB"]["COUNT"],  # number of elements on the page
                "TXT_N": found_el_stats["TXT"]["COUNT"],
                "IMG_N": found_el_stats["IMG"]["COUNT"],
                "HDR_N": found_el_stats["HDR"]["COUNT"],

                "H_line": bool(long_horiz),
                "V_line": bool(long_vert),

                "Form": found_el_stats["TAB"]["large"],  # element take more than half ( 1/2 ) space
                "Manuscript": found_el_stats["TXT"]["large"],  # of the whole pagr
                "Gallery": found_el_stats["IMG"]["large"],
                "Front": found_el_stats["HDR"]["large"],

                }}, ensure_ascii=True)

    return page_text, j


# fi = Path("/lnet/work/people/lutsai/pythonProject/pages_src/CTX199706756/6591569c-2e8c-4db6-a7a9-84ab997c7f34-12.png")
# page_visual_analysis(fi)