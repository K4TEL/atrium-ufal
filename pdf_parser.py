import os.path

import time
import fitz
import pandas
from opencv_line_detector import *

import deepdoctection as dd
# from IPython.core.display import HTML
from matplotlib import pyplot as plt
import json

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
    def __init__(self, output_folder: Path = None, credibility: float = None, pro_ocr: bool = False, force: bool = False ):
        self.layout_dd_output_folder = Path(os.environ['FOLDER_LAYOUTS_DD']) if output_folder is None else output_folder

        self.page_stat_output_folder = Path(os.environ['FOLDER_STATS'])
        self.pdf_stat_output_folder = Path(os.environ['FOLDER_SUMMARY'])

        self.page_output_folder = Path(os.environ['FOLDER_PAGES'])
        self.page_source_folder = self.page_output_folder / "src"

        self.pdf_text_output_folder = Path(os.environ['FOLDER_TEXTS'])

        self.pro_dd = pro_ocr
        self.cred_min_tresh = abs(credibility) if credibility is not None else credibility
        self.force_stats = force

        self.dir_pdf_file_list = []

        self.detector_dd = None

        # current file processing variables

        self.cur_filename = ""

        self.cur_file_page_src_folder = self.page_source_folder / self.cur_filename
        self.cur_file_page_folder = self.page_output_folder / self.cur_filename

        self.cur_stats_summary_file = Path(f"{'pro_' if self.pro_dd else ''}{self.cur_filename}.tsv")
        self.cur_pdf_page_count = 0

        self.cur_plots_layout = []
        self.cur_plots_line = []
        self.cur_layouts_dd = []
        self.cur_json_stats = []

        self.cur_page_src = []
        # self.cur_pdf_src_paths = []

        # creating output directories

        if not self.layout_dd_output_folder.is_dir():
            self.layout_dd_output_folder.mkdir()

        if not self.page_stat_output_folder.is_dir():
            self.page_stat_output_folder.mkdir()

        if not self.pdf_stat_output_folder.is_dir():
            self.pdf_stat_output_folder.mkdir()

        if not self.page_output_folder.is_dir():
            self.page_output_folder.mkdir()

        if not self.page_source_folder.is_dir():
            self.page_source_folder.mkdir()

    # load from file sys current PDF's supplementary files
    def update_cur_vars(self, pdf_file: Path, next_file: bool = False) -> None:
        prefix = 'pro_' if self.pro_dd else ''

        if next_file:
            self.cur_filename = pdf_file.stem

            self.cur_file_page_src_folder = self.page_source_folder / self.cur_filename
            self.cur_file_page_folder = self.page_output_folder / self.cur_filename

            if not self.cur_file_page_src_folder.is_dir():
                self.cur_file_page_src_folder.mkdir()

            if not self.cur_file_page_folder.is_dir():
                self.cur_file_page_folder.mkdir()

            self.cur_stats_summary_file = self.pdf_stat_output_folder / f"{prefix}{self.cur_filename}_cred_{self.cred_min_tresh}.tsv"
            self.cur_pdf_page_count = fitz.open(pdf_file).page_count

        self.cur_layouts_dd = list(self.layout_dd_output_folder.glob(f"{prefix}{self.cur_filename}_page*"))
        self.cur_json_stats = list(self.page_stat_output_folder.glob(f"{prefix}{self.cur_filename}_page*"))
        self.cur_page_src = list(self.cur_file_page_src_folder.glob("*"))
        self.cur_plots_layout = list(self.cur_file_page_src_folder.glob("*"))

    # saving PDF as grayscale images in separate folder
    def check_page_sources(self, pdf_file: Path, save=True) -> None:
        if len(self.cur_page_src) == self.cur_pdf_page_count:
            print(f"[ IMG ] \t{self.cur_filename} has {len(self.cur_page_src)} source page(s)")
            return
        elif len(self.cur_page_src) != self.cur_pdf_page_count and len(self.cur_page_src) > 0:
            print(f"[ IMG ] \t{self.cur_filename} Incorrect page source number, restarting again")
            for sp in self.cur_file_page_src_folder.glob("*"):
                Path.unlink(sp)
        else:
            print(f"[ IMG ] \t{self.cur_filename} to {self.cur_pdf_page_count} page source files")

        self.cur_page_src = convert_from_path(str(pdf_file), output_folder=str(self.cur_file_page_src_folder),
                                                   grayscale=True, paths_only=True, fmt="png")
        print(f"[ +IMG ] \t{self.cur_filename} saved as {len(self.cur_page_src)} source page(s)")

        self.update_cur_vars(pdf_file)

    # saving PDF pages as dd layout JSONs and prediction image plots
    def check_page_layouts_dd(self, pdf_file: Path, save: bool = True) -> None:
        if len(self.cur_layouts_dd) == len(self.cur_json_stats) == self.cur_pdf_page_count and not self.force_stats:
            print(f"[ PDF ] \t{self.cur_filename} was already processed into {len(self.cur_json_stats)} stats and layouts")
            return
        elif len(self.cur_layouts_dd) == self.cur_pdf_page_count and len(self.cur_json_stats) != self.cur_pdf_page_count:
            print(f"[ JSON ] \t{self.cur_filename} has {len(self.cur_layouts_dd)} layout JSON(s)")
        elif len(self.cur_layouts_dd) == self.cur_pdf_page_count and self.force_stats:
            print(f"[ JSON ] \t{self.cur_filename} has {len(self.cur_layouts_dd)} layout JSON(s)")
        else:
            print(f"[ PDF ] {self.cur_filename} to {self.cur_pdf_page_count} page JSON stats and layout analysis...")

            if self.detector_dd is None:
                print(f" * * * Loading {'pro' if self.pro_dd else ''} DeepDoctection model...")
                self.detector_dd = dd.get_dd_analyzer(config_overwrite = dd_basic_config if not self.pro_dd else dd_pro_config)

            results = self.detector_dd.analyze(path=pdf_file)
            results.reset_state()

            pages = iter(results)
            for i, page in enumerate(pages):
                i += 1
                cur_page_layout_file = self.layout_dd_output_folder / f"{'pro_' if self.pro_dd else ''}{self.cur_filename}_page_{i}.json"

                ending = f"{generate_page_file_sufix(i, self.cur_pdf_page_count)}"
                cur_page_image_filename = Path(list(self.cur_file_page_src_folder.glob(f"*-{ending}.png"))[0])

                if save and not cur_page_image_filename.is_file():
                    save_parsed_image(page, cur_page_image_filename)

                if save and not cur_page_layout_file.is_file():
                    page.save(image_to_json=True, path=cur_page_layout_file)
                    print(f"[ + JSON ] \t{i}/{self.cur_pdf_page_count} page layout {self.layout_dd_output_folder}")

        self.update_cur_vars(pdf_file, self.cur_filename == "")

    # load PDF, go through page layouts and save them as JSON
    def pdf_to_json(self, pdf_file: Path, save: bool = True) -> None:
        self.update_cur_vars(pdf_file, self.cur_filename == "")

        self.check_page_sources(pdf_file, save)
        self.check_page_layouts_dd(pdf_file, save)

        pdf_text = ""
        out_prefix = 'pro_' if self.pro_dd else ''

        if len(self.cur_layouts_dd) == self.cur_pdf_page_count != len(self.cur_json_stats) or \
                (len(self.cur_layouts_dd) == self.cur_pdf_page_count and self.force_stats):

            page_numbers = [a for a in range(self.cur_pdf_page_count)]
            for pn in page_numbers:
                pn += 1

                ending = f"{generate_page_file_sufix(pn, self.cur_pdf_page_count)}"
                src_page_image_file = Path(list(self.cur_file_page_src_folder.glob(f"*-{ending}.png"))[0])
                cur_page_layout_file = self.layout_dd_output_folder / f"{out_prefix}{self.cur_filename}_page_{pn}.json"
                output_page_lines_file = self.page_output_folder / self.cur_filename / f"lines_{self.cur_filename}_page_{pn}.png"

                if not cur_page_layout_file.is_file():
                    time.sleep(2.5 if not self.pro_dd else 5)

                print(f"[ PDF ] \t{self.cur_filename}\t{pn + 1}/{self.cur_pdf_page_count} Page Layout Analysis...")

                page_text, page_json = page_layout_analysis(cur_page_layout_file,
                                                            src_page_image_file,
                                                            output_page_lines_file,
                                                            self.cred_min_tresh,
                                                            self.pro_dd)

                pdf_text += f"\n{pn}\n{page_text}"

                page_num = generate_page_file_sufix(pn, self.cur_pdf_page_count)
                with open(self.page_stat_output_folder / f"{out_prefix}{self.cur_filename}_page_{page_num}_cred_{self.cred_min_tresh}.json", 'w') as f:
                    json.dump(page_json, f)
                print(f"[ + JSON ] \t{self.cur_filename} {pn}/{self.cur_pdf_page_count} stats {self.page_stat_output_folder}")

            self.update_cur_vars(pdf_file)

        out_ocr_file = self.pdf_text_output_folder / f"{self.cur_filename}.txt"
        if save and not out_ocr_file.is_file():
            with open(out_ocr_file, "w") as text_file:
                text_file.write(pdf_text)
            print(f"[ + TXT ] \t{self.cur_filename} text saved in {self.pdf_text_output_folder}")

        table_summary = merge_stats(self.cur_json_stats,
                                    self.page_stat_output_folder,
                                    self.cred_min_tresh,
                                    self.pro_dd,
                                    no_categ=True)
        table_summary.to_csv(self.cur_stats_summary_file, sep="\t", index=False)
        print(f"[ + TAB ] \t{self.cur_filename} summary {self.pdf_stat_output_folder}")

    # load PDF, go through pages and save them as PNG
    def pdf_to_table(self, pdf_file: Path) -> pandas.DataFrame:
        self.update_cur_vars(pdf_file, True)

        if not self.cur_stats_summary_file.is_file() or self.force_stats:
            self.pdf_to_json(pdf_file)

        if self.cur_stats_summary_file.is_file() and not self.force_stats:
            # print(f"[ TSV ] \t{self.cur_file_name} has summary in the table format")
            return pandas.read_csv(self.cur_stats_summary_file, sep="\t")

        if not self.cur_stats_summary_file.is_file() or self.force_stats:
            print(f"[ JSON ] \t{self.cur_filename}Enough page stats found, summarizing...")
            table_summary = merge_stats(self.cur_json_stats,
                                        self.page_stat_output_folder,
                                        self.cred_min_tresh,
                                        self.pro_dd,
                                        no_categ=True)

            table_summary.to_csv(self.cur_stats_summary_file, sep="\t", index=False)
            print(f"[ + TAB ] \t{self.cur_filename} summary {self.pdf_stat_output_folder}")

            return table_summary

    # called to process directory path
    def folder_to_page_layouts(self, folder_path: Path) -> None:
        self.dir_pdf_file_list = directory_scraper(Path(folder_path), "pdf")

        for file_path in self.dir_pdf_file_list:
            self.pdf_to_table(Path(file_path))


# getting stats about particular element type from the layout elements
def el_type_parse(page: deepdoctection.Page,
                  layout: deepdoctection.Layout,
                  el_name: str,
                  el_out_dict: dict,
                  credibility: float = 0.0) -> (dict, bool):

    dd_name = None if not el_name else el_name
    dd_name = "title" if el_name == "HDR" else dd_name
    dd_name = "figure" if el_name == "IMG" else dd_name
    dd_name = "text" if el_name == "TXT" else dd_name

    wrong_table = False

    if layout.category_name == dd_name or layout.category_name == "list" and dd_name == "TXT":

        if layout.score < credibility:
            return el_out_dict

        print(f"{round(layout.score, 2)}\t{el_name}: {layout.text}\t{int(layout.bounding_box.area)}px")
        el_out_dict[el_name]["AREA"] += layout.bounding_box.area
        el_out_dict[el_name]["COUNT"] += 1

    elif el_name == "TAB" and len(page.tables) > 0:
        for j in range(len(page.tables)):
            tab = page.tables[j]

            if tab.score < credibility:
                continue

            non_empty_cells = []
            cells_n = 0
            for row in tab.csv:
                for row_col in row:
                    cells_n += 1
                    if any(char.isdigit() for char in row_col) or any(char.isalpha() for char in row_col):
                        non_empty_cells.append(row_col)
            # print(non_empty_cells)

            if len(non_empty_cells) < cells_n / 2:
                wrong_table = True

                print(f"{round(tab.score, 2)}\tTAB: {tab.text}\t{int(tab.bounding_box.area)}px")
                if len(non_empty_cells) > 0:

                    total_txt_area, total_txt_count = 0, 0

                    el_out_dict["IMG"]["AREA"] += 100
                    el_out_dict["IMG"]["COUNT"] += 1
                    print(f"[--]\tIMG: 100px")

                    for content_cell in non_empty_cells:
                        approx_area = 10 * len(content_cell)
                        el_out_dict["TXT"]["AREA"] += approx_area
                        el_out_dict["TXT"]["COUNT"] += 1
                        print(f"[--]\tTXT: {content_cell}\t{approx_area}px")

            else:
                print(f"{round(tab.score, 2)}\tTAB: {tab.text}\t{int(tab.bounding_box.area)}px")
                print(tab.csv)

                el_out_dict[el_name]["AREA"] += tab.bounding_box.area
                el_out_dict[el_name]["COUNT"] += 1

    # print({el_name}, el_out_dict[el_name])
    return el_out_dict, wrong_table


# saving page with recognized elements in color
def save_parsed_image(page: deepdoctection.Page, cur_page_image_filename: Path) -> None:
    image = page.viz()

    plt.figure(figsize=(16, 23))  # A4 paper ratio
    plt.axis('off')

    plt.imshow(image)
    plt.tight_layout()

    plt.savefig(cur_page_image_filename)
    print(f"[ + PNG ] \t{cur_page_image_filename.stem} layouts plot {cur_page_image_filename.parent}")

    plt.close()


# create page stat json
def page_layout_analysis(layout_filename: Path, image_filename: Path, output_filename: Path,
                         credibility: float = 0.0, dd_pro: bool = True ) -> (str, json):
    long_horiz, long_vert, pictures, long_n, short_n = page_visual_analysis(image_filename, output_filename)
    # print(long_horiz, long_vert)

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

    wrong_table_recognition = False
    for layout in page.layouts:
        found_el_stats, wrong_table_recognition = el_type_parse(page, layout, "TAB", found_el_stats, credibility)
        found_el_stats, _ = el_type_parse(page, layout, "HDR", found_el_stats, credibility)
        found_el_stats, _ = el_type_parse(page, layout, "TXT", found_el_stats, credibility)
        found_el_stats, _ = el_type_parse(page, layout, "IMG", found_el_stats, credibility)

    if wrong_table_recognition and pictures:
        gallery_area = int(page_area * 0.7)
        print(f"[--] \tIMG: \t{gallery_area}px")

        found_el_stats["IMG"]["AREA"] += gallery_area
        found_el_stats["IMG"]["COUNT"] += 2

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

                                    "long_l": long_n,
                                    "short_l": short_n

                                    }}, ensure_ascii=True)

    return page_text, j

