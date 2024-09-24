import os.path
import time

import fitz
import pandas
import pandas as pd
from common_utils import *
import deepdoctection as dd
# from IPython.core.display import HTML
# from matplotlib import pyplot as plt
import json

dd_basic_config = ["LANGUAGE='ces'",
                     "USE_OCR=False",
                     # "USE_LAYOUT=False",
                     "USE_TABLE_SEGMENTATION=False",
                     "USE_TABLE_REFINEMENT=False",
                     "USE_PDF_MINER=False"
                     ]
dd_pro_config = ["LANGUAGE='ces'"]


load_dotenv()


# PDF files to png pages parser
class PDF_parser:
    def __init__(self, output_folder: Path = Path(os.environ['FOLDER_LAYOUTS_DD'])):
        self.layout_output_folder = output_folder
        self.stat_output_folder = Path(os.environ['FOLDER_STATS'])
        self.pdf_output_folder = Path(os.environ['FOLDER_SUMMARY'])

        self.pdf_file_list = []

        self.detector_basic = None
        self.detector_pro = None

        self.cur_file_name = ""
        self.cur_stats_summary_file = f"{self.cur_file_name}.tsv"
        self.cur_pdf_page_count = 0

        self.cur_pages = []
        self.cur_layouts = []
        self.cur_stats = []

        # if not os.path.exists(self.layout_output_folder):
        #     os.makedirs(self.layout_output_folder)
        #
        # if not os.path.exists(self.stat_output_folder):
        #     os.makedirs(self.stat_output_folder)
        #
        # if not os.path.exists(self.pdf_output_folder):
        #     os.makedirs(self.pdf_output_folder)

        if not self.layout_output_folder.is_dir():
            self.layout_output_folder.mkdir()

        if not self.stat_output_folder.is_dir():
            self.stat_output_folder.mkdir()

        if not self.pdf_output_folder.is_dir():
            self.pdf_output_folder.mkdir()


    # load from file sys current PDF's supplementary files
    def update_cur_vars(self, pdf_file: Path, next_file: bool = False) -> None:
        if next_file:
            # self.cur_file_name = pdf_file.split("/")[-1].split(".")[0]
            self.cur_file_name = pdf_file.stem
            self.cur_stats_summary_file = self.pdf_output_folder / f"{self.cur_file_name}.tsv"
            self.cur_pdf_page_count = fitz.open(pdf_file).page_count

        self.cur_layouts = list(self.layout_output_folder.glob(f"{self.cur_file_name}_page"))
        self.cur_stats = list(self.stat_output_folder.glob(f"{self.cur_file_name}_page"))
        # self.cur_stats = [page_file for page_file in os.listdir(self.stat_output_folder) if
        #                   page_file.startswith(f"{self.cur_file_name}_page")]

    # load PDF, go through page layouts and save them as JSON
    def pdf_to_json(self, pdf_file: Path, save=True) -> None:
        # pdf_layout_prefix = self.layout_output_folder / self.cur_file_name
        # pdf_stat_prefix = self.stat_output_folder / self.cur_file_name

        self.update_cur_vars(pdf_file, self.cur_file_name == "")

        if len(self.cur_stats) == self.cur_pdf_page_count:
            print(f"[ JSON ] \t{self.cur_file_name} has {len(self.cur_stats)} page stat JSON(s)")
        elif len(self.cur_layouts) == self.cur_pdf_page_count and len(self.cur_stats) != self.cur_pdf_page_count:
            print(f"[ JSON ] \t{self.cur_file_name} has {len(self.cur_layouts)} layout JSON(s)")
        else:
            print(f"{self.cur_file_name} to {self.cur_pdf_page_count} page JSON stats and layout analysis...")
            if self.detector_basic is None:
                print("Loading DeepDoctection model...")
                self.detector_basic = dd.get_dd_analyzer(config_overwrite=dd_basic_config)

            # if self.detector_pro is None:
            #     self.detector_pro = dd.get_dd_analyzer(config_overwrite=dd_pro_config)
            results = self.detector_basic.analyze(path=pdf_file)
            results.reset_state()

            pages = iter(results)
            for i, page in enumerate(pages):
                i += 1
                cur_page_filename = self.layout_output_folder / f"{self.cur_file_name}_page_{i}.json"
                if save and not cur_page_filename.is_file():
                    page.save(image_to_json=True, path=cur_page_filename)
                    print(f"[ + JSON ] \t{i}/{self.cur_pdf_page_count} page layout {self.layout_output_folder}")

        self.update_cur_vars(pdf_file)
        if len(self.cur_layouts) == self.cur_pdf_page_count:
            page_numbers = [a for a in range(self.cur_pdf_page_count)]
            for pn in page_numbers:
                pn += 1
                # if not os.path.isfile(f"{pdf_layout_prefix}_page_{pn}.json"):
                #     time.sleep(2.5)
                cur_page_filename = self.layout_output_folder / f"{self.cur_file_name}_page_{pn}.json"
                if not cur_page_filename.is_file():
                    time.sleep(2.5)
                page = dd.Page.from_file(file_path=cur_page_filename)
                page_area = page.height * page.width
                print(f"IMG\t{page_area}")
                print(f"IMG\t{pn + 1}/{self.cur_pdf_page_count} Page Layout Analysis...")

                has_image = False
                image_area = 0

                has_text = False
                text_area = 0

                has_table = False
                table_area = 0

                has_header = False
                has_list = False

                if len(page.tables) > 0:
                    has_table = True

                    for j in range(len(page.tables)):
                        tab = page.tables[j]
                        # HTML(table.HTML)
                        print(f"{tab.score}\tTable: {tab.bounding_box.area}")
                        table_area += tab.bounding_box.area

                    # print(table.csv)
                    print("TAB", has_table, table_area, table_area / page_area)

                for layout in page.layouts:
                    if layout.category_name == "title":
                        has_header = True
                        print(f"{layout.score}\tTitle: {layout.text}\t{layout.bounding_box.area}")
                        print("HDR", has_header)

                    if layout.category_name == "text":
                        has_text = True
                        print(f"{layout.score}\tText: {layout.text}\t{layout.bounding_box.area}")
                        text_area += layout.bounding_box.area
                        print("TXT", has_text, text_area, text_area / page_area)

                    if layout.category_name == "figure":
                        has_image = True
                        print(f"{layout.score}\tFigure: {layout.bounding_box.area}")
                        image_area += layout.bounding_box.area
                        print("FIG", has_image, image_area, image_area / page_area)

                    if layout.category_name == "list":
                        has_list = True
                        print(f"{layout.score}\tList: {layout.text}\t{layout.bounding_box.area}")

                is_large_table = table_area > (page_area / 2)
                is_large_image = image_area > (page_area / 2)
                is_large_text = text_area > (page_area / 2)

                j = json.dumps({'FIG': {"area": page_area,
                                        "TAB": table_area,
                                        "Blank": is_large_table,
                                        "TXT": text_area,
                                        "Manuscript": is_large_text,
                                        "IMG": image_area,
                                        "Gallery": is_large_image,
                                        "Listed": has_list,
                                        "HDR": has_header}}, ensure_ascii=True)

                with open(self.stat_output_folder / f"{self.cur_file_name}_page_{pn}.json", 'w') as f:
                    json.dump(j, f)
                print(f"[ + JSON ] \t{self.cur_file_name} {pn}/{self.cur_pdf_page_count} stats {self.stat_output_folder}")

            # if render:
            #     image = page.viz()
                # plt.figure(figsize=(25, 17))
                # plt.axis('off')
                # imgplot = plt.imshow(image)
                # plt.show()

        self.update_cur_vars(pdf_file)
        table_summary = merge_stats(self.cur_stats, self.stat_output_folder)
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
            table_summary = merge_stats(self.cur_stats, self.stat_output_folder)

            table_summary.to_csv(self.cur_stats_summary_file, sep="\t", index=False)
            print(f"[ + TAB ] \t{self.cur_file_name} summary {self.pdf_output_folder}")

            return table_summary

    # called to process directory path
    def folder_to_page_layouts(self, folder_path: Path) -> None:
        self.pdf_file_list = directory_scraper(Path(folder_path), "pdf")

        for file_path in self.pdf_file_list:
            self.pdf_to_table(Path(file_path))


# merge page stats to pdf summary table
def merge_stats(stats: list, stat_output_folder: Path) -> pandas.DataFrame:
    pdf_pages = []
    for page_stat_name in stats:
        with open(stat_output_folder / page_stat_name, 'r') as f:
            json_data = json.load(f)

        json_data = json.loads(json_data)
        # json_data['FIG']['FILE'] = page_stat_name.split("_")[0]
        # json_data['FIG']['PAGE'] = page_stat_name.split("_")[-1].split(".")[0]

        stat_filename = page_stat_name.stem.split("_")

        json_data['FIG']['FILE'] = stat_filename[0]
        json_data['FIG']['PAGE'] = stat_filename[-1]

        pdf_pages.append(json_data["FIG"])

    table_summary = pd.DataFrame.from_records(pdf_pages)

    table_summary = table_summary.sort_values(by=['FILE', 'PAGE'])

    print(table_summary)
    return table_summary
