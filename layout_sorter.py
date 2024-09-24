import os.path
import time
import pandas
import pandas as pd
from common_utils import *

import json

dd_basic_config = [
    "LANGUAGE='ces'",
    "USE_OCR=False",
    # "USE_LAYOUT=False",
    "USE_TABLE_SEGMENTATION=False",
    "USE_TABLE_REFINEMENT=False",
    "USE_PDF_MINER=False"
     ]
dd_pro_config = ["LANGUAGE='ces'"]

categories = {
    1: "Form",
    2: "Text-body",
    3: "Gallery",
    4: "Table-text",
    5: "Figure-text",
    6: "Headers-text",
    7: "Listed-text",
    8: "Listed-title",
    9: "Figure-legend",
    10: "Figure-title",
    11: "Form-figure",
    12: "Form-small",
    13: "Table-title",
    14: "Mixed",
    15: "Neither"
}

numerical = ["TXT", "IMG", "TAB"]
boolean = ["HDR", "Listed"]
all_boolean = ["HDR", "Listed", "Gallery", "Blank", "Manuscript"]

load_dotenv()


# manual algo to determine 1 of 15 category of the page by its JSON stats
def predict_category(input_dict: dict) -> (int, dict):
    category_id = 0

    if input_dict["Listed"] and input_dict["HDR"]:
        category_id = 8
    elif input_dict["Listed"] and input_dict["IMG"] > 0:
        category_id = 9
    elif input_dict["Listed"]:
        category_id = 7
    elif input_dict["Manuscript"] and input_dict["HDR"]:
        category_id = 6
    elif input_dict["Manuscript"]:
        category_id = 2
    elif input_dict["Gallery"] and input_dict["TXT"] > 0:
        category_id = 5
    elif input_dict["Gallery"] and input_dict["HDR"]:
        category_id = 10
    elif input_dict["Gallery"]:
        category_id = 3
    elif input_dict["TAB"] > 0 and input_dict["TXT"] > 0:
        category_id = 4
    elif input_dict["TAB"] > 0 and input_dict["HDR"]:
        category_id = 13
    elif input_dict["Blank"] and input_dict["IMG"] > 0:
        category_id = 11
    elif input_dict["Blank"] and check_other_zeros(input_dict, "TAB", "Blank"):
        category_id = 1
    elif input_dict["TAB"] > 0 and check_other_zeros(input_dict, "TAB"):
        category_id = 12
    elif input_dict["IMG"] > 0 and input_dict["HDR"]:
        category_id = 10
    elif input_dict["IMG"] > 0 and input_dict["TAB"] > 0:
        category_id = 9
    elif input_dict["IMG"] > 0 and input_dict["TXT"] > 0:
        category_id = 5
    elif input_dict["TXT"] > 0 and check_other_zeros(input_dict, "TXT"):
        category_id = 2
    elif input_dict["IMG"] > 0 and check_other_zeros(input_dict, "IMG"):
        category_id = 3

    if category_id == 0:
        category_id = 15

    input_dict["TXT"] = input_dict["TXT"] / input_dict["area"]
    input_dict["IMG"] = input_dict["IMG"] / input_dict["area"]
    input_dict["TAB"] = input_dict["TAB"] / input_dict["area"]

    input_dict["Category_ID"] = category_id
    input_dict["Category"] = categories[category_id]

    return category_id, input_dict


# PDF files to png pages parser
class Layout_sorter:
    def __init__(self, output_folder: Path = None):
        self.results_output_folder = Path(os.environ['FOLDER_RESULTS']) if output_folder is None else output_folder
        self.layout_input_folder = Path(os.environ['FOLDER_SUMMARY'])
        self.stat_input_folder = Path(os.environ['FOLDER_STATS'])

        self.page_stat_list = []
        self.pdf_stat_list = []

        if not self.results_output_folder.is_dir():
            self.results_output_folder.mkdir()

    # merge pdf level stats into a single table
    def pdf_level_summary(self, pdf_stat_files: list[Path] = None) -> pandas.DataFrame:
        total_stats = []
        if pdf_stat_files is None:
            pdf_stat_files = self.pdf_stat_list
        for pdf_stat_name in pdf_stat_files:
            filename = self.stat_input_folder / pdf_stat_name

            pdf_cat_count = {cat_id: 0 for cat_id in categories.keys()}
            page_count = 0

            if os.path.getsize(filename) > 1:
                pdf_stats = pd.read_csv(filename, sep="\t")
                pdf_stats_list = pdf_stats.to_dict(orient='records')

                for page_row in pdf_stats_list:
                    category_id, pdf_page_stats = predict_category(page_row)
                    pdf_cat_count[category_id] += 1
                    page_count += 1

                pdf_cat_count = {categories[res_cat_id]: item for res_cat_id, item in pdf_cat_count.items()}

                pdf_cat_count["FILE"] = page_row["FILE"]
                pdf_cat_count["SIZE"] = page_count

                total_stats.append(pdf_cat_count)
            else:
                pdf_cat_count["FILE"] = str(filename)
                pdf_cat_count["SIZE"] = page_count

        total_stats = pd.DataFrame(total_stats)
        print(total_stats)
        print(total_stats.info())

        return total_stats

    # merge page level stats into a single table
    def page_level_summary(self, page_stat_files: list[Path] = None) -> pandas.DataFrame:
        if page_stat_files is None:
            page_stat_files = self.page_stat_list

        pdf_pages = []
        for page_stat_name in page_stat_files:
            with open(self.stat_input_folder / page_stat_name, 'r') as f:
                json_data = json.load(f)

            json_data = json.loads(json_data)

            stat_filename = page_stat_name.stem.split("_")
            json_data['FIG']['FILE'] = stat_filename[0]
            json_data['FIG']['PAGE'] = stat_filename[-1]

            category_id, json_data["FIG"] = predict_category(json_data["FIG"])

            del json_data['FIG']['Blank'], json_data['FIG']['Manuscript'], json_data['FIG']['area'], \
                json_data['FIG']['HDR'], json_data['FIG']['Listed'], json_data['FIG']['Gallery']
            pdf_pages.append(json_data["FIG"])

        table_summary = pd.DataFrame.from_records(pdf_pages)
        table_summary = table_summary.round(2)
        print(table_summary)
        print(table_summary.info())

        return table_summary

    # called to process directory path
    def folder_process_level(self, page: bool = True) -> pandas.DataFrame:
        if page:
            self.page_stat_list = directory_scraper(self.stat_input_folder, "json")
            results_table = self.page_level_summary()
            results_table = results_table.sort_values(by=['FILE', 'PAGE'])
        else:
            self.pdf_stat_list = directory_scraper(self.layout_input_folder, "tsv")
            results_table = self.pdf_level_summary()
            results_table = results_table.sort_values(by=["FILE"])

        timestr = time.strftime("%Y%m%d-%H%M%S")
        results_table.to_csv(self.results_output_folder / f"{timestr}-{'page' if page else 'pdf'}-result.csv",
                             sep=",", index=False)
        print(f"[ + TAB ] \t{'page' if page else 'pdf'} summary {self.results_output_folder}")
        return results_table


# check that other numerical fields are zero and boolean fields are false
def check_other_zeros(json_dict: dict, num_val: str = None, bool_val: str = None) -> bool:
    zeros = True
    if num_val is not None:
        for name in numerical:
            if num_val == name:
                continue
            else:
                if json_dict[name] > 0:
                    zeros = False
    if bool_val is not None:
        for name in boolean:
            if bool_val == name:
                continue
            else:
                if json_dict[name]:
                    zeros = False
    return zeros



