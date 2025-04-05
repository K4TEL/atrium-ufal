from pathlib import Path
import json
import pandas as pd
import deepdoctection as dd
import argparse
import os
import configparser
import time

from opencv_line_detector import *

category_id_map = {
    1: "Form",
    2: "Form-figure",
    3: "Table",
    4: "Text-body",
    5: "Gallery",
    6: "Figure-text",
    7: "Table-text",
    8: "Headers-text",
    9: "Mixed",
    10: "Neither",
    11: "Empty-text"
}


# get list of all files in the folder and nested folders by file format
def directory_scraper(folder_path: Path, file_format: str = "png", file_list: list = None) -> list[str]:
    if file_list is None:
        file_list = []
    file_list += list(folder_path.rglob(f"*.{file_format}"))
    print(f"[ {file_format.upper()} ] \tFrom directory {folder_path} collected {len(file_list)} {file_format} files")
    return file_list


# from int get string
def generate_page_file_sufix(cur_page: int, total_n: int) -> str:
    if total_n < 100:
        ending = f"0{cur_page}" if cur_page <= 9 < total_n else str(cur_page)
    else:
        ending = str(cur_page)
        if cur_page <= 9:
            ending = f"00{cur_page}"
        elif cur_page <= 99:
            ending = f"0{cur_page}"
    return ending


# parse file path to PDF filename and page number
def filename_parse(file_path: Path, credibility: bool = True, pro_dd: bool = True) -> (str, str):
    stat_filename = file_path.stem.split("_")

    filename = "_".join(stat_filename[1 if pro_dd else 0: -4 if credibility else -2]) if len(stat_filename) > 5 \
        else stat_filename[1 if pro_dd else 0]
    page_num = stat_filename[-3 if credibility else -1]

    return filename, page_num


# merge page stats to pdf summary table dataframe
def merge_stats(stats: list, stat_output_folder: Path, credibility: float, pro_dd: bool,
                no_categ: bool = False) -> pd.DataFrame:
    pdf_pages = []
    pdf_stats = selection(stats, credibility, pro_dd)
    for page_stat_name in pdf_stats:
        with open(stat_output_folder / page_stat_name, 'r') as f:
            json_data = json.load(f)

        json_data = json.loads(json_data)
        json_data['FIG']['FILE'], json_data['FIG']['PAGE'] = filename_parse(page_stat_name, credibility is not None)
        if not no_categ:
            cat_id, json_data["FIG"] = predict_page_category_analysis(json_data["FIG"])
        pdf_pages.append(json_data["FIG"])

    return pd.DataFrame.from_records(pdf_pages).sort_values(by=['FILE', 'PAGE'])


# manual algo to determine 1 of 10 category of the page by its JSON stats               ( OLD - DONT USE)
def predict_page_category(input_dict: dict) -> (int, dict):
    category_id = 10

    # print(input_dict)

    # for k, v in input_dict.items():
    #     print(k, v)

    img_line_count_tresh = 500

    if input_dict["TAB"] > 0.9:
        category_id = 1 # form

    elif input_dict["TAB"] > 0 and input_dict["IMG"] > 0:
        category_id = 2 # table-fig

    elif input_dict["TAB"] == 1 and input_dict["Form"]:
        category_id = 3 # table

    elif input_dict["Manuscript"]:
        category_id = 4 # text-body

    elif (input_dict["IMG"] > 0.9 or input_dict["IMGs"] or input_dict["Gallery"]) and \
            (input_dict["long_l"] > img_line_count_tresh or input_dict["short_l"] > img_line_count_tresh) and \
            input_dict["IMG"] > input_dict["TXT"]:
        category_id = 5 # gallery

    elif input_dict["IMG"] > 0:
        if input_dict["TXT"] > 0 and input_dict["HDR"] > 0:
            category_id = 6 # fig-text

    elif input_dict["TAB"] > 0:
        if input_dict["HDR"] > 0 or input_dict["TXT"] > 0:
            category_id = 7 # table-text

    elif input_dict["HDRs"] and input_dict["TXT"] > 0:
        category_id = 8 # title-text

    elif input_dict["H_line"] and input_dict["V_line"] and not input_dict["TXT"] > 0.9:
        category_id = 3  # table

    elif (input_dict["H_line"] or input_dict["V_line"]) and input_dict["long_l"] > img_line_count_tresh and \
            input_dict["short_l"] > img_line_count_tresh:
        category_id = 5  # gallery

    elif input_dict["H_line"] and not input_dict["TXT"] > 0.9 and \
            (input_dict["long_l"] > img_line_count_tresh or input_dict["short_l"] > img_line_count_tresh):
        category_id = 2 # table-fig

    elif (input_dict["H_line"] or input_dict["V_line"]) and \
            input_dict["long_l"] > img_line_count_tresh and input_dict["short_l"] > img_line_count_tresh:
        category_id = 1 # form

    elif input_dict["H_line"] and not input_dict["TXT"] > 0.9:
        category_id = 1 # form

    elif not input_dict["H_line"] and input_dict["V_line"] and input_dict["TXT"] > 0.9:
        category_id = 3  # table

    elif any([input_dict["IMG"], input_dict["HDR"], input_dict["TXT"], input_dict["TAB"]]) > 0:
        category_id = 9 # mixed

    # input_dict["TXT"] = input_dict["TXT"] / input_dict["area"]
    # input_dict["IMG"] = input_dict["IMG"] / input_dict["area"]
    # input_dict["TAB"] = input_dict["TAB"] / input_dict["area"]

    input_dict["Category_ID"] = category_id
    input_dict["Category"] = category_id_map[category_id]

    return category_id, input_dict


# manual algo to determine 1 of 11 category of the page by its JSON stats
def predict_page_category_analysis(input_dict: dict) -> (int, dict):
    category_id = 10

    # print(input_dict)

    # for k, v in input_dict.items():
    #     print(k, v)

    empty_line_count_tresh = 100
    img_line_count_tresh = 500

    short_line_form_tresh = 1000

    short_line_gallery_tresh = 3000
    long_line_gallery_tresh = 1000

    def not_pictures(in_dict):
        return in_dict["long_l"] < img_line_count_tresh and in_dict["short_l"] < short_line_form_tresh

    def maybe_picture(in_dict):
        return in_dict["long_l"] > img_line_count_tresh or in_dict["short_l"] > img_line_count_tresh

    def indeed_picture(in_dict):
        return in_dict["long_l"] > long_line_gallery_tresh and in_dict["short_l"] > short_line_gallery_tresh

    def maybe_galley(in_dict):
        return in_dict["long_l"] > long_line_gallery_tresh or in_dict["short_l"] > short_line_gallery_tresh

    def empty_page(in_dict):
        return in_dict["long_l"] < empty_line_count_tresh and in_dict["short_l"] < empty_line_count_tresh

    if input_dict["TXT"] > 0:
        if input_dict["TXT"] > 0.9:

            if input_dict["Manuscript"]:
                category_id = 4 # text-body
            else:
                if input_dict["H_line"] and not_pictures(input_dict) and input_dict["TXT_N"] > 1:
                    category_id = 1  # form
                elif 0 < input_dict["HDR"] < input_dict["TXT"] and input_dict["content"] > 0.1:
                    category_id = 4  # text-body
                elif input_dict["content"] < 0.2 and maybe_picture(input_dict):
                    category_id = 6  # fig-text
                else:
                    category_id = 4  # text-body

        else:
            if 0 < input_dict["IMG"] < input_dict["TXT"]:
                category_id = 6  # fig-text
            elif 0 < input_dict["TXT"] < input_dict["IMG"] and maybe_picture(input_dict):
                category_id = 6  # fig-text
            elif input_dict["TXT"] < 0.2 < input_dict["IMG"] and maybe_galley(input_dict):
                category_id = 5  # gallery
            elif 0 < input_dict["TAB"] < input_dict["TXT"]:
                category_id = 7  # table-text
            elif 0 < input_dict["TXT"] < input_dict["TAB"]:
                category_id = 1  # form
            elif input_dict["TXT"] < input_dict["HDR"]:
                category_id = 8  # title-text
            elif input_dict["H_line"] and maybe_galley(input_dict):
                category_id = 5  # gallery
            elif input_dict["H_line"] and maybe_picture(input_dict):
                category_id = 2  # table-fig
            elif (input_dict["H_line"] or input_dict["V_line"]) and \
                    input_dict["TXT"] < 0.2 and not empty_page(input_dict):
                category_id = 5  # gallery
            elif (input_dict["H_line"] or input_dict["V_line"]) and \
                    input_dict["content"] < 0.1 and not empty_page(input_dict):
                category_id = 5  # gallery
            elif input_dict["content"] < 0.1 and not empty_page(input_dict):
                category_id = 5  # gallery

    else:
        if input_dict["H_line"] or input_dict["V_line"] and \
                not any([input_dict["IMG"], input_dict["HDR"], input_dict["TXT"], input_dict["TAB"]]) > 0:
            if not_pictures(input_dict) and not maybe_picture(input_dict):
                category_id = 3  # table
            elif maybe_galley(input_dict):
                category_id = 5  # gallery
            elif maybe_picture(input_dict):
                category_id = 2  # table-fig
            else:
                category_id = 5  # gallery

        elif input_dict["TAB"] > 0 and input_dict["IMG"] > 0:
            category_id = 2 # table-fig
        elif input_dict["TAB"] == 1 and input_dict["Form"]:
            category_id = 3  # table
        elif input_dict["TAB"] > 0.9 and input_dict["Form"]:
            category_id = 1 # form
        elif input_dict["content"] < 0.1 and not empty_page(input_dict):
            category_id = 5  # gallery
        elif any([input_dict["IMG"], input_dict["HDR"], input_dict["TXT"], input_dict["TAB"]]) > 0:
            category_id = 9 # mixed

    if empty_page(input_dict):
        category_id = 11  # text-empty
    if indeed_picture(input_dict):
        category_id = 5  # gallery
    if input_dict["Gallery"]:
        category_id = 5  # gallery
    if maybe_picture(input_dict):
        if not input_dict["Manuscript"] and not input_dict["Form"]:
            if (input_dict["content"] > 0.5 and input_dict["TXT"] > input_dict["IMG"]) or \
                    (input_dict["content"] < 0.5 and input_dict["IMG"] > input_dict["TXT"]):
                category_id = 5  # gallery
    if input_dict["content"] > 0.2 and input_dict["IMG"] > 0.9:
        category_id = 5  # gallery

    input_dict["Category_ID"] = category_id
    input_dict["Category"] = category_id_map[category_id]

    return category_id, input_dict



# filtering list of file paths by cred_min and pro_dd settings
def selection(files_list: list, cred_min: float, pro_dd: bool) -> list:
    files_list = [filepath for filepath in files_list
                  if str(Path(filepath).stem).endswith(f"cred_{cred_min}")]
    if pro_dd:
        files_list = [filepath for filepath in files_list if str(Path(filepath).stem).startswith("pro")]
    return files_list





# getting stats about particular element type from the layout elements
def el_type_parse(page: dd.Page,
                  layout: dd.Layout,
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