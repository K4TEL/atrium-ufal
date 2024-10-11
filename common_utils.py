from dotenv import load_dotenv
from pathlib import Path
import json
import pandas as pd

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

