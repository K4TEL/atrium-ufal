import os.path
import time
import pandas
from common_utils import *

dd_basic_config = [
    "LANGUAGE='ces'",
    "USE_OCR=False",
    # "USE_LAYOUT=False",
    "USE_TABLE_SEGMENTATION=False",
    "USE_TABLE_REFINEMENT=False",
    "USE_PDF_MINER=False"
     ]
dd_pro_config = ["LANGUAGE='ces'"]

numerical = ["TXT", "IMG", "TAB"]
boolean = ["HDR", "Listed"]
all_boolean = ["HDR", "Listed", "Gallery", "Blank", "Manuscript"]

load_dotenv()

# PDF files to png pages parser
class Layout_sorter:
    def __init__(self, output_folder: Path = None, credibility: float = None, pro_ocr: bool = False):
        self.results_output_folder = Path(os.environ['FOLDER_RESULTS']) if output_folder is None else output_folder
        self.pdf_stat_input_folder = Path(os.environ['FOLDER_SUMMARY'])
        self.page_stat_input_folder = Path(os.environ['FOLDER_STATS'])

        self.cred_min_tresh = abs(credibility) if credibility is not None else credibility
        self.pro_dd = pro_ocr

        self.page_stat_list = []
        self.pdf_sum_list = []

        if not self.results_output_folder.is_dir():
            self.results_output_folder.mkdir()

    # merge pdf level stats into a single table
    def pdf_level_summary(self, pdf_sum_files: list[Path] = None) -> pandas.DataFrame:
        total_stats = []
        if pdf_sum_files is None:
            pdf_sum_files = self.pdf_sum_list
        for pdf_sum_name in pdf_sum_files:
            filename = self.pdf_stat_input_folder / pdf_sum_name

            pdf_cat_count = {cat_id: 0 for cat_id in category_id_map.keys()}
            page_count = 0

            if os.path.getsize(filename) > 1:
                pdf_stats = pd.read_csv(filename, sep="\t")
                pdf_stats_list = pdf_stats.to_dict(orient='records')

                for page_row in pdf_stats_list:
                    category_id, pdf_page_stats = predict_page_category(page_row)
                    pdf_cat_count[category_id] += 1
                    page_count += 1

                pdf_cat_count = {category_id_map[res_cat_id]: item for res_cat_id, item in pdf_cat_count.items()}

                pdf_cat_count["FILE"] = page_row["FILE"]
                pdf_cat_count["SIZE"] = page_count

                total_stats.append(pdf_cat_count)
            else:
                pdf_cat_count["FILE"] = str(filename)
                pdf_cat_count["SIZE"] = page_count

        total_stats = pd.DataFrame(total_stats).sort_values(by=["FILE"])
        # print(total_stats)
        print(total_stats.info())

        return total_stats

    # merge page level stats into a single table
    def page_level_summary(self, page_stat_files: list[Path] = None, rename: bool = False ) -> pandas.DataFrame:
        if page_stat_files is None:
            page_stat_files = self.page_stat_list

        table_summary = merge_stats(page_stat_files, self.page_stat_input_folder, self.cred_min_tresh, self.pro_dd)
        table_summary = table_summary.round(2)
        # print(table_summary)
        print(table_summary.info())

        return table_summary

    # called to process PDF file path
    def file_process_level(self, file_path: Path) -> pandas.DataFrame:
        file_name = file_path.stem
        files_prefix = f"{'pro_' if self.pro_dd else ''}{file_name}_page*"
        self.page_stat_list = [filepath for filepath in list(self.page_stat_input_folder.glob(files_prefix)) if
                               str(Path(filepath).stem).endswith(f"cred_{self.cred_min_tresh}")]

        results_table = self.page_level_summary()

        timestr = time.strftime("%Y%m%d-%H%M%S")
        output_file = self.results_output_folder / f"{'pro-' if self.pro_dd else ''}{timestr}-pages-{self.cred_min_tresh}-{file_name}.csv"
        results_table.to_csv(output_file, sep=",", index=False)
        print(f"[ + TAB ] \t{file_name} page level summary {self.results_output_folder}")
        return results_table

    # called to process directory path
    def folder_process_level(self, page: bool = True) -> pandas.DataFrame:
        if page:
            self.page_stat_list = self.selection(directory_scraper(self.page_stat_input_folder, "json"))
            results_table = self.page_level_summary()
        else:
            self.pdf_sum_list = self.selection(directory_scraper(self.pdf_stat_input_folder, "tsv"))
            results_table = self.pdf_level_summary()

        timestr = time.strftime("%Y%m%d-%H%M%S")
        output_file = self.results_output_folder / f"{'pro-' if self.pro_dd else ''}{timestr}-{'page' if page else 'pdf'}-result-{self.cred_min_tresh}.csv"
        results_table.to_csv(output_file, sep=",", index=False)
        print(f"[ + TAB ] \t{'page' if page else 'pdf'} summary {self.results_output_folder}")
        return results_table

    # filtering list of file paths by cred_min and pro_dd settings
    def selection(self, files_list: list) -> list:
        files_list = [filepath for filepath in files_list
                      if str(Path(filepath).stem).endswith(f"cred_{self.cred_min_tresh}")]
        if self.pro_dd:
            files_list = [filepath for filepath in files_list if str(Path(filepath).stem).startswith("pro")]
        return files_list


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




