from common_utils import *

dd_basic_config = [
    "LANGUAGE='ces'",
    "USE_OCR=False",
    "USE_LAYOUT=False",
    "USE_TABLE_SEGMENTATION=False",
    "USE_TABLE_REFINEMENT=False",
    "USE_PDF_MINER=False"
     ]

dd_pro_config = ["LANGUAGE='ces'",
                 "USE_OCR=False",
                 "USE_LAYOUT=True",
                 "USE_TABLE_SEGMENTATION=True",
                 "USE_TABLE_REFINEMENT=True",
                 "USE_PDF_MINER=True",
                 ]


class ImageClassifier:
    def __init__(self, num_labels: int, pro: bool, credibility: float, out_folder: Path):
        """
        Initialize the image classifier with the specified checkpoint.
        """
        print(f" * * * Loading {'pro' if pro else ''} DeepDoctection model...")
        self.detector_dd = dd.get_dd_analyzer(config_overwrite=dd_basic_config if not pro else dd_pro_config)

        self.pro = pro
        self.cred = credibility
        self.num_labels = num_labels
        self.output_folder = out_folder

    def pdf_to_json(self, directory_path: Path, save: bool = True) -> None:
        results = self.detector_dd.analyze(path=directory_path)

        results.reset_state()
        pages = iter(results)
        for i, page in enumerate(pages):
            image_name = Path(page.file_name).stem
            name_parts = image_name.split("-")
            page_num = int(name_parts[-1])
            document = name_parts[0] if len(name_parts) == 2 else "-".join(name_parts[:-1])

            print(document, page_num)

            layout_file = f"{document}-{page_num}.json"
            cur_page_layout_file = self.output_folder / "layouts_dd" / layout_file

            if save and not cur_page_layout_file.is_file():
                page.save(path=cur_page_layout_file, image_to_json=False)
                print(f"Saved page layout to {cur_page_layout_file}")

        print(f" * * * Finished processing {directory_path} with DeepDoctection")

    def post_process(self) -> pd.DataFrame:
        page_results = []
        dd_layouts = directory_scraper(self.output_folder / "layouts_dd", "json")
        for cur_layout in dd_layouts:
            with open(cur_layout, 'r') as f:
                json_data = json.load(f)

            image_name = Path(json_data["file_name"]).stem
            name_parts = image_name.split("-")
            page_num = int(name_parts[-1])
            document = name_parts[0] if len(name_parts) == 2 else "-".join(name_parts[:-1])

            in_image_file = Path(json_data["location"])
            out_image_file = self.output_folder / "pages" / f"{document}-{page_num}.png"

            print(document, page_num)

            layout_file = f"{document}-{page_num}.json"

            cur_stat_layout_file = self.output_folder / "layout_stat" / layout_file
            if not cur_stat_layout_file.is_file():
                page_text, page_json = page_layout_analysis(Path(cur_layout),
                                                            in_image_file,
                                                            out_image_file,
                                                            self.cred,
                                                            self.pro)
                print(page_json)
                page_json = json.loads(page_json)
                with open(cur_stat_layout_file, 'w') as f:
                    json.dump(page_json, f)
                print(f"Saved page stat to {cur_stat_layout_file}")
            else:
                with open(cur_stat_layout_file, 'r') as f:
                    page_json = json.load(f)

            cat_id, json_stat = predict_page_category_analysis(page_json["FIG"])
            json_stat["FILE"] = document
            json_stat["PAGE"] = page_num

            print(json_stat)

            page_results.append(json_stat)

        total_stats = pd.DataFrame(page_results).sort_values(by=["FILE", "PAGE"])
        return total_stats

    def merge_results(self):
        page_results = []
        dd_stats = directory_scraper(self.output_folder / "layout_stat", "json")
        for cur_stat in dd_stats:
            with open(cur_stat, 'r') as f:
                json_data = json.load(f)

            image_name = Path(json_data["file_name"]).stem
            name_parts = image_name.split("-")
            page_num = int(name_parts[-1])
            document = name_parts[0] if len(name_parts) == 2 else "-".join(name_parts[:-1])

            print(document, page_num)

            json_stat = json_data["FIG"]
            json_stat["FILE"] = document
            json_stat["PAGE"] = page_num

            print(json_stat)

            page_results.append(json_stat)

        total_stats = pd.DataFrame(page_results).sort_values(by=["FILE", "PAGE"])
        return total_stats












