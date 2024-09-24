from dotenv import load_dotenv
import os
from pathlib import Path

# get list of all files in the folder and nested folders by file format
def directory_scraper(folder_path: Path, format: str ="png", file_list: list = None) -> list[str]:
    if file_list is None:
        file_list = []

    # dir_path = Path(folder_path)
    # file_list += list(folder_path.glob(f"*.{format}"))
    file_list += list(folder_path.rglob(f"*.{format}"))

    # for root, dirs, files in os.walk(folder_path):
    #     for file in files:
    #         if file.endswith(f".{format}"):
    #             file_list.append(f"{root}/{file}")

    print(f"From directory {folder_path} collected {len(file_list)} {format} files")
    return file_list