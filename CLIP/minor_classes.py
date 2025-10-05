import h5py
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import torchvision
from torchvision import transforms
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter

from huggingface_hub import PyTorchModelHubMixin

from collections import defaultdict

Image.MAX_IMAGE_PIXELS = 700_000_000
import csv

from utils import *


class ImageFolderCustom(torch.utils.data.Dataset):
    """
    Custom Dataset for loading images from a directory and assigning category labels.
    Limits the number of samples per category if specified.
    """

    def __init__(self, targ_dir: str, seed: int, max_category_samples: int | None,
                 img_size: int, preprocess_fn: callable = None, model_name: str = "ViT",
                 ignore_dir: str = None, use_advanced_split: bool = True, split_type: str = 'train',
                 file_format: str = "png", test_ratio: float = 0.1) -> None:
        self.targ_dir = targ_dir
        self.max_category_samples = max_category_samples
        self.preprocess = preprocess_fn
        self.size = img_size
        self.use_advanced_split = use_advanced_split
        self.split_type = split_type
        self.classes = []
        self.class_to_idx = {}
        self.paths = []
        self.targets = []
        self.seed_random = seed
        self.file_format = file_format
        self.eval_ratio = test_ratio

        all_categories = sorted(entry.name for entry in os.scandir(targ_dir) if entry.is_dir())

        if not all_categories:
            raise FileNotFoundError(f"Couldn't find any classes in {targ_dir}.")

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(all_categories)}
        self.classes = all_categories

        print(f"Discovered categories: {self.classes}")

        for category_name in self.classes:
            category_path = Path(targ_dir) / category_name
            all_files_in_category = list(category_path.glob(f"*.{self.file_format}"))

            if ignore_dir is not None and os.path.exists(ignore_dir):
                ignored_category = Path(ignore_dir) / category_name
                ignored_files = list(ignored_category.glob(f"*.{self.file_format}"))
                all_files_in_category = [f for f in all_files_in_category if f not in ignored_files]

            # Sort files for temporal consistency
            all_files_in_category = sorted(all_files_in_category)

            if self.max_category_samples is not None and len(all_files_in_category) > self.max_category_samples:
                random.seed(self.seed_random)
                random.shuffle(all_files_in_category)
                all_files_in_category = all_files_in_category[:self.max_category_samples]

            class_idx = self.class_to_idx[category_name]
            for file_path in all_files_in_category:
                self.paths.append(file_path)
                self.targets.append(class_idx)

        # Apply splitting strategy
        if self.use_advanced_split:
            from classifier import CLIP, split_data_80_10_10

            print(f"Using advanced split_data_80_10_10 for {split_type} set")
            train_files, val_files, test_files, train_labels, val_labels, test_labels = split_data_80_10_10(
                files=self.paths,
                labels=self.targets,
                random_seed=self.seed_random,
                max_categ=self.max_category_samples if self.max_category_samples else 15000,
                safe_check=True,
            )

            time_stamp = time.strftime("%Y%m%d-%H%M")
            # record datasets
            out_filename = f"result/stats/{time_stamp}_{model_name}_{self.seed_random}_DATASETS.txt"
            features =  "_".join(out_filename.split("_")[1:])

            # check whether a file with the same ending exists already or not
            existing_files = list(Path("result/stats").glob(f"*_{features}"))
            if len(existing_files) > 0:
                print(f"Warning: A dataset file with the same settings already exists: {existing_files[0].name}.")
                print("Skipping recording dataset splits to avoid overwriting.")
            else:
                print(f"Recording dataset splits to {out_filename}")
                with open(out_filename, "w") as f:
                    f.write(f"Training set ({len(train_files)} images):\n")
                    for file in train_files:
                        f.write(f"{file}\n")
                    f.write(f"\nValidation set ({len(val_files)} images):\n")
                    for file in val_files:
                        f.write(f"{file}\n")
                    f.write(f"\nTest set ({len(train_files)} images):\n")
                    for file in train_files:
                        f.write(f"{file}\n")

            if split_type == 'train':
                self.paths = train_files.tolist()
                self.targets = train_labels.tolist()
            elif split_type == 'val':
                self.paths = val_files.tolist()
                self.targets = val_labels.tolist()
            elif split_type == 'test':
                self.paths = test_files.tolist()
                self.targets = test_labels.tolist()
        else:
            # Original simple split
            # (paths, _, labels, _) = train_test_split(
            #     np.array(self.paths),
            #     np.array(self.targets),
            #     test_size=self.eval_ratio,
            #     random_state=self.seed_random,
            #     stratify=np.array(self.targets)
            # )
            # No split, whole data is used for training
            self.paths = self.paths
            self.targets = self.targets

        print(f"Total images in {split_type} set: {len(self.paths)}")

    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        try:
            img = self.load_image(index)
            img.load()
            class_idx = self.targets[index]

            if self.preprocess:
                transformed_img = self.preprocess(img)
            else:
                transformed_img = img

            return transformed_img, class_idx
        except OSError as e:
            print(f"Skipping image at path {self.paths[index]} due to error: {e}")
            dummy_image = torch.zeros(3, self.size, self.size)
            dummy_label = 0
            return dummy_image, dummy_label

# Added from few_shot_finetuning.py for balanced sampling during training
class CLIP_BalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


def visualize_results(csv_file: str, output_dir: str, zero_shot: bool = False):
    """
    Generate a bar plot from a CSV file of model accuracies.

    :param csv_file: Path to the CSV file containing model accuracies.
    :param output_dir: Directory where the plot will be saved.
    :param vis_orders: Dictionary to define custom sorting order.
    :param base_model_colors: Dictionary to map base model names to specific colors.
    """
    base_model_colors = {
        "ViT-B/32 ": "steelblue",
        "ViT-B/16 ": "indigo",
        "ViT-L/14 ": "orange",
        "ViT-L/14-336 ": "gold"
    }

    category_codes = {
        "average": 10,
        "detail": 9,  # 9
        "extra": 8,  # 8
        "gemini": 6,  # 6
        "gpt": 4,  # 4
        "mid": 2,  # 2
        "min": 3,  # 3
        "short": 5,  # 5
        "init": 1,  # 1
    }

    vis_order = {}

    results_df = pd.read_csv(csv_file)

    for vis_model_name in results_df['model_name'].tolist():
        for code, order in category_codes.items():
            if code in vis_model_name:
                vis_order[vis_model_name] = order
                break

    # Load the CSV into a DataFrame

    if not zero_shot:
        # Apply custom sorting based on vis_orders
        results_df['vis_order'] = results_df['model_name'].apply(lambda x: vis_order.get(x, 0))
        results_df.sort_values(by='vis_order', inplace=True, ascending=True)
        results_df.drop(columns='vis_order', inplace=True)

    # Assign colors based on base model
    results_df['color'] = results_df['model_name'].apply(
        lambda x: next((color for base, color in base_model_colors.items() if base in x), 'black')
    )

    # Generate the bar plot
    plt.figure(figsize=(12, 7))
    plt.bar(results_df['model_name'], results_df['accuracy'], color=results_df['color'])
    plt.xlabel("Model Name")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # set min-max y-axis values
    plt.ylim(results_df['accuracy'].min()-1, 100 if results_df['accuracy'].max() == 100 else results_df['accuracy'].max()+1)

    # Save the plot
    plot_output_dir = Path(output_dir)
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    plot_output_path = plot_output_dir / f"model_accuracy_plot{'_zero' if zero_shot else ''}.png"
    plt.savefig(plot_output_path, dpi=300)
    plt.close()
    print(f"Accuracy plot saved to {plot_output_path}")


def evaluate_multiple_models(model_dir: str, eval_dir: str, device: str, cat_prefix: str,
                             upper_categ_limit: int, random_seed: int, img_size: int, input_format: str,
                             model_suffix: str = "05.pt", vis: bool = True,
                             preprocess_func: callable = None, test_fraction: float = 0.1,
                             batch_size: int = 8, zero_shot: bool = False):
    """
    Evaluates multiple saved models in a directory and records their Top-1 accuracy.
    :param model_dir: Directory containing the saved model files.
    :param eval_dir: Directory for evaluation data.
    :param model_suffix: Suffix to filter model files (e.g., ".pt", "_cp.pt").
    :param vis: If True, visualize results in a bar graph.
    :param batch_size: Batch size for evaluation.
    :param zero_shot: If True, perform zero-shot evaluation using pre-computed text features.
    :param upper_categ_limit: Maximum number of samples per category during evaluation.
    :param random_seed: Random seed for reproducibility.
    :param img_size: Image size for preprocessing.
    :param input_format: Image file format (e.g., "png", "jpg").
    :param preprocess_func: Preprocessing function for images.
    :param test_fraction: Fraction of data to use for testing if not using advanced split.
    :param device: Device to run the evaluation on (e.g., "cpu", "cuda").
    """
    from classifier import CLIP, split_data_80_10_10

    map_base_name = {
        "ViTB32_": "ViT-B/32",
        "ViTB16_": "ViT-B/16",
        "ViTL14_": "ViT-L/14",
        "ViTL14336px_": "ViT-L/14@336px",
    }

    category_sufix = {
        "000c": "average",
        "01c": "detail",  # 9
        "02c": "extra",  # 8
        "03c": "gemini",  # 6
        "04c": "gpt",  # 4
        "05c": "mid",  # 2
        "06c": "min",  # 3
        "07c": "short",  # 5
        "08c": "init",  # 1
    }

    model_dir_path = Path(model_dir)
    if not model_dir_path.is_dir():
        print(f"Error: Model directory not found at {model_dir}")
        return

    model_files = list(model_dir_path.rglob(f"*{model_suffix}"))
    if not model_files:
        print(f"No model files found with suffix '{model_suffix}' in {model_dir}")
        return

    print(f"Found {len(model_files)} models with suffix '{model_suffix}'.")
    accuracies = {}  # To store filename_stem: top1_accuracy

    cur = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    output_dir = cur / "result"
    output_dir.mkdir(exist_ok=True)

    if zero_shot:
        print("Zero-shot evaluation mode enabled. Using pre-computed text features.")
        for base_name in map_base_name.values():
            print(f"Using base model: {base_name}")
            vis_model_name = f"{base_name} zero"

            try:
                clip_instance = CLIP(None, None, 1, base_name, device,
                                     seed=random_seed, test_ratio=test_fraction, input_format=input_format,
                                     cat_prefix=cat_prefix, output_dir=str(output_dir), avg=True, zero_shot=False)

                # Prepare evaluation dataset and dataloader once
                eval_dataset = ImageFolderCustom(eval_dir, max_category_samples=upper_categ_limit,
                                         preprocess_fn=preprocess_func, img_size=img_size,
                                         file_format=input_format, use_advanced_split=False, test_ratio=test_fraction,
                                         split_type='test', seed=random_seed, model_name=base_name)
                eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)

                accuracies[vis_model_name] = clip_instance.test(eval_dataloader, vis=False)
                print(f"Top 1 Accuracy for {vis_model_name} {base_name}: {accuracies[vis_model_name]:.2f}%")
            except Exception as e:
                print(f"Error evaluating model {base_name}: {e}")
                if vis_model_name not in accuracies.keys():
                    accuracies[vis_model_name] = "Error"  # Indicate error
    else:
        for model_path in sorted(model_files):
            model_name_stem = model_path.stem
            print(f"\nEvaluating model: {model_name_stem}")

            base_name = None
            for short, clip in map_base_name.items():
                if short in model_name_stem:
                    base_name = clip
                    break

            vis_categ = "UNK"  # Default category
            for code, categ in category_sufix.items():
                if code in model_name_stem:
                    vis_categ = categ
                    break

            if base_name is None:
                vis_model_name = f"{model_name_stem} {vis_categ}"
                accuracies[vis_model_name] = "Error"  # Indicate error
            else:
                vis_model_name = f"{base_name.replace('@', '-')} {vis_categ}"
                print(vis_model_name)
                try:
                    # Load model state dict
                    clip_instance = CLIP(None, None, 1, base_name, device,
                                         seed=random_seed, test_ratio=test_fraction, input_format=input_format,
                                         cat_prefix=cat_prefix, output_dir=str(output_dir), avg=True, zero_shot=False)

                    # Prepare evaluation dataset and dataloader once
                    eval_dataset = ImageFolderCustom(eval_dir, max_category_samples=None, seed=random_seed,
                                                     test_ratio=test_fraction, split_type='test',
                                                     file_format=input_format, model_name=model_name_stem,
                                                     preprocess_fn=clip_instance.preprocess, use_advanced_split=False,
                                                     img_size=clip_instance.preprocess.transforms[0].size)
                    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)

                    checkpoint = torch.load(model_path, map_location=device)
                    clip_instance.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Model loaded from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}.")

                    accuracies[vis_model_name] = clip_instance.test(eval_dataloader, vis=False)
                    print(f"Top 1 Accuracy for {vis_model_name} {model_name_stem}: {accuracies[vis_model_name]:.2f}%")

                except Exception as e:
                    print(f"Error evaluating model {model_name_stem}: {e}")
                    if vis_model_name not in accuracies.keys():
                        accuracies[vis_model_name] = "Error"  # Indicate error

    # Save results to a table
    if accuracies:
        results_df = pd.DataFrame(list(accuracies.items()), columns=['model_name', 'accuracy'])

        # Create a dedicated directory for evaluation statistics
        eval_stats_output_dir = Path(output_dir) / 'stats'
        eval_stats_output_dir.mkdir(parents=True, exist_ok=True)

        csv_output_path = eval_stats_output_dir / f"model_accuracies{'_zero' if zero_shot else ''}.csv"
        results_df.to_csv(csv_output_path, index=False)
        print(f"Evaluation results saved to {csv_output_path}")

        if vis:
            # sort results by vis order
            visualize_results(str(csv_output_path), str(Path(output_dir) / 'stats'), zero_shot)
    else:
        print("No models were successfully evaluated.")


def load_categories(tsv_file, directory = None, prefix=None):
    """
    Loads categories and descriptions from TSV files for the CLIP model.
    If prefix is provided, it loads all files starting with that prefix from the directory.
    """
    categories_data = defaultdict(list)

    dir = directory if directory else "category_descriptions"
    base_dir = Path(__file__).parent  # directory of tables
    if not base_dir.is_dir():
        print(f"Error: {base_dir} not a directory")
        return {}

    if prefix:
        files_to_load = list(base_dir.glob(f"{prefix}*.tsv")) + list(base_dir.glob(f"{prefix}*.csv"))
        if not files_to_load:
            print(f"Warning: No TSV files with prefix '{prefix}' found in {base_dir}. Using default categories.")
            return {"DRAW": ["a drawing"], "PHOTO": ["a photo"], "TEXT": ["text"], "LINE": ["a table"]}

        print(f"Found {len(files_to_load)} category files with prefix '{prefix}'.")
        for file_path in files_to_load:
            try:
                with open(file_path, "r") as file:
                    reader = csv.DictReader(file, delimiter="\t") if file_path.suffix == '.tsv' else csv.DictReader(file, delimiter=",")
                    for row in reader:
                        categories_data[row["label"].replace("+AF8-", "_")].append(row["description"])
            except Exception as e:
                print(f"Error reading categories file {file_path}: {e}")

        # for cat, descs in categories_data.items():
        #     print(f"Category: {cat}, Descriptions: {descs}")
        return categories_data

    else:  # Original behavior
        categories = []
        tsv_file_or_dir = base_dir / tsv_file

        if not os.path.exists(tsv_file_or_dir):
            print(f"Warning: Categories file not found at {tsv_file_or_dir}. Using default categories.")
            return [("DRAW", "a drawing"), ("PHOTO", "a photo"), ("TEXT", "text"), ("LINE", "a table")]
        try:
            with open(tsv_file_or_dir, "r") as file:
                reader = csv.DictReader(file, delimiter="\t") if tsv_file_or_dir.suffix == '.tsv' else csv.DictReader(file, delimiter=",")
                for row in reader:
                    categories.append((row["label"].replace("+AF8-", "_"), row["description"]))

            uniques_categs = set([cat for cat, _ in categories])
            if len(categories) > len(uniques_categs):
                categories_data = defaultdict(list)
                for cat, desc in categories:
                    categories_data[cat].append(desc)
                categories = {cat: descs for cat, descs in categories_data.items()}
        except Exception as e:
            print(f"Error reading categories file: {e}")
            return []  # Fallback
        print(f"Loaded {len(categories)} categories from {tsv_file_or_dir}")
        return categories