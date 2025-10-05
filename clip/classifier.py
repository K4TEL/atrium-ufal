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
import clip
from PIL import Image, ImageEnhance, ImageFilter

from huggingface_hub import PyTorchModelHubMixin

Image.MAX_IMAGE_PIXELS = 700_000_000
import string
import torch.nn as nn

from minor_classes import *


class CLIP(nn.Module, PyTorchModelHubMixin):
    # --- HF‐Hub Mixin metadata (used in the generated model card) ---
    library_name = "ufal/clip-historical-page"
    tags          = ["vision-language", "clip", "custom"]
    pipeline_tag  = "few-shot-image-classification"

    def __init__(self,
                 max_category_samples: int | None,
                 eval_max_category_samples: int | None,
                 top_N: int,
                 model_name: str,
                 device: str,
                 seed: int,
                 test_ratio: float,
                 input_format: str,
                 categories_tsv: str,
                 categories_dir: str,
                 output_dir: str = None,
                 cat_prefix: str = None,
                 avg: bool = True,
                 zero_shot: bool = False):
        super().__init__()  # initialize nn.Module
        # all your existing init logic follows unchanged:
        self.upper_category_limit      = max_category_samples
        self.upper_category_limit_eval = eval_max_category_samples
        self.top_N                     = top_N
        self.seed                      = seed
        self.avg                       = avg
        self.device                    = device
        self.zero_shot                 = zero_shot

        self.test_fraction = test_ratio
        self.file_format = input_format

        self.output_dir = Path(__file__).parent / "result" if output_dir is None else Path(output_dir)
        self.download_root = '/lnet/work/projects/atrium/cache/clip'

        # Must set jit=False for training
        self.model, self.preprocess = clip.load(model_name, device=device,
                                                download_root=self.download_root, jit=False)

        image_size = (self.preprocess.transforms[0].size, self.preprocess.transforms[0].size)
        image_mean = self.preprocess.transforms[-1].mean
        image_std = self.preprocess.transforms[-1].std

        # Define transformations
        self.train_transforms = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5),
                transforms.ColorJitter(contrast=0.5),
                transforms.ColorJitter(saturation=0.5),
                transforms.ColorJitter(hue=0.5),
                transforms.Lambda(lambda img: ImageEnhance.Sharpness(img).enhance(random.uniform(0.5, 1.5))),
                transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 2))))
            ], p=0.5),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std)
        ])

        self.eval_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std)
        ])

        if self.avg:
            loaded_cats = load_categories(categories_tsv, prefix=cat_prefix, directory=categories_dir)
            self.categories = sorted(loaded_cats.keys())
            self.class_to_idx = {cat: i for i, cat in enumerate(self.categories)}
            self.texts = loaded_cats  # dict of lists
            print(f"Categories: {self.categories} with multiple descriptions per category.")
            self.text_features = self._get_averaged_text_features()
            for cat, descs in self.texts.items():
                print(f"Category: {cat}, Descriptions:")
                for desc in descs:
                    print(f"  - {desc}")
        else:
            loaded_cats = load_categories(categories_tsv, directory=categories_dir)
            if type(loaded_cats) is dict:
                self.categories = sorted(loaded_cats.keys())
                self.class_to_idx = {cat: i for i, cat in enumerate(self.categories)}
                self.texts = loaded_cats  # dict of lists
                print(f"Categories: {self.categories} with multiple descriptions per category.")
                self.text_features = self._get_averaged_text_features()
                for cat, descs in self.texts.items():
                    print(f"Category: {cat}, Descriptions:")
                    for desc in descs:
                        print(f"  - {desc}")
                self.avg = True
            else:
                self.categories = [label for label, desc in loaded_cats]
                self.texts = [desc for label, desc in loaded_cats]
                self.text_inputs = torch.cat([clip.tokenize(f"a scan of {description}") for description in self.texts]).to(
                        device)
                print(f"Categories: {self.categories} with single description per category.")

        self.model_name = model_name

        self.num_prediction_classes = len(self.categories)
        print(f"Number of prediction classes: {self.num_prediction_classes}")
        print(f"Model name: {self.model_name}")

        self.categories_dir = categories_dir
        self.categories_tsv = categories_tsv

    def _get_averaged_text_features(self):
        """Computes averaged text features for each category."""
        all_features = []
        with torch.no_grad():
            for category in self.categories:
                descriptions = [f"a scan of {desc}" for desc in self.texts[category]]
                tokens = torch.cat([clip.tokenize(desc) for desc in descriptions]).to(self.device)
                features = self.model.encode_text(tokens)
                features /= features.norm(dim=-1, keepdim=True)
                mean_features = features.mean(dim=0)
                mean_features /= mean_features.norm()
                all_features.append(mean_features)
        return torch.stack(all_features)

    def train(self, train_dir: str, eval_dir: str, log_dir: str, num_epochs: int = 5, batch_size: int = 8,
              learning_rate: float = 1e-7, save_interval: int = 1):
        """
        Fine-tunes the CLIP model based on the provided training and evaluation directories.
        """
        print("Starting CLIP model fine-tuning...")
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        remove_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        model_name_sanitized = self.model_name.translate(remove_punctuation).replace(" ", "")

        print(f"Current model name: \t{model_name_sanitized}")

        def convert_models_to_fp32(model):
            for p in model.parameters():
                if p.grad is not None:
                    p.data = p.data.float()
                    p.grad.data = p.grad.data.float()

        if self.device == "cpu":
            self.model.float()
        else:
            clip.model.convert_weights(self.model)

        writer = SummaryWriter(log_dir=log_dir)
        weights_path = Path("model_checkpoints")
        weights_path.mkdir(exist_ok=True)

        train_dataset = ImageFolderCustom(train_dir,
                                        max_category_samples=self.upper_category_limit,
                                        preprocess_fn=self.preprocess,
                                        img_size=self.preprocess.transforms[0].size,
                                        use_advanced_split=True,  # Enable new split
                                        split_type='train', seed=self.seed,
                                        file_format=self.file_format,
                                        test_ratio=self.test_fraction)

        train_labels = torch.tensor(train_dataset.targets)
        train_sampler = CLIP_BalancedBatchSampler(train_labels, batch_size, 1)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler)

        test_dataset = ImageFolderCustom(train_dir,
                                          max_category_samples=self.upper_category_limit_eval,
                                          preprocess_fn=self.preprocess,
                                          img_size=self.preprocess.transforms[0].size,
                                          use_advanced_split=True,  # Enable new split
                                          split_type='val', seed=self.seed,
                                          file_format=self.file_format,
                                          test_ratio=self.test_fraction)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

        loss_img = torch.nn.CrossEntropyLoss()
        loss_txt = torch.nn.CrossEntropyLoss()

        num_batches_train = len(train_dataloader)

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * num_batches_train,
                                                               eta_min=1e-10)

        print(f"Number of training batches: \t{num_batches_train}")
        print(f"Number of evaluation batches: \t{len(test_dataloader)}")

        ever_best_accuracy = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch: {epoch}/{num_epochs}")
            epoch_train_loss, step = 0, 0
            self.model.train()
            for batch in tqdm(train_dataloader, total=num_batches_train, desc="Training"):
                step += 1
                optimizer.zero_grad()

                images, class_ids = batch
                images = images.to(self.device)

                if self.avg:
                    # Encode images
                    image_features = self.model.encode_image(images)
                    text_features = torch.stack([self.text_features[label_id] for label_id in class_ids]).to(
                        self.device)

                    # Normalize features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    # Calculate logits
                    logit_scale = self.model.logit_scale.exp()
                    logits_per_image = logit_scale * image_features @ text_features.T.to(image_features.dtype)
                    logits_per_text = logits_per_image.T

                else:
                    texts = [f"a scan of {self.texts[label_id]}" for label_id in class_ids]
                    texts = clip.tokenize(texts).to(self.device)
                    logits_per_image, logits_per_text = self.model(images, texts)

                ground_truth = torch.arange(len(images), dtype=torch.long, device=self.device)

                total_train_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text,
                                                                                        ground_truth)) / 2
                total_train_loss.backward()
                epoch_train_loss += total_train_loss.item()

                torch.nn.utils.clip_grad_norm_(params, 1.0)

                if self.device != "cpu":
                    convert_models_to_fp32(self.model)
                optimizer.step()
                if self.device != "cpu":
                    clip.model.convert_weights(self.model)
                scheduler.step()

                if step % 25 == 0:
                    print(
                        f"Step {step}/{num_batches_train}, Loss: {total_train_loss.item():.4f}, lr: {optimizer.param_groups[0]['lr']:.10f}")

            avg_train_loss = epoch_train_loss / num_batches_train
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)
            print(f"{model_name_sanitized}\t Epoch {epoch} train loss: {avg_train_loss:.4f}")

            if epoch == num_epochs - 1:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_train_loss,
                        'rnd_data_seed': self.seed,
                    },
                    weights_path / f"model_{model_name_sanitized}_{self.upper_category_limit}c_{str(learning_rate)}_{num_epochs}e.pt")
                print(
                    f"Saved weights to {weights_path}/model_{model_name_sanitized}_{self.upper_category_limit}c_{str(learning_rate)}_{num_epochs}e.pt.")

            # Evaluation
            self.model.eval()

            # Fix: Use the correct number of classes for torchmetrics
            if self.avg:
                num_classes = len(self.categories)  # Use self.categories when avg=True
            else:
                num_classes = len(test_dataset.classes)  # Use dataset classes when avg=False

            acc_top1_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
            acc_top5_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(self.device)

            # For evaluation, always use ALL pre-computed text features
            if self.avg:
                text_features_eval = self.text_features  # Use the pre-computed full set of averaged text features
                text_features_eval = text_features_eval / text_features_eval.norm(dim=-1,
                                                                                  keepdim=True)  # Ensure normalized
            else:
                # Original logic for non-averaged categories
                all_texts = torch.cat([clip.tokenize(f"a scan of {c}") for c in self.texts]).to(self.device)
                with torch.no_grad():
                    text_features_eval = self.model.encode_text(all_texts)
                    text_features_eval = text_features_eval / text_features_eval.norm(dim=-1, keepdim=True)

            with torch.no_grad():
                for batch in tqdm(test_dataloader, desc="Evaluating"):
                    images, class_ids = batch
                    images, class_ids = images.to(self.device), class_ids.to(self.device)

                    image_features = self.model.encode_image(images)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    similarity = (100.0 * image_features @ text_features_eval.T)

                    acc_top1_metric.update(similarity, class_ids)
                    acc_top5_metric.update(similarity, class_ids)

            mean_top1_accuracy = acc_top1_metric.compute()
            mean_top5_accuracy = acc_top5_metric.compute()

            if mean_top1_accuracy > ever_best_accuracy:
                ever_best_accuracy = mean_top1_accuracy
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_train_loss,
                        'rnd_data_seed': self.seed,
                    },
                    weights_path / f"model_{model_name_sanitized}_{self.upper_category_limit}c_{str(learning_rate)}_cp.pt")
                print(
                    f"Saved checkpoint weights to {weights_path}/model_{model_name_sanitized}_{self.upper_category_limit}c_{str(learning_rate)}_cp.pt.")

            print(f"Mean Top 1 Accuracy: {mean_top1_accuracy.item() * 100:.2f}%.")
            print(f"Mean Top 5 Accuracy: {mean_top5_accuracy.item() * 100:.2f}%.")
            writer.add_scalar("Test Accuracy/Top1", mean_top1_accuracy, epoch)
            writer.add_scalar("Test Accuracy/Top5", mean_top5_accuracy, epoch)

            acc_top1_metric.reset()
            acc_top5_metric.reset()

        writer.flush()
        writer.close()
        print("Fine-tuning finished.")

        self.test(test_dataloader)

    def evaluate_saved_model(self, model_path: str, eval_dir: str, batch_size: int = 8):
        """
        Loads a saved model and evaluates its performance on the specified evaluation directory.
        """
        if model_path is not None:
            print(f"Loading model from {model_path} for evaluation...")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}.")

        eval_dataset = ImageFolderCustom(eval_dir, max_category_samples=self.upper_category_limit_eval,
                                         preprocess_fn=self.preprocess, img_size=self.preprocess.transforms[0].size,
                                         file_format=self.file_format, use_advanced_split=False, test_ratio=self.test_fraction,
                                         split_type='test', seed=self.seed, model_name=self.model_name)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)

        print("Starting evaluation of the loaded model...")
        self.test(eval_dataloader, image_files=eval_dataset.paths,)
        print("Evaluation finished.")

    def top_N_prediction(self, image_data: torch.Tensor, N: int):
        """
        Predicts the top N categories for a given image tensor.
        :param image_data: A tensor of shape (1, 3, H, W) representing the image.
        :param N: The number of top categories to return.
        """
        image_data = image_data.to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_data)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            if self.avg or self.zero_shot:
                # Use pre-computed averaged features for efficiency
                text_features = self.text_features
                similarity = (100.0 * image_features @ text_features.T)
                probs = similarity.softmax(dim=-1).cpu().numpy()
            else:
                logits_per_image, _ = self.model(image_data, self.text_inputs)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        pred_scores = probs[0]
        best_n_indices = np.argsort(pred_scores)[-N:][::-1]
        best_n_scores = pred_scores[best_n_indices]
        return best_n_scores, best_n_indices, pred_scores

    def prediction(self, image_data: torch.Tensor) -> (np.array, int):
        """
        Predicts the top N categories for a single image tensor and returns the scores and indices.
        :param image_data:
        :return:
        """
        # This method is less used, but we'll align it.
        scores, indices, _ = self.top_N_prediction(image_data.unsqueeze(0), len(self.categories))
        return scores, indices[0]

    def test(self, test_dataloader: torch.utils.data.DataLoader, model_name_sanitized: str = None,
             vis: bool = True, tab: bool = True, image_files: list = []):
        """
        Evaluates the model on the provided test dataloader and generates a confusion matrix plot.
        :param test_dataloader:
        :return:
        """
        remove_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        model_name_sanitized = self.model_name.translate(remove_punctuation).replace(" ", "") if model_name_sanitized is None else model_name_sanitized

        plot_path = Path(f'{self.output_dir}/plots')
        table_path = Path(f'{self.output_dir}/tables')
        plot_path.mkdir(parents=True, exist_ok=True)
        time_stamp = time.strftime("%Y%m%d-%H%M")
        plot_image = plot_path / f'{time_stamp}_EVAL_conf_{self.top_N}n_{self.upper_category_limit}c_{model_name_sanitized}.png'
        table_file = table_path / f'{time_stamp}_EVAL_table_{self.top_N}n_{self.upper_category_limit}c_{model_name_sanitized}.csv'

        all_pred_scores = []
        all_predictions = []
        all_true_labels = []

        self.model.eval()

        if self.avg:
            text_features_test = self.text_features  # Use the pre-computed full set of averaged text features
            text_features_test /= text_features_test.norm(dim=-1, keepdim=True)  # Ensure normalized
        else:
            # Original logic for non-averaged categories
            # The `test_dataloader.dataset.texts` is not directly accessible if not `self.avg`.
            # Instead, use the `self.texts` which holds all descriptions.
            all_texts = torch.cat(
                [clip.tokenize(f"a scan of {c}") for c in self.texts]).to(self.device)
            with torch.no_grad():
                text_features_test = self.model.encode_text(all_texts)
                text_features_test /= text_features_test.norm(dim=-1, keepdim=True)

        images = []
        with torch.no_grad():
            for images, class_ids in tqdm(test_dataloader, desc="Testing"):
                images = images.to(self.device)
                image_features = self.model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                similarity = (100.0 * image_features @ text_features_test.T)

                all_pred_scores.append(similarity.cpu().numpy())

                _, predicted_labels = similarity.max(dim=1)

                all_predictions.extend(predicted_labels.cpu().numpy())
                all_true_labels.extend(class_ids.cpu().numpy())
                # images.append(images.cpu().numpy())

        acc = round(100 * np.sum(np.array(all_predictions) == np.array(all_true_labels)) / len(all_true_labels), 2)
        print('Accuracy: ', acc)

        if vis:
            # Ensure display labels match the order of predictions
            display_labels = self.categories if self.avg else test_dataloader.dataset.classes

            disp = ConfusionMatrixDisplay.from_predictions(
                np.array(all_true_labels), np.array(all_predictions), cmap='inferno',
                normalize="true", display_labels=np.array(display_labels)
            )

            tick_positions = disp.ax_.get_xticks()
            short_labels = [f"{label[0]}{label.split('_')[-1][0] if '_' in label else ''}" for label in disp.display_labels]
            disp.ax_.set_xticks(tick_positions)
            disp.ax_.set_xticklabels(short_labels)

            disp.ax_.set_title(f"TOP {self.top_N} {self.upper_category_limit_eval}c {self.model_name} CM")
            plt.savefig(plot_image, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Confusion matrix saved to {plot_image}")

        if tab:
            out_df, _ = dataframe_results(test_images= image_files, test_predictions=all_pred_scores, raw_scores=None,
                                          top_N= self.top_N, categories=self.categories)
            all_true_labels = np.asarray(all_true_labels, dtype=int)
            out_df["TRUE"] = [self.categories[i] for i in all_true_labels]
            out_df.sort_values(['FILE', 'PAGE'], ascending=[True, True], inplace=True)
            out_df.to_csv(table_file, sep=",", index=False)
            print(f"Results for TOP-{self.top_N} predictions are recorded into {self.output_dir}/tables/ directory")

        return acc


    def save_model(self, save_directory: str):
        """
        Save the fine-tuned model and processor to the specified directory using PyTorchModelHubMixin.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        # The PyTorchModelHubMixin's save_pretrained method handles saving the model and its configuration.
        configs = {
            "vision_feat_dim": self.preprocess.transforms[0].size,
            "text_feat_dim": self.model.text_projection.shape[1] if hasattr(self.model, 'text_projection') else 512, # Assuming text feature dim from CLIP
            "model_name": self.model_name,
            "avg": self.avg,
            "categories": self.categories,
            "texts": self.texts, # Save the texts for reconstruction
            "zero_shot": self.zero_shot,
            "rnd_data_seed": self.seed,
            "image_size": self.preprocess.transforms[0].size,
            "image_mean": self.preprocess.transforms[-1].mean,
            "image_std": self.preprocess.transforms[-1].std
        }

        # expand config with base model configs
        if hasattr(self.model, 'config'):
            configs.update(self.model.config.to_dict())
            configs.update(self.preprocess.config.to_dict() if hasattr(self.preprocess, 'config') else {})

        self.save_pretrained(save_directory, config=configs)
        print(f"Model and configuration saved to {save_directory}")

    def load_model(self, load_directory: str):
        """
        Load a fine-tuned model and its configuration from the specified directory using PyTorchModelHubMixin.
        """
        # The from_pretrained method of PyTorchModelHubMixin loads the model into the current instance.
        # It also handles loading the associated configuration.
        loaded_model = self.from_pretrained(load_directory,
                                            max_category_samples=self.upper_category_limit,
                                            eval_max_category_samples=self.upper_category_limit_eval,
                                            top_N=self.top_N,
                                            model_name=self.model_name,  # will be set from config if needed
                                            device=self.device,
                                            categories_tsv=self.categories_tsv,
                                            categories_dir=self.categories_dir
                                            )
        self.model = loaded_model.model
        self.preprocess = loaded_model.preprocess # Assuming preprocess is also part of the loaded state or can be re-initialized

        # Re-initialize other necessary attributes from the loaded configuration
        self.model_name = loaded_model.model_name if hasattr(loaded_model, 'model_name') else self.model_name
        self.avg = loaded_model.avg if hasattr(loaded_model, 'avg') else self.avg
        self.categories = loaded_model.categories if hasattr(loaded_model, 'categories') else self.categories
        self.texts = loaded_model.texts if hasattr(loaded_model, 'texts') else self.texts
        self.zero_shot = loaded_model.zero_shot if hasattr(loaded_model, 'zero_shot') else self.zero_shot


        # Recompute text features if `avg` is true and `texts` were loaded
        if self.avg and hasattr(self, 'texts') and self.texts:
            print("Recomputing averaged text features after loading model.")
            self.text_features = self._get_averaged_text_features()

        self.num_prediction_classes = len(self.categories)
        print(f"Model and configuration loaded from {load_directory}")


    def pushing_to_hub(self, repo_id: str, private: bool = False,
                    token: str = None, revision: str = "main"):
        """
        Upload the fine-tuned model and its configuration to the Hugging Face Model Hub.

        Args:
            repo_id (str): The name of the repository to create or update on the Hugging Face Hub (e.g., "username/my-clip-model").
            private (bool, optional): Whether the repository should be private. Defaults to False.
            token (str, optional): The authentication token for Hugging Face Hub. Defaults to None.
            revision (str, optional): The revision (branch) to push to. Defaults to "main".
        """
        # The PyTorchModelHubMixin's push_to_hub method handles saving the model locally
        # and then pushing it to the Hugging Face Hub.
        # Ensure the config includes necessary parameters for re-instantiation.
        self.push_to_hub(repo_id, private=private, token=token, branch=revision, config={
            "vision_feat_dim": self.preprocess.transforms[0].size,
            "text_feat_dim": self.model.text_projection.shape[1] if hasattr(self.model, 'text_projection') else 512,
            "model_name": self.model_name,
            "avg": self.avg,
            "categories": self.categories,
            "texts": self.texts,
            "zero_shot": self.zero_shot,
            "rnf_data_seed": self.seed,
            "image_size": self.preprocess.transforms[0].size,
            "image_mean": self.preprocess.transforms[-1].mean,
            "image_std": self.preprocess.transforms[-1].std
        })
        print(f"Model and configuration pushed to the Hugging Face Hub: {repo_id}")

    def load_from_hub(self, repo_id: str, revision: str = "main"):
        """
        Load a model and its configuration from the Hugging Face Hub.

        Args:
            repo_id (str): The name of the repository on the Hugging Face Hub.
            revision (str, optional): The revision of the repository to load. Defaults to "main".
        """
        # The from_pretrained method of PyTorchModelHubMixin loads the model and its configuration
        # directly into the current instance.
        loaded_model = self.from_pretrained(repo_id, revision=revision,
                                            max_category_samples=self.upper_category_limit,
                                            eval_max_category_samples=self.upper_category_limit_eval,
                                            top_N=self.top_N,
                                            model_name=self.model_name,  # will be set from config if needed
                                            device=self.device,
                                            categories_tsv=self.categories_tsv,
                                            categories_dir=self.categories_dir
                                            )

        self.model = loaded_model.model
        self.preprocess = loaded_model.preprocess # Assuming preprocess is also part of the loaded state or can be re-initialized

        # Re-initialize other necessary attributes from the loaded configuration
        self.model_name = loaded_model.model_name if hasattr(loaded_model, 'model_name') else self.model_name
        self.avg = loaded_model.avg if hasattr(loaded_model, 'avg') else self.avg
        self.categories = loaded_model.categories if hasattr(loaded_model, 'categories') else self.categories
        self.texts = loaded_model.texts if hasattr(loaded_model, 'texts') else self.texts
        self.zero_shot = loaded_model.zero_shot if hasattr(loaded_model, 'zero_shot') else self.zero_shot

        # Recompute text features if `avg` is true and `texts` were loaded
        if self.avg and hasattr(self, 'texts') and self.texts:
            print("Recomputing averaged text features after loading model from Hub.")
            self.text_features = self._get_averaged_text_features()

        self.num_prediction_classes = len(self.categories)
        print(f"Model and configuration loaded from the Hugging Face Hub: {repo_id}")


    def predict_single(self, image_file: str) -> str:
        """
        Predicts the category of a single image file.
        :param image_file:
        :return:
        """
        image = Image.open(image_file)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        _, best_n_indices, _ = self.top_N_prediction(image_input, self.top_N)
        pred_label = self.categories[best_n_indices[0]]
        return pred_label

    def predict_top(self, image_file: str) -> (list, list):
        """
        Predicts the TOP-N categories of a single image file.
        :param image_file:
        :return:
        """
        image = Image.open(image_file)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        best_n_scores, best_n_indices, _ = self.top_N_prediction(image_input, self.top_N)
        best_n_scores = np.round(best_n_scores, 3).tolist()
        pred_labels = [self.categories[i] for i in best_n_indices]
        return best_n_scores, pred_labels

    def predict_directory(self, folder_path: str, raw: bool = False, out_table: str = None):
        """
        Predicts categories for all images in a directory and saves results to a CSV file.
        :param folder_path:
        :param raw:
        :param out_table:
        :return:
        """
        images = directory_scraper(Path(folder_path), self.file_format)
        print(f"Predicting {len(images)} images from {folder_path}")

        time_stamp = time.strftime("%Y%m%d-%H%M")  # for results files

        res_list, raw_list, tru_images = [], [], []

        for img_path in tqdm(images, desc="Predicting directory"):
            try:
                image = Image.open(img_path)
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                scores, indices, raw_scores = self.top_N_prediction(image_input, self.top_N)
                res_list.append(indices)
                if raw:
                    raw_list.append(raw_scores.tolist())

                tru_images.append(img_path.name)

            except Exception as e:
                print(f"Error processing file {img_path}: {e}")

        res_list = np.concatenate(res_list, axis=0)
        # print(res_list)
        out_df, raw_df = dataframe_results(test_images=tru_images, test_predictions=res_list, raw_scores=raw_list,
                                           top_N=self.top_N, categories=self.categories)

        out_df.sort_values(['FILE', 'PAGE'], ascending=[True, True], inplace=True)

        out_table = out_table if out_table is not None \
            else f"{self.output_dir}/tables/{time_stamp}_result_{self.top_N}n_{self.model_name.replace('/', '')}.csv"
        out_df.to_csv(out_table, sep=",", index=False)
        print(f"Results for TOP-{self.top_N} predictions are recorded into {self.output_dir}/tables/ directory")

        if raw:
            raw_df.sort_values(self.categories, ascending=[False] * len(self.categories), inplace=True)
            raw_df.to_csv(f"{self.output_dir}/tables/{time_stamp}_RAW_{self.model_name.replace('/', '')}.csv", sep=",", index=False)
            print(f"RAW Results are recorded into {self.output_dir}/tables/ directory")



def split_data_80_10_10(files: list, labels: list, random_seed: int, max_categ: int,
                        safe_check: bool = True):
    """
    Splits the data into training, validation, and test sets with an 80/10/10 ratio.
    The split uses uniform distribution selection to maintain temporal distribution
    across the sorted files (by creation date). Test and dev sets are selected first,
    with remaining samples going to training.

    Args:
        files: List of file paths (should be sorted alphabetically by creation date)
        labels: List of corresponding labels
        random_seed: Random seed for reproducibility
        max_categ: Maximum number of samples per category to consider
        safe_check: If True, checks for corrupted images and excludes them
    Returns:
        tuple: (train_files, val_files, test_files, train_labels, val_labels, test_labels)
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    files = np.array(files)
    labels = np.array(labels)

    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    for label, indices in label_to_indices.items():
        indices = np.array(indices)
        n_samples = len(indices)

        if n_samples > max_categ:
            print(f"Label {label} has {n_samples} samples, limiting to {max_categ}.")
            indices = np.random.choice(indices, size=max_categ, replace=False)

        label_to_indices[label] = indices.tolist()

    total_files = [files[idx] for label in label_to_indices for idx in label_to_indices[label]]
    total_labels = [labels[idx] for label in label_to_indices for idx in label_to_indices[label]]

    if safe_check:
        print(f"Checking {len(total_files)} files for corrupted images...")
        good_files, good_labels = [], []
        for file, label in zip(total_files, total_labels):
            try:
                Image.open(file).load()
                good_files.append(file)
                good_labels.append(label)
            except Exception as e:
                print(f"File {file} is corrupted: {e}")
                continue
        print(f"Total usable images found: {len(good_files)} / {len(total_files)}")
    else:
        good_files, good_labels = total_files, total_labels

    files, labels = np.array(good_files), np.array(good_labels)
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    test_indices = []
    val_indices = []
    train_indices = []

    for label, indices in label_to_indices.items():
        indices = np.array(indices)
        n_samples = len(indices)

        n_test = max(1, int(n_samples * 0.1))
        n_val = max(1, int(n_samples * 0.1))

        if n_test + n_val > n_samples:
            n_test = n_samples // 2
            n_val = n_samples - n_test

        if n_test > 0:
            test_step = n_samples / n_test
            test_positions = np.arange(0, n_samples, test_step)[:n_test]
            test_positions += np.random.uniform(-test_step / 4, test_step / 4, size=len(test_positions))
            test_positions = np.clip(test_positions, 0, n_samples - 1).astype(int)
            selected_test = indices[test_positions]
            test_indices.extend(selected_test)

        remaining_mask = np.ones(n_samples, dtype=bool)
        if n_test > 0:
            remaining_mask[test_positions] = False
        remaining_indices = indices[remaining_mask]
        n_remaining = len(remaining_indices)

        if n_val > 0 and n_remaining > 0:
            val_step = n_remaining / n_val if n_val <= n_remaining else 1
            val_positions = np.arange(0, n_remaining, val_step)[:n_val]
            if len(val_positions) > n_remaining:
                val_positions = np.arange(n_remaining)
            val_positions += np.random.uniform(-val_step / 4 if val_step > 1 else 0,
                                               val_step / 4 if val_step > 1 else 0,
                                               size=len(val_positions))
            val_positions = np.clip(val_positions, 0, n_remaining - 1).astype(int)
            selected_val = remaining_indices[val_positions]
            val_indices.extend(selected_val)

            val_mask = np.ones(n_remaining, dtype=bool)
            val_mask[val_positions] = False
            train_indices.extend(remaining_indices[val_mask])
        else:
            train_indices.extend(remaining_indices)

    test_indices = np.array(test_indices)
    val_indices = np.array(val_indices)
    train_indices = np.array(train_indices)

    test_files = files[test_indices]
    test_labels = labels[test_indices]

    val_files = files[val_indices]
    val_labels = labels[val_indices]

    train_files = files[train_indices]
    train_labels = labels[train_indices]

    return train_files, val_files, test_files, train_labels, val_labels, test_labels



