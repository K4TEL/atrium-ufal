import sys
import torch
import clip
from PIL import Image
import csv
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import time
from utils import directory_scraper, dataframe_results, confusion_plot



# Function to load categories from a TSV file
def load_categories(tsv_file):
    categories = []
    try:
        with open(tsv_file, "r") as file:
            reader = csv.DictReader(file, delimiter="\t")
            for row in reader:
                categories.append((row["label"], row["description"]))
    except Exception as e:
        print(f"Error reading categories file: {e}")
        sys.exit(1)
    return categories

def process_images(image_paths, model, preprocess, categories, device):
    results = []
    raw_scores = []
    predictions = []

    true_labels = []
    
    for image_path in image_paths:
        try:
            # Load and preprocess the image
            image = Image.open(image_path)
            image_input = preprocess(image).unsqueeze(0).to(device)

            # Prepare text descriptions for the CLIP model
            text_inputs = torch.cat([clip.tokenize(description) for _, description in categories]).to(device)

            # Perform inference
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)

                # Compute similarity scores
                logits_per_image, _ = model(image_input, text_inputs)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

            # Store results
            print(f"\nResults for image: {image_path}")
            
            # Create a list of (index, score) tuples
            scored_categories = [(i, score) for i, score in enumerate(probs)]
            # Sort by score, highest first
            scored_categories.sort(key=lambda x: x[1], reverse=True)
            
            # Store top prediction
            top_pred = scored_categories[0][0]
            predictions.append(top_pred)
            
            # Store raw scores for confusion matrix generation
            raw_scores.append(probs)
            
            # Store top 3 predictions with scores
            results.append((image_path, scored_categories[:3]))
            
            # Print top results
            for i, prob in scored_categories[:3]:
                label, _ = categories[i]
                print(f"{label}: {prob * 100:.2f}%")

            parent_dir = image_path.parent.name
            if parent_dir in categories:
                true_labels.append(categories[parent_dir])
            else:
                # If no match, use -1 as unknown
                true_labels.append(-1)

        except Exception as e:
            print(f"Error processing file {image_path}: {e}")
    
    return results, raw_scores, predictions, true_labels

def extract_true_labels_from_directories(image_paths, subdirs):
    """Extract true labels based on subdirectory names"""
    true_labels = []
    
    # Create a mapping from subdirectory name to index
    subdir_to_index = {subdir: i for i, subdir in enumerate(subdirs)}
    
    for path in image_paths:
        path_obj = Path(path)
        # The parent directory name should match a category
        parent_dir = path_obj.parent.name
        if parent_dir in subdir_to_index:
            true_labels.append(subdir_to_index[parent_dir])
        else:
            # If no match, use -1 as unknown
            true_labels.append(-1)
    
    return true_labels

def main():
    # Automatically select the device based on availability
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use CUDA-enabled GPU on systems with NVIDIA GPUs
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Use MPS on macOS with Apple Silicon
    else:
        device = torch.device("cpu")  # Fallback to CPU if no GPU is available

    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description='CLIP Zero-Shot Image Classification')
    parser.add_argument('--dir', type=str, required=True, help='Directory containing images to classify')
    parser.add_argument('--categories', type=str, default="page_categories.tsv", help='TSV file with categories')
    parser.add_argument('--model', type=str, default="ViT-L/14@336px", help='CLIP model to use')
    parser.add_argument('--inner', action='store_true', help='Process subdirectories of the given directory')
    args = parser.parse_args()
    
    # Load the CLIP model and preprocess function
    print(f"Loading CLIP model: {args.model}")
    model, preprocess = clip.load(args.model, device=device)

    # Load categories from TSV file
    categories = load_categories(args.categories)
    category_names = [label for label, _ in categories]
    print(f"Loaded {len(categories)} categories from {args.categories}")
    
    # Get list of image paths from directory
    input_dir = Path(args.dir)
    if args.inner:
        image_paths = sorted(directory_scraper(input_dir, "png"))
    else:
        files = [f for f in input_dir.glob("*.png")]
        image_paths = [str(f) for f in files]
    
    print(f"Found {len(image_paths)} PNG images to process")
    
    # Process all images
    results, raw_scores, predictions, true_labels = process_images(image_paths, model, preprocess, categories, device)
    
    # Generate results dataframe
    time_stamp = time.strftime("%Y%m%d-%H%M")
    
    # Format results for dataframe_results function
    formatted_results = []
    for img_path, preds in results:
        formatted_results.append(preds)
    
    # Generate a results table
    rdf, raw_df = dataframe_results(
        image_paths,
        formatted_results,
        category_names,
        top_N=3,
        raw_scores=raw_scores
    )
    
    # Save results to CSV
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    tables_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    output_file = tables_dir / f"{time_stamp}_clip-zero_TOP-3.csv"
    rdf.sort_values(['FILE', 'PAGE'], ascending=[True, True], inplace=True)
    rdf.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    
    # Check if we need to generate confusion matrix (input directory has subdirectories matching categories)
    # Generate confusion matrix
    confusion_plot(
        predictions,
        true_labels,
        category_names,
        "clip-zero",
        top_N=1,
        output_dir=str(output_dir)
    )
    print(f"Confusion matrix generated in {plots_dir}")

if __name__ == "__main__":
    main()
