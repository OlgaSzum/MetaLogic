import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import shutil
import json

# --- Configuration ---
INPUT_CSV_FILE = "outputs/cutouts_with_embeddings.csv"
EMBEDDING_COLUMN = 'openai/clip-vit-base-patch32_Embedding'
CUTOUT_PATH_COLUMN = 'Cutout Image Path'

CLIP_HF_MODEL_ID = "openai/clip-vit-base-patch32"
N_NEIGHBORS_TO_RETURN = 80
KNN_METRIC = 'cosine'

# --- Image Query ---
QUERY_IMAGE_PATH_DEFAULT = '/Users/olga/MetaLogic/data/pojazdy_transport/turzanskiw_574.tif'
QUERY_IMAGE_PATH = os.environ.get("QUERY_IMAGE_PATH", QUERY_IMAGE_PATH_DEFAULT)


# --- Output Configuration ---
OUTPUT_FOLDER_BASE = "outputs/image_search_neighbors"
COPY_NEIGHBOR_IMAGES = True
SAVE_NEIGHBOR_CSV = True # Set to True to save neighbor details to a CSV
NEIGHBOR_CSV_FILENAME = 'nearest_neighbors_details.csv' # Filename for the neighbor CSV

# --- Device ---
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# --- Load CLIP ---
processor = CLIPProcessor.from_pretrained(CLIP_HF_MODEL_ID)
model = CLIPModel.from_pretrained(CLIP_HF_MODEL_ID).to(device)

# --- Main Logic ---

# Load data
df = pd.read_csv(INPUT_CSV_FILE)

df[CUTOUT_PATH_COLUMN] = df[CUTOUT_PATH_COLUMN].astype(str).apply(
    lambda p: os.path.abspath(p) if p and not os.path.isabs(p) else p
)

# Convert string embeddings to NumPy arrays
emb_list = df[EMBEDDING_COLUMN].apply(json.loads).tolist()     # list of list[float]

# Prepare dataset embeddings matrix
embeddings_matrix = np.vstack(emb_list).astype(np.float32)     # shape: (N, 512)

# Generate image query embedding
image = Image.open(QUERY_IMAGE_PATH).convert("RGB")
with torch.no_grad():
     inputs = processor(images=image, return_tensors="pt").to(device)
     features = model.get_image_features(**inputs)
     query_embedding = features / features.norm(dim=-1, keepdim=True)
     query_embedding = query_embedding.cpu().numpy().reshape(1, -1)

# Perform Nearest Neighbor Search
knn = NearestNeighbors(n_neighbors=N_NEIGHBORS_TO_RETURN, metric=KNN_METRIC)
knn.fit(embeddings_matrix)
distances, indices = knn.kneighbors(query_embedding)

# --- Create Neighbor DataFrame ---
nearest_neighbors_df = pd.DataFrame() # Initialize an empty DataFrame

if indices.size > 0:
    neighbor_indices_iloc = indices[0]
    neighbor_distances = distances[0]

    # Select the rows from the original DataFrame
    nearest_neighbors_df = df.iloc[neighbor_indices_iloc].copy()

    # Add distance and similarity
    nearest_neighbors_df['Distance'] = neighbor_distances
    nearest_neighbors_df['Similarity'] = 1 - neighbor_distances

    # Sort by distance (ascending)
    nearest_neighbors_df = nearest_neighbors_df.sort_values(by='Distance')

    print("\nFound Nearest Neighbors:")
    # Display basic info
    cols = ['Distance', 'Similarity']zz
    if CUTOUT_PATH_COLUMN in nearest_neighbors_df.columns:
        cols.append(CUTOUT_PATH_COLUMN)
    if 'Original Image Path' in nearest_neighbors_df.columns:
        cols.append('Original Image Path')

    print(nearest_neighbors_df[cols].to_string())
else:
    print("\nNo nearest neighbors found.")

# --- Copy Nearest Neighbor Images ---
if COPY_NEIGHBOR_IMAGES and not nearest_neighbors_df.empty:
    output_image_folder = os.path.join(OUTPUT_FOLDER_BASE, 'neighbor_cutouts')
    os.makedirs(output_image_folder, exist_ok=True)

    print(f"\nCopying {len(nearest_neighbors_df)} nearest neighbor images to '{output_image_folder}'...")

    copied_count = 0
    for index, row in nearest_neighbors_df.iterrows():
        cutout_path = row.get(CUTOUT_PATH_COLUMN)
        if isinstance(cutout_path, str) and os.path.exists(cutout_path):
            try:
                distance_str = f"{row.get('Distance', 0.0):.4f}".replace('.', '_')
                original_index = row.name # Use the DataFrame index
                base_filename = os.path.basename(cutout_path)
                dest_filename = f"Dist_{distance_str}_Index_{original_index}_{base_filename}"
                dest_filename = dest_filename.replace('/', '_').replace('\\', '_').replace(':', '_')
                dest_path = os.path.join(output_image_folder, dest_filename)
                shutil.copy2(cutout_path, dest_path)
                copied_count += 1
            except Exception as e:
                print(f"Could not copy image '{cutout_path}': {e}")
    print(f"Finished copying {copied_count} images.")

# --- Save Neighbor DataFrame to CSV ---
if SAVE_NEIGHBOR_CSV and not nearest_neighbors_df.empty:
    output_csv_path = os.path.join(OUTPUT_FOLDER_BASE, NEIGHBOR_CSV_FILENAME)
    os.makedirs(OUTPUT_FOLDER_BASE, exist_ok=True) # Ensure the base folder exists for CSV

    print(f"\nSaving nearest neighbor details to '{output_csv_path}'...")

    df_to_save = nearest_neighbors_df.copy()
    for col in ['Distance', 'Similarity']:
         if col in df_to_save.columns and pd.api.types.is_numeric_dtype(df_to_save[col]):
              df_to_save[col] = df_to_save[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else '')

    try:
         df_to_save.to_csv(output_csv_path, index=False)
         print("Neighbor details CSV saved.")
    except Exception as e:
         print(f"Error saving neighbor details CSV: {e}")

