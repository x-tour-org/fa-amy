import os
import torch
from torch.utils.data import DataLoader

from dataset.embedding_dataset import EmbeddingDataset
from helpers.seeder import Seeder
from model.fa_amy import FAAmyModule

# ======================================================
# CONFIGURATION AREA - PLEASE MODIFY THE PATHS BELOW
# ======================================================
# Path to your pre-computed embedding features (.npy file)
INPUT_NPY_PATH = "Test.npy"

# Path to save the final prediction results
OUTPUT_CSV_PATH = "prediction_results.csv"

# Path to your trained model checkpoint (.pth file)
MODEL_CHECKPOINT_PATH = "final_model.pth"

# Inference Settings
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================

Seeder.set_seed(777)


def run_prediction():
    print(f"--- FA-Amy Inference Tool ---")
    print(f"Loading features from: {INPUT_NPY_PATH}")

    # Check if files exist
    if not os.path.exists(INPUT_NPY_PATH):
        print(f"Error: {INPUT_NPY_PATH} not found.")
        return

    # Initialize Dataset and DataLoader
    dataset = EmbeddingDataset(INPUT_NPY_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Model
    model = FAAmyModule().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE))
        print(f"Successfully loaded model weights from: {MODEL_CHECKPOINT_PATH}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.eval()
    all_probs = []

    print(f"Starting inference on {DEVICE}...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            batch_data = batch_data.to(DEVICE)
            outputs = model(batch_data)
            probs = torch.sigmoid(outputs).squeeze(-1).cpu().numpy()
            all_probs.extend(probs)

            if (batch_idx + 1) % 5 == 0:
                print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

    # Save results to CSV
    print(f"Writing results to: {OUTPUT_CSV_PATH}")
    with open(OUTPUT_CSV_PATH, "w") as f:
        f.write("Index,Probability,Prediction\n")
        for i, prob in enumerate(all_probs):
            pred = 1 if prob >= 0.5 else 0
            f.write(f"{i},{prob:.4f},{pred}\n")

    print(f"Done! Processed {len(all_probs)} sequences.")


if __name__ == "__main__":
    # Later account for the difference in Fa-Amy module channel count,
    # which diverges from test and train implementations
    run_prediction()
