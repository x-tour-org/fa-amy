
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score

from model.fa_amy import FAAmyModule

# ======================================================
# CONFIGURATION AREA - PLEASE MODIFY THE PATHS BELOW
# ======================================================
# 1. Path to pre-computed positive and negative embeddings (.npy)
POS_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Dataset", "Generalized_dataset", "esmc_pos_test.npy")
NEG_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Dataset", "Generalized_dataset", "esmc_neg_test.npy")

# 2. Path to the saved model weight (.pth)
MODEL_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "Model-saved", "final_model.pth")

# 3. Hyperparameters
BATCH_SIZE = 8
SEED = 777
# ======================================================

def set_random_seed(seed):
    """Ensure reproducibility by fixing random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set deterministic algorithms (required for reproducibility in some PyTorch versions)
    torch.use_deterministic_algorithms(True, warn_only=True)

set_random_seed(SEED)

class BioinformaticsDataset(Dataset):
    """Custom dataset for loading protein embeddings and labels."""
    def __init__(self, label, prot):
        self.lb = label
        self.df_prot = prot

    def __getitem__(self, index):
        prot = torch.tensor(self.df_prot[index], dtype=torch.float)
        label = torch.tensor(self.lb[index], dtype=torch.float)
        return prot, label

    def __len__(self):
        return len(self.df_prot)

# ================== Model Architecture ==================




# ================== Evaluation Loop ==================

def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Test on Device: {device} ---")

    # 1. Load Pre-computed Data
    try:
        pos_test = np.load(POS_DATA_PATH)
        neg_test = np.load(NEG_DATA_PATH)
        print(f"Data Loaded: Positive={len(pos_test)}, Negative={len(neg_test)}")
    except FileNotFoundError as e:
        print(f"Error: Could not find .npy files. {e}")
        return

    # Combine data and create labels
    test_prot = np.concatenate((pos_test, neg_test), axis=0)
    test_labels = np.concatenate((np.ones(len(pos_test)), np.zeros(len(neg_test))))

    # 2. Data Loader
    test_loader = DataLoader(BioinformaticsDataset(test_labels, test_prot), batch_size=BATCH_SIZE, shuffle=False)

    # 3. Initialize Model
    model = FAAmyModule().to(device)
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=device))
        print(f"Model loaded successfully from: {MODEL_CHECKPOINT_PATH}")
    else:
        print(f"Warning: Checkpoint not found at {MODEL_CHECKPOINT_PATH}. Using random weights.")

    model.eval()
    y_true, y_probs = [], []

    print("Evaluating...")
    with torch.no_grad():
        for prot_x, labels_y in test_loader:
            prot_x = prot_x.to(device)
            outputs = torch.sigmoid(model(prot_x))
            y_true.extend(labels_y.numpy())
            y_probs.extend(outputs.cpu().numpy())

    # 4. Metric Calculation
    labels = np.array(y_true)
    probs = np.array(y_probs).flatten()
    preds = np.around(probs)

    auc_score = roc_auc_score(labels, probs)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    # Calculate Metrics
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = (tp + tn) / (tp + tn + fn + fp)
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0
    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = np.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))
    mcc = mcc_num / mcc_den if mcc_den > 0 else 0
    f1 = 2 * pre * sn / (pre + sn) if (pre + sn) > 0 else 0

    print("\n" + "="*40)
    print(f"TEST RESULTS:")
    print(f"SN: {sn:.4f} | SP: {sp:.4f} | ACC: {acc:.4f}")
    print(f"MCC: {mcc:.4f} | Pre: {pre:.4f} | AUC: {auc_score:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("="*40)

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    run_test()
