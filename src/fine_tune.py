from keras import models
import sys
import numpy as np
from pathlib import Path

current_file = Path(__file__).resolve()
src_dir = current_file.parent
project_root = src_dir.parent
sys.path.append(str(project_root))
config_file = project_root / "config" / "config.json"
vocab_file = project_root/ "data" / "vocab.txt"
model_file = project_root / "model" / "s_a_i.keras"
processed_dir = project_root / "data" / "processed"

model = models.load_model(model_file)

def load_finetune_data():
    finetune_data_path = processed_dir / "finetune_data.npz"
    data = np.load(finetune_data_path, allow_pickle=True)

    input = data["input"]
    respones = data["respones"]
    input_lengths = data["input_lengths"]
    respones_lengths = data["respones_lengths"]

    return input, respones, input_lengths, respones_lengths

