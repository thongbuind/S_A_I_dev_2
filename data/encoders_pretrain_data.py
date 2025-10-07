import json
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer

current_file = Path(__file__).resolve()
data_dir = current_file.parent
config_file = data_dir.parent / "config" / "config.json"
raw_dir = data_dir / "raw"
processed_dir = data_dir / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

with open(config_file, 'r') as f:
    config = json.load(f)
max_seq_len = config['max_seq_len']

tokenizer_file = processed_dir / "tokenizer.json"
tokenizer = Tokenizer.from_file(str(tokenizer_file))
vocab = tokenizer.get_vocab()
idx2word = {i: w for w, i in vocab.items()}

dataset = []
with open(raw_dir / "shorted_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            continue

        if isinstance(obj, dict) and "text" in obj:
            text = obj["text"].strip()
            if text:
                dataset.append(text)

X, Y, lengths = [], [], []
total_lines = len(dataset)
for idx, line in enumerate(dataset):
    if idx % 10000 == 0:
        print(f"🔄 Đang xử lý dòng {idx}/{total_lines}...")

    encoded = tokenizer.encode(line)
    tokens = encoded.ids

    if len(tokens) < 2 or len(tokens) + 2 > max_seq_len:
        continue

    inp = [vocab["[BOS]"]] + tokens
    tgt = tokens + [vocab["[EOS]"]]
    
    X.append(inp)
    Y.append(tgt)
    lengths.append(len(inp))

print(f"\n📊 THỐNG KÊ DỮ LIỆU:")
print(f"📊 Tổng số mẫu: {len(X)}")
print(f"📈 Độ dài sequence trung bình: {np.mean(lengths):.2f}")
print(f"📉 Độ dài sequence min/max: {min(lengths)}/{max(lengths)}")

np.savez_compressed(
    processed_dir / "data_ids.npz",
    X=np.array(X, dtype=object),
    Y=np.array(Y, dtype=object),
    lengths=np.array(lengths)
)

print(f"✅ Đã lưu dữ liệu vào: {processed_dir}/data_ids.npz")
