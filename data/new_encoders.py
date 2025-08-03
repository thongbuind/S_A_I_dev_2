from tokenizers import Tokenizer, trainers, models, pre_tokenizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from tokenizers.pre_tokenizers import Whitespace
import json
import numpy as np
from pathlib import Path

current_file = Path(__file__).resolve()
data_dir = current_file.parent
raw_dir = data_dir / "raw"
processed_dir = data_dir / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

# Bước 1: Tải dữ liệu
dataset = []
with open(raw_dir / "pre_train.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)
    dataset = [item.strip() for item in json_data if isinstance(item, str) and item.strip()]

# Bước 2: Tạo tokenizer BPE
tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
tokenizer.pre_tokenizer = Whitespace()
trainer = trainers.BpeTrainer(
    vocab_size=20000, min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
)
tokenizer.train_from_iterator(dataset, trainer=trainer)

# Bước 3: Lưu tokenizer và vocab
tokenizer.save(str(processed_dir / "bpe_tokenizer.json"))

vocab = tokenizer.get_vocab()
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
with open(data_dir / "new_vocab.txt", 'w', encoding='utf-8') as f:
    for token, idx in sorted_vocab:
        f.write(f"{token}\t{idx}\n")

# Bước 4: Tạo X, Y, lengths (giống format BOS + tokens | tokens + EOS)
BOS_id = vocab.get("[BOS]", 0)
EOS_id = vocab.get("[EOS]", 1)

X, Y, lengths = [], [], []

for line in dataset:
    encoded = tokenizer.encode(line)
    token_ids = encoded.ids
    if len(token_ids) < 1:
        continue
    inp = [BOS_id] + token_ids
    tgt = token_ids + [EOS_id]
    X.append(inp)
    Y.append(tgt)
    lengths.append(len(inp))

# Bước 5: Lưu toàn bộ thành 1 file .npz
np.savez_compressed(
    processed_dir / "new_data_tokenized.npz",
    X=np.array(X, dtype=object),
    Y=np.array(Y, dtype=object),
    lengths=np.array(lengths)
)

print("✅ Đã lưu X, Y, lengths vào: new_data_tokenized.npz")
print(f"📊 Tổng số mẫu: {len(X)} | Độ dài TB: {np.mean(lengths):.2f}")
