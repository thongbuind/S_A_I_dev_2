import json
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer, trainers, models
from tokenizers.normalizers import NFD, Lowercase, Sequence
from tokenizers.pre_tokenizers import Whitespace

current_file = Path(__file__).resolve()
data_dir = current_file.parent
config_file = data_dir.parent / "config" / "config.json"
raw_dir = data_dir / "raw"
processed_dir = data_dir / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

with open(config_file, 'r', encoding='utf-8') as f:
    config = json.load(f)
max_seq_len = config['max_seq_len']
vocab_size = config['vocab_size']

dataset = []
with open(raw_dir / "pre_train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict) and "text" in obj:
            text = obj["text"].strip()
            if text:
                dataset.append(text)

with open(raw_dir / "word.txt", "r", encoding="utf-8") as f:
    sample_words = {line.strip() for line in f if line.strip()}

joined_data = " ".join(dataset)
words_in_data = set(joined_data.split())
valid_sample_words = sorted(sample_words & words_in_data)

print(f"✅ Số từ mẫu còn lại sau khi đối chiếu với dữ liệu: {len(valid_sample_words)}")

tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = Sequence([
    NFD(),
    Lowercase()
])

tokenizer.pre_tokenizer = Whitespace()

initial_alphabet = sorted(set("".join(valid_sample_words)))

trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    min_frequency=2,
    initial_alphabet=initial_alphabet,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
)

tokenizer.train_from_iterator(dataset, trainer=trainer)

tokenizer.save(str(processed_dir / "tokenizer.json"))

vocab = tokenizer.get_vocab()
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

with open(data_dir / "vocab.txt", 'w', encoding='utf-8') as f:
    for token, idx in sorted_vocab:
        f.write(f"{token}\t{idx}\n")

actual_vocab_size = len(vocab)
if actual_vocab_size != vocab_size:
    print(f"📝 Cập nhật vocab_size trong config: {vocab_size} -> {actual_vocab_size}")
    config['vocab_size'] = actual_vocab_size
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Đã cập nhật config.json với vocab_size = {actual_vocab_size}")
else:
    print(f"✅ Vocab_size trong config đã đúng: {vocab_size}")

X, Y, lengths = [], [], []
for line in dataset:
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
