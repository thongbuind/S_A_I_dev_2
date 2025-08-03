import json
from pathlib import Path
from vncorenlp import VnCoreNLP
import time

current_file = Path(__file__).resolve()
data_dir = current_file.parent
raw_dir = data_dir / "raw"
vocab_file = data_dir / "vocab.txt"
config_file = data_dir.parent / "config" / "config.json"

VNCORENLP_PATH = "/Users/thongbui.nd/vncorenlp/VnCoreNLP/VnCoreNLP-1.1.1.jar"
annotator = VnCoreNLP(VNCORENLP_PATH, annotators="wseg", max_heap_size='-Xmx2g')

# ----------- ĐỌC FILE pre_train.json -----------
file1 = raw_dir / "pre_train.json"
texts = []
with open(file1, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            texts.append(line)

# ----------- TẠO VOCABULARY -----------
vocab = set()
batch_size = 50
max_retries = 3

for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    print(f"Đang xử lý batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
    
    for sentence in batch:
        sentence = sentence.lower()
        retry_count = 0
        while retry_count < max_retries:
            try:
                result = annotator.tokenize(sentence)
                for word_list in result:
                    vocab.update(word_list)
                break
            except Exception as e:
                retry_count += 1
                print(f"Lỗi khi tokenize (lần thử {retry_count}): {e}")
                if retry_count < max_retries:
                    print("Đang khởi động lại VnCoreNLP...")
                    try:
                        annotator.close()
                    except:
                        pass
                    time.sleep(2)
                    annotator = VnCoreNLP(VNCORENLP_PATH, annotators="wseg", max_heap_size='-Xmx2g')
                    time.sleep(1)
                else:
                    print(f"Bỏ qua câu: {sentence[:50]}...")
    
    time.sleep(0.1)

special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]", "[BOS]", "[EOS]"]
sorted_vocab = special_tokens + sorted(vocab)
word_to_id = {word: idx for idx, word in enumerate(sorted_vocab)}
 
# ----------- LƯU vocab.txt -----------
with open(vocab_file, "w", encoding="utf-8") as f:
    for word, idx in word_to_id.items():
        f.write(f"{word}\t{idx}\n")

print("✅ Đã tách từ, thêm token đặc biệt và lưu vào vocab.txt")

# ----------- CẬP NHẬT config.json -----------
with open(config_file, "r", encoding="utf-8") as f:
    config = json.load(f)

config["vocab_size"] = len(word_to_id)

with open(config_file, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=4)

print(f"✅ Đã cập nhật 'vocab_size' = {len(word_to_id)} vào config.json")
