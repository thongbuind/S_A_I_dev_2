import tensorflow as tf
import numpy as np
from keras import models
import json
import sys
from pathlib import Path
from vncorenlp import VnCoreNLP

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

model_path = project_root / "model" / "s_a_i.keras"
model = models.load_model(model_path)

config_dir = current_file.parent.parent / "config" / "config.json"
with open(config_dir, 'r') as f:
    config = json.load(f)
max_seq_len = config['max_seq_len']

# Đọc vocab
vocab = {}
vocab_path = current_file.parent.parent/ "data" / "vocab.txt"
with open(vocab_path, "r", encoding="utf-8") as f:
    for line in f:
        word, idx = line.strip().split('\t')
        vocab[word] = int(idx)

idx2word = {i: w for w, i in vocab.items()}

VNCORENLP_PATH = "/Users/thongbui.nd/vncorenlp/VnCoreNLP/VnCoreNLP-1.1.1.jar"
annotator = VnCoreNLP(VNCORENLP_PATH, annotators="wseg", max_heap_size='-Xmx2g')

def tokenize(sentence):
    """Chuyển đổi câu thành token số, sử dụng VnCoreNLP để tách từ tiếng Việt"""
    word_segments = annotator.tokenize(sentence.lower())
    words = [word for segment in word_segments for word in segment]
    tokens = [vocab.get(w, vocab["[UNK]"]) for w in words]
    return tokens

def detokenize(tokens):
    special_tokens = {0, 1, 2, 3, 4, 5, 6}  # PAD, UNK, BOS, EOS, SEP
    words = []

    for t in tokens:
        if t in special_tokens or t not in idx2word:
            continue
        word = idx2word[t]
        
        if word in {",", ".", ":", ";", "!", "?"} and words:
            words[-1] += word
        else:
            words.append(word)
    return " ".join(words)

def generate_response(sentence, max_new_tokens=max_seq_len, loss_margin=0.2):
    """
    Tạo phản hồi từ câu đầu vào:
        current_sequence = [BOS] + req
        sequence = loop(predict(current_sequence))

    Sampling strategy:
        - Chọn token có loss thấp nhất
        - Tìm các token có loss không vượt quá best_loss + 0.2
        - Random trong nhóm đó
    """
    req_tokens = tokenize(sentence)
    current_sequence = [vocab["[BOS]"]] + req_tokens

    padded_input = tf.keras.preprocessing.sequence.pad_sequences(
        [current_sequence], maxlen=max_seq_len, padding='post', dtype='int32'
    )

    for step in range(max_new_tokens):
        preds = model(padded_input, training=False)[0, len(current_sequence) - 1, :].numpy()
        preds = np.clip(preds, 1e-9, 1.0)

        token_losses = -np.log(preds)
        best_loss = np.min(token_losses)

        # Tìm các token có loss <= best_loss + margin
        candidate_indices = np.where(token_losses <= best_loss + loss_margin)[0]
        next_token = np.random.choice(candidate_indices)

        if next_token in [vocab["[EOS]"], vocab["[PAD]"]]:
            break

        current_sequence.append(int(next_token))
        padded_input[0, len(current_sequence) - 1] = next_token

        if len(current_sequence) >= max_seq_len:
            break

    return detokenize(current_sequence[1:])

# ================
# Kiểm Tra Mô Hình
# ================

inputs = [
    "bánh mì",
    "việt nam",
    "việt nam sở hữu",
    "phở",
    "buổi sáng người việt nam thường",
    "đám mây",
    "Đinh Tiên Hoàng",
    "Lê Lợi đã",
    "sau khi lên ngôi",
    "công thức 1",
    "sáng hôm ấy",
    "sau khi ăn xong, chúng tôi",
    "mặc dù",
    "bởi vì trời mưa,"
]

print("\n=== Test pre-train ===")
i=1
for req in inputs:
    print(f"Req {i}: {req} \nRes {i}: {generate_response(req)}")
    i+=1
