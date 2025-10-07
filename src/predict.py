import tensorflow as tf
import numpy as np
from keras import models
import json
import sys
from pathlib import Path
from tokenizers import Tokenizer

current_file = Path(__file__).resolve()
src_dir = current_file.parent
project_root = src_dir.parent
sys.path.append(str(project_root))
config_file = project_root / "config" / "config.json"
vocab_file = project_root/ "data" / "vocab.txt"
model_file = project_root / "model" / "s_a_i.keras"
processed_dir = project_root / "data" / "processed"

model = models.load_model(model_file)

with open(config_file, 'r') as f:
    config = json.load(f)
max_seq_len = config['max_seq_len']

# vocab = {}
# with open(vocab_file, "r", encoding="utf-8") as f:
#     for line in f:
#         word, idx = line.strip().split('\t')
#         vocab[word] = int(idx)

# idx2word = {i: w for w, i in vocab.items()}

tokenizer_file = processed_dir / "tokenizer.json"
tokenizer = Tokenizer.from_file(str(tokenizer_file))
vocab = tokenizer.get_vocab()
idx2word = {i: w for w, i in vocab.items()}

# Special tokens (nên lấy từ config cho chắc)
special_tokens = {
    "PAD": vocab.get("[PAD]", 0),
    "UNK": vocab.get("[UNK]", 1),
    "BOS": vocab.get("[BOS]", 2),
    "EOS": vocab.get("[EOS]", 3),
    "SEP": vocab.get("[SEP]", 4),
}

def tokenize(sentence):
    encoded = tokenizer.encode(sentence.lower())
    tokens = encoded.ids
    
    return tokens

def detokenize(tokens):
    # special_tokens = {0, 1, 2, 3, 4, 5, 6}  # PAD, UNK, BOS, EOS, SEP
    # words = []

    # for t in tokens:
    #     if t in special_tokens or t not in idx2word:
    #         continue
    #     word = idx2word[t]
        
    #     if word in {",", ".", ":", ";", "!", "?"} and words:
    #         words[-1] += word
    #     else:
    #         words.append(word)
    # return " ".join(words)
    return tokenizer.decode(tokens)

def generate_response(sentence, max_new_tokens=50, loss_margin=0.2):
    """
    Tạo phản hồi từ câu đầu vào:
        current_sequence = [BOS] + req
        sequence = loop(predict(current_sequence))

    Sampling strategy:
        - Chọn token có loss thấp nhất
        - Tìm các token có loss không vượt quá best_loss + loss_margin
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

def generate_autoregressive(sentence, max_new_tokens=50, top_k=3):
    req_tokens = tokenizer.encode(sentence).ids
    bos_id = tokenizer.get_vocab()["[BOS]"]

    seed_tokens = [bos_id] + req_tokens
    generated = list(seed_tokens)

    for step in range(max_new_tokens):
        inp_padded = tf.keras.preprocessing.sequence.pad_sequences(
            [generated], maxlen=max_seq_len, padding="post", dtype="int32"
        )

        logits = model(inp_padded, training=False).numpy()
        pred_pos = len(generated) - 1
        log_probs = tf.nn.log_softmax(logits, axis=-1).numpy()[0, pred_pos]

        if top_k == 1:
            next_id = int(np.argmax(log_probs))
        else:
            k = min(top_k, len(log_probs))
            top_k_idx = np.argpartition(-log_probs, k-1)[:k]
            top_k_probs = np.exp(log_probs[top_k_idx])
            top_k_probs = top_k_probs / (top_k_probs.sum() + 1e-12)
            next_id = int(np.random.choice(top_k_idx, p=top_k_probs))

        generated.append(next_id)

        eos_id = tokenizer.get_vocab().get("[EOS]", None)
        pad_id = tokenizer.get_vocab().get("[PAD]", None)
        if (eos_id is not None and next_id == eos_id) or (pad_id is not None and next_id == pad_id):
            break

    return tokenizer.decode(generated)

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
    print(f"Req {i}: {req} \nRes 1 {i}: {generate_response(req)}\nRes 2 {i}: {generate_autoregressive(req)}\n")
    i+=1




# import tensorflow as tf
# import numpy as np
# from keras import models
# import json
# import sys
# from pathlib import Path
# from tokenizers import Tokenizer

# current_file = Path(__file__).resolve()
# src_dir = current_file.parent
# project_root = src_dir.parent
# sys.path.append(str(project_root))
# config_file = project_root / "config" / "config.json"
# model_file = project_root / "model" / "s_a_i.keras"
# processed_dir = project_root / "data" / "processed"

# # Load model
# model = models.load_model(model_file)

# # Load config
# with open(config_file, 'r') as f:
#     config = json.load(f)
# max_seq_len = config['max_seq_len']

# # Load tokenizer
# tokenizer_file = processed_dir / "tokenizer.json"
# tokenizer = Tokenizer.from_file(str(tokenizer_file))
# vocab = tokenizer.get_vocab()
# idx2word = {i: w for w, i in vocab.items()}

# # Special tokens (nên lấy từ config cho chắc)
# special_tokens = {
#     "PAD": vocab.get("[PAD]", 0),
#     "UNK": vocab.get("[UNK]", 1),
#     "BOS": vocab.get("[BOS]", 2),
#     "EOS": vocab.get("[EOS]", 3),
#     "SEP": vocab.get("[SEP]", 4),
# }

# # =====================
# # Tokenize / Detokenize
# # =====================
# def tokenize(sentence):
#     return tokenizer.encode(sentence.lower()).ids

# def detokenize(tokens):
#     words = []
#     for t in tokens:
#         if t in special_tokens.values() or t not in idx2word:
#             continue
#         word = idx2word[t]
#         if word in {",", ".", ":", ";", "!", "?"} and words:
#             words[-1] += word
#         else:
#             words.append(word)
#     return " ".join(words)

# # =====================
# # Response Generation
# # =====================
# def generate_response(sentence, max_new_tokens=50, top_k=3):
#     """
#     Sinh câu trả lời từ mô hình:
#     - Input: [BOS] + req
#     - Mỗi bước predict token tiếp theo
#     - Sampling bằng Top-K
#     """

#     req_tokens = tokenize(sentence)
#     current_sequence = [special_tokens["BOS"]] + req_tokens

#     for step in range(max_new_tokens):
#         # Pad input tại mỗi vòng để tránh lỗi overwrite
#         padded_input = tf.keras.preprocessing.sequence.pad_sequences(
#             [current_sequence], maxlen=max_seq_len, padding="post", dtype="int32"
#         )

#         preds = model(padded_input, training=False)[0, len(current_sequence) - 1, :].numpy()
#         preds = np.clip(preds, 1e-9, 1.0)  # tránh log(0)

#         # Top-k sampling
#         top_k_indices = np.argpartition(-preds, top_k)[:top_k]
#         top_k_probs = preds[top_k_indices]
#         top_k_probs /= np.sum(top_k_probs)  # normalize

#         next_token = np.random.choice(top_k_indices, p=top_k_probs)

#         # Nếu gặp EOS hoặc PAD thì dừng
#         if next_token in [special_tokens["EOS"], special_tokens["PAD"]]:
#             break

#         current_sequence.append(int(next_token))

#         if len(current_sequence) >= max_seq_len:
#             break

#     return detokenize(current_sequence[1:])

# # =====================
# # Kiểm Tra
# # =====================
# if __name__ == "__main__":
#     inputs = [
#         "bánh mì",
#         "việt nam",
#         "việt nam sở hữu",
#         "phở",
#         "buổi sáng người việt nam thường",
#         "đám mây",
#         "Đinh Tiên Hoàng",
#         "Lê Lợi đã",
#         "sau khi lên ngôi",
#         "công thức 1",
#         "sáng hôm ấy",
#         "sau khi ăn xong, chúng tôi",
#         "mặc dù",
#         "bởi vì trời mưa,"
#     ]

#     print("\n=== Test pre-train ===")
#     for i, req in enumerate(inputs, 1):
#         print(f"Req {i}: {req} \nRes {i}: {generate_response(req)}\n")
