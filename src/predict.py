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

tokenizer_file = processed_dir / "tokenizer.json"
tokenizer = Tokenizer.from_file(str(tokenizer_file))
vocab = tokenizer.get_vocab()
idx2word = {i: w for w, i in vocab.items()}

special_tokens = {
    "PAD": vocab.get("[PAD]", 0),
    "UNK": vocab.get("[UNK]", 1),
    "BOS": vocab.get("[BOS]", 2),
    "EOS": vocab.get("[EOS]", 3),
    "SEP": vocab.get("[SEP]", 4),
}

def tokenize(sentence):
    return tokenizer.encode(sentence.lower()).ids

def detokenize(tokens):
    return tokenizer.decode(tokens)

def generate_response_0(sentence, max_new_tokens=100, loss_margin=0.2):
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
        preds = tf.nn.softmax(preds).numpy() # vì đầu ra của model không có softmax
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

def generate_response(sentence, max_new_tokens=100, loss_margin=0.2, lookahead=3):
    req_tokens = tokenize(sentence)
    current_sequence = [vocab["[BOS]"]] + req_tokens

    for step in range(max_new_tokens):
        padded_input = tf.keras.preprocessing.sequence.pad_sequences(
            [current_sequence], maxlen=max_seq_len, padding='post', dtype='int32'
        )
        preds = model(padded_input, training=False)[0, len(current_sequence) - 1, :]
        preds = tf.nn.softmax(preds)
        preds = tf.clip_by_value(preds, 1e-9, 1.0)

        token_losses = -tf.math.log(preds)
        best_loss = tf.reduce_min(token_losses)
        margin_mask = token_losses <= best_loss + loss_margin
        candidate_indices = tf.where(margin_mask)[:, 0].numpy()

        if len(candidate_indices) == 0:
            break

        batch_inputs = []
        for idx in candidate_indices:
            seq = current_sequence + [int(idx)]
            padded_seq = tf.keras.preprocessing.sequence.pad_sequences(
                [seq], maxlen=max_seq_len, padding='post', dtype='int32'
            )[0]
            batch_inputs.append(padded_seq)

        batch_inputs = np.stack(batch_inputs)
        total_losses = np.zeros(len(candidate_indices))

        for i in range(lookahead):
            preds_batch = model(batch_inputs, training=False)
            step_indices = np.array([len(current_sequence) + i - 1 for _ in range(len(candidate_indices))])
            step_indices_tf = tf.stack([tf.range(len(candidate_indices)), tf.cast(step_indices, tf.int32)], axis=1)
            step_preds = tf.gather_nd(preds_batch, step_indices_tf)
            step_preds = tf.nn.softmax(step_preds)
            step_preds = tf.clip_by_value(step_preds, 1e-9, 1.0)
            step_losses = -tf.math.log(step_preds)

            min_losses = tf.reduce_min(step_losses, axis=-1).numpy()
            total_losses += min_losses

            next_tokens = tf.argmin(step_losses, axis=-1).numpy()
            for j, token in enumerate(next_tokens):
                if len(current_sequence) + i < max_seq_len:
                    batch_inputs[j, len(current_sequence) + i] = token

        best_total = np.min(total_losses)
        valid_indices = np.where(total_losses <= best_total + loss_margin)[0]
        chosen_idx = np.random.choice(valid_indices)

        next_token = candidate_indices[chosen_idx]
        if next_token in [vocab["[EOS]"], vocab["[PAD]"]]:
            break

        current_sequence.append(int(next_token))
        if len(current_sequence) >= max_seq_len:
            break

    return detokenize(current_sequence[1:])


# ================
# Kiểm Tra Mô Hình
# ================

inputs = [
    "thái tổ cao hoàng đế",
    "Đinh Tiên Hoàng",
    "bánh mì",
    "việt nam",
    "việt nam sở hữu",
    "phở",
    "buổi sáng người việt nam thường",
    "đám mây",
    "Lê Lợi",
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
    print(f"Req {i}: {req} \nRes {i}: {generate_response(req)}\n")
    i+=1
