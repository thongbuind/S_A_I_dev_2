import numpy as np
import json
from vncorenlp import VnCoreNLP
from pathlib import Path
import time

current_file = Path(__file__).resolve()
data_dir = current_file.parent
config_file = data_dir.parent / "config" / "config.json"
vocab_file = data_dir / "vocab.txt"
data_tokenized_dir = data_dir / "processed"  # Lưu folder thay vì file .py
raw_dir = data_dir / "raw"

with open(config_file, 'r') as f:
    config = json.load(f)
max_seq_len = config['max_seq_len']

vocab = {}
with open(vocab_file, "r", encoding="utf-8") as f:
    for line in f:
        word, idx = line.strip().split('\t')
        vocab[word] = int(idx)
idx2word = {i: w for w, i in vocab.items()}

VNCORENLP_PATH = "/Users/thongbui.nd/vncorenlp/VnCoreNLP/VnCoreNLP-1.1.1.jar"
annotator = VnCoreNLP(VNCORENLP_PATH, annotators="wseg", max_heap_size='-Xmx2g')

def tokenize(sentence):
    word_segments = annotator.tokenize(sentence.lower())
    words = [word for segment in word_segments for word in segment]
    tokens = [vocab.get(w, vocab["[UNK]"]) for w in words]
    return tokens

def load_pretrain_dataset(file_path):
    dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        dataset = [item.strip() for item in json_data if isinstance(item, str) and item.strip()]
    return dataset

def prepare_data(pretrain_data, vocab, max_seq_len, batch_size=50):
    X, Y, lengths = [], [], []
    max_retries = 3
    pretrain_samples = int(len(pretrain_data))

    for i in range(0, pretrain_samples, batch_size):
        batch_data = pretrain_data[i:i+batch_size]
        print(f"Đang xử lý batch pretrain {i//batch_size + 1}/{(pretrain_samples + batch_size - 1)//batch_size}")
        for sentence in batch_data:
            retry_count = 0
            while retry_count < max_retries:
                try:
                    tokens = tokenize(sentence)
                    if len(tokens) < 2 or len(tokens) + 2 > max_seq_len:
                        break
                    inp = [vocab["[BOS]"]] + tokens
                    tgt = tokens + [vocab["[EOS]"]]
                    X.append(inp)
                    Y.append(tgt)
                    lengths.append(len(inp))
                    break
                except Exception as e:
                    retry_count += 1
                    print(f"Lỗi khi tokenize (lần thử {retry_count}): {e}")
                    if retry_count < max_retries:
                        print("Đang thử lại...")
                        time.sleep(1)
                    else:
                        print(f"Bỏ qua câu: {sentence[:50]}...")
        time.sleep(0.1)
    return X, Y, lengths

# Tải và chuẩn bị dữ liệu
pretrain_data = load_pretrain_dataset(raw_dir / "pre_train.json")
X, Y, lengths = prepare_data(pretrain_data, vocab, max_seq_len, batch_size=50)

np.set_printoptions(threshold=np.inf)

# Tạo thư mục lưu nếu chưa có
data_tokenized_dir.mkdir(parents=True, exist_ok=True)

# Lưu dữ liệu dưới dạng npy
np.savez_compressed(
    data_tokenized_dir / "data_tokenized.npz",
    X=np.array(X, dtype=object),
    Y=np.array(Y, dtype=object),
    lengths=np.array(lengths)
)

print(f"✅ Đã lưu dữ liệu dynamic padding vào thư mục: {data_tokenized_dir}")
print(f"📊 Tổng số mẫu: {len(X)}")
print(f"📈 Độ dài sequence trung bình: {np.mean(lengths):.2f}")
print(f"📉 Độ dài sequence min/max: {min(lengths)}/{max(lengths)}")

# Gợi ý tích hợp EWC để ngăn catastrophic forgetting
"""
Để ngăn catastrophic forgetting, có thể tích hợp Elastic Weight Consolidation (EWC) trong quá trình huấn luyện:
1. Tính Fisher Information Matrix trên dữ liệu gộp để xác định các trọng số quan trọng.
2. Thêm penalty term vào hàm mất mát:
   L = L_combined + λ/2 * Σ(F_i * (θ_i - θ_i_initial)^2)
3. Ví dụ code EWC trong TensorFlow:

class EWC:
    def __init__(self, model, dataset, lambda_ewc=1.0):
        self.model = model
        self.dataset = dataset
        self.lambda_ewc = lambda_ewc
        self.params = {var.name: var for var in model.trainable_variables}
        self.means = {var.name: tf.identity(var) for var in model.trainable_variables}
        self.precision_matrices = self.compute_fisher()

    def compute_fisher(self):
        precision_matrices = {name: tf.zeros_like(param) for name, param in self.params.items()}
        for x, y in self.dataset:
            with tf.GradientTape() as tape:
                logits = self.model(x)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            for var, grad in zip(self.model.trainable_variables, gradients):
                precision_matrices[var.name] += tf.square(grad) / len(self.dataset)
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for var in model.trainable_variables:
            name = var.name
            loss += tf.reduce_sum(self.precision_matrices[name] * tf.square(var - self.means[name]))
        return self.lambda_ewc * loss

# Huấn luyện với EWC
model = tf.keras.Sequential([...])
ewc = EWC(model, tf.data.Dataset.from_tensor_slices((combined_X, combined_Y)).batch(32))
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
        loss += ewc.penalty(model)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

for epoch in range(num_epochs):
    for x_batch, y_batch in tf.data.Dataset.from_tensor_slices((combined_X, combined_Y)).batch(32):
        loss = train_step(x_batch, y_batch)
"""
