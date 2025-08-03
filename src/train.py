import json
import numpy as np
import tensorflow as tf
from model import Model
import sys
from pathlib import Path

data_tokenized_path = Path(__file__).parent.parent / "data" / "processed" / "data_tokenized.npz"
data = np.load(data_tokenized_path, allow_pickle=True)
X = data["X"]
Y = data["Y"]
lengths = data["lengths"]

current_file = Path(__file__).resolve()
src_dir = current_file.parent
project_root = src_dir.parent
sys.path.append(str(project_root))
config_file = project_root / "config" / "config.json"

with open(config_file, 'r') as f:
    config = json.load(f)
vocab_size = config['vocab_size']
max_seq_len = config['max_seq_len']
d_model = config['d_model']
num_heads = config['num_heads']
num_layers = config['num_layers']
ff_dim = config['ff_dim']
dropout = config['dropout']
epochs = config['epochs']
batch_size = config['batch_size']
train_ratio = config['train_ratio']
val_ratio = config['val_ratio']

def create_dynamic_batch(X, Y, lengths, batch_indices):
    batch_X = [X[i] for i in batch_indices]
    batch_Y = [Y[i] for i in batch_indices]
    batch_lengths = [lengths[i] for i in batch_indices]
    
    max_len_in_batch = max(batch_lengths)
    
    batch_X_padded = tf.keras.preprocessing.sequence.pad_sequences(
        batch_X, maxlen=max_len_in_batch, padding='post'
    )
    batch_Y_padded = tf.keras.preprocessing.sequence.pad_sequences(
        batch_Y, maxlen=max_len_in_batch, padding='post'
    )
    
    return batch_X_padded, batch_Y_padded, batch_lengths

def split_train_val_test(X, Y, lengths, train_ratio, val_ratio):
    """Chia dữ liệu thành train/validation/test set"""
    total_samples = len(X)
    indices = np.random.permutation(total_samples)
    
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    X_train = [X[i] for i in train_indices]
    Y_train = [Y[i] for i in train_indices]
    lengths_train = [lengths[i] for i in train_indices]
    
    X_val = [X[i] for i in val_indices]
    Y_val = [Y[i] for i in val_indices]
    lengths_val = [lengths[i] for i in val_indices]
    
    X_test = [X[i] for i in test_indices]
    Y_test = [Y[i] for i in test_indices]
    lengths_test = [lengths[i] for i in test_indices]
    
    return X_train, Y_train, lengths_train, X_val, Y_val, lengths_val, X_test, Y_test, lengths_test

def evaluate_model(model, X_val, Y_val, lengths_val, batch_size):
    num_val_samples = len(X_val)
    num_val_batches = (num_val_samples + batch_size - 1) // batch_size
    total_loss = 0.0
    total_batches = 0
    
    for i in range(num_val_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_val_samples)
        batch_indices = list(range(start_idx, end_idx))
        
        batch_X, batch_Y, batch_lengths = create_dynamic_batch(X_val, Y_val, lengths_val, batch_indices)
        
        if batch_X.shape[0] < batch_size:
            pad_size = batch_size - batch_X.shape[0]
            current_seq_len = batch_X.shape[1]
            batch_X = np.pad(batch_X, [(0, pad_size), (0, 0)], mode='constant', constant_values=0)
            batch_Y = np.pad(batch_Y, [(0, pad_size), (0, 0)], mode='constant', constant_values=0)
        
        loss = model.test_on_batch(batch_X, batch_Y)
        total_loss += loss
        total_batches += 1
    
    return total_loss / total_batches if total_batches > 0 else float('inf')

X_train, Y_train, lengths_train, X_val, Y_val, lengths_val, X_test, Y_test, lengths_test = split_train_val_test(X, Y, lengths, train_ratio, val_ratio)

class CustomLRScheduler:
    def __init__(self, model, X_val, Y_val, lengths_val, batch_size, patience=3, min_lr=1e-6):
        self.model = model
        self.X_val = X_val
        self.Y_val = Y_val
        self.lengths_val = lengths_val
        self.batch_size = batch_size
        self.patience = patience
        self.min_lr = min_lr
        self.best_val_loss = float('inf')
        self.wait = 0
        self.current_lr = 0.0005
        
    def on_epoch_end(self, epoch):
        val_loss = evaluate_model(self.model, self.X_val, self.Y_val, self.lengths_val, self.batch_size)
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            old_lr = self.current_lr
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
            if self.current_lr < old_lr:
                print(f"║ Giảm learning rate từ {old_lr:.6f} xuống {self.current_lr:.6f} ║")
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.current_lr)
                self.wait = 0
        
        return val_loss

model = Model(vocab_size, d_model, num_heads, num_layers, ff_dim, max_seq_len, dropout)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer)

lr_scheduler = CustomLRScheduler(model, X_val, Y_val, lengths_val, batch_size)

num_train_samples = len(X_train)
num_train_batches = (num_train_samples + batch_size - 1) // batch_size

print("╔═════════════════════════════════════════╗")
print("║            BẮT ĐẦU PRE-TRAIN            ║")
print("╠═════════════════════════════════════════╣")

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    # SHUFFLE TRAIN SET trước mỗi epoch
    train_indices = np.random.permutation(len(X_train))
    X_train_shuffled = [X_train[i] for i in train_indices]
    Y_train_shuffled = [Y_train[i] for i in train_indices]
    lengths_train_shuffled = [lengths_train[i] for i in train_indices]
    
    epoch_train_loss = 0.0
    num_train_samples = len(X_train_shuffled)
    num_train_batches = (num_train_samples + batch_size - 1) // batch_size
    
    for i in range(num_train_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_train_samples)
        batch_indices = list(range(start_idx, end_idx))
        
        batch_X, batch_Y, batch_lengths = create_dynamic_batch(X_train_shuffled, Y_train_shuffled, lengths_train_shuffled, batch_indices)
        
        if batch_X.shape[0] < batch_size:
            pad_size = batch_size - batch_X.shape[0]
            current_seq_len = batch_X.shape[1]
            batch_X = np.pad(batch_X, [(0, pad_size), (0, 0)], mode='constant', constant_values=0)
            batch_Y = np.pad(batch_Y, [(0, pad_size), (0, 0)], mode='constant', constant_values=0)
        
        loss = model.train_on_batch(batch_X, batch_Y)
        epoch_train_loss += loss
    
    avg_train_loss = epoch_train_loss / num_train_batches
    
    val_loss = lr_scheduler.on_epoch_end(epoch)
    current_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
    
    print(f"║ Epoch: {epoch+1:2d}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.4f} ║")
    model_folder = project_root / "model"
    model_folder.mkdir(parents=True, exist_ok=True)
    model.save(model_folder / "s_a_i.keras")

print("╠═════════════════════════════════════════╣")
print("║          ĐÁNH GIÁ TRÊN TEST SET         ║")
print("╠═════════════════════════════════════════╣")

test_loss = evaluate_model(model, X_test, Y_test, lengths_test, batch_size)

print(f"║ Test Loss: {test_loss:.4f}                       ║")
print("╚═════════════════════════════════════════╝")

print(f"Đã lưu model cuối cùng vào: {model_folder / 's_a_i.keras'}")
print(f"Test Loss cuối cùng: {test_loss:.4f}")
