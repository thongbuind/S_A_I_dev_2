import psutil, os, sys
import numpy as np
import json
import tensorflow as tf
from model import Model
from pathlib import Path
import gc
from pympler import asizeof

current_file = Path(__file__).resolve()
src_dir = current_file.parent
project_root = src_dir.parent
sys.path.append(str(project_root))
config_file = project_root / "config" / "config.json"
model_folder = project_root / "model"
model_folder.mkdir(parents=True, exist_ok=True)

def log_progress(text):
    fixed_width = 82
    formatted_text = f"║ {text:<{fixed_width}} ║"
    print(formatted_text)

def log_memory_usage(note="", top_k=20):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_in_mb = mem_info.rss / (1024 ** 2)
    log_progress(f"[MEMORY] {note} | RSS RAM used: {rss_in_mb:.2f} MB")

    all_objects = gc.get_objects()
    var_sizes = []

    for obj in all_objects:
        try:
            size = asizeof.asizeof(obj)
            var_sizes.append((type(obj).__name__, size, repr(obj)[:80]))
        except Exception:
            continue

    var_sizes.sort(key=lambda x: x[1], reverse=True)

    log_progress(f"Top {top_k} Python objects by size:")
    for typename, size, preview in var_sizes[:top_k]:
        log_progress(f"  {typename:<25} {size/1024/1024:.2f} MB | {preview}")

    try:
        cpu_mem = tf.config.experimental.get_memory_info('CPU:0')
        log_progress(f"[TF-Allocator][CPU] Current: {cpu_mem['current']/1024**2:.2f} MB | Peak: {cpu_mem['peak']/1024**2:.2f} MB")
    except Exception:
        pass

    try:
        gpu_mem = tf.config.experimental.get_memory_info('GPU:0')
        log_progress(f"[TF-Allocator][GPU] Current: {gpu_mem['current']/1024**2:.2f} MB | Peak: {gpu_mem['peak']/1024**2:.2f} MB")
    except Exception:
        pass

def load_data():
    """Load configuration and data"""
    sample_ratio = 0.1
    end_index = 277734   # nếu -1 thì là full data
    
    data_tokenized_path = Path(__file__).parent.parent / "data" / "processed" / "data_ids.npz"
    data = np.load(data_tokenized_path, allow_pickle=True)
    total_samples = len(data["X"])

    if total_samples < end_index:
        log_progress("error")

    if end_index != -1:
            data = {
                "X": data["X"][:end_index],
                "Y": data["Y"][:end_index],
                "lengths": data["lengths"][:end_index]
            }

    num_samples_to_keep = int(total_samples * sample_ratio)
    indices_to_keep = np.random.choice(total_samples, size=num_samples_to_keep, replace=False)
    
    reduced_data = {
        "X": data["X"][indices_to_keep],
        "Y": data["Y"][indices_to_keep],
        "lengths": data["lengths"][indices_to_keep]
    }
    data.close()

    X = reduced_data["X"]
    Y = reduced_data["Y"]
    lengths = reduced_data["lengths"]
    reduced_data.clear()
    
    return X, Y, lengths

def split_train_val_test(X, Y, lengths, train_ratio, val_ratio, seed=54):
    """Split data into train, validation, and test sets"""
    total = len(X)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total)

    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train, Y_train, lengths_train = X[train_idx], Y[train_idx], lengths[train_idx]
    X_val, Y_val, lengths_val = X[val_idx], Y[val_idx], lengths[val_idx]
    X_test, Y_test, lengths_test = X[test_idx], Y[test_idx], lengths[test_idx]
    
    return (X_train, Y_train, lengths_train, 
            X_val, Y_val, lengths_val, 
            X_test, Y_test, lengths_test)

def create_dataset(X, Y, lengths, batch_size, shuffle=False):
    """
    Tạo dataset từ tensor - TensorFlow quản lý memory tốt hơn
    Chuyển từ thuần Dynamic sang Bucket padding - padding theo từng mốc (20, 40, 60, ...)
    """
    log_progress(f"Đang tạo dataset từ {len(X)} samples...")
    
    X_tensor = tf.ragged.constant([x for x in X], dtype=tf.int32)
    Y_tensor = tf.ragged.constant([y for y in Y], dtype=tf.int32)
    lengths_tensor = tf.constant(lengths, dtype=tf.int32)
    log_progress("Đã convert sang tensors, đang tạo dataset...")
    
    ds = tf.data.Dataset.from_tensor_slices((X_tensor, Y_tensor, lengths_tensor))

    def pad_batch(x, y, lengths):
        max_len = tf.reduce_max(lengths)

        bucket = tf.cast(
            tf.math.ceil(tf.cast(max_len, tf.float32) / 20.0) * 20.0,
            tf.int32
        )

        x_dense = x.to_tensor(default_value=0)
        y_dense = y.to_tensor(default_value=0)

        cur_len = tf.shape(x_dense)[1]
        pad_len = tf.maximum(bucket - cur_len, 0)

        x_padded = tf.pad(x_dense, [[0, 0], [0, pad_len]])
        y_padded = tf.pad(y_dense, [[0, 0], [0, pad_len]])

        return x_padded, y_padded

    ds = tf.data.Dataset.from_tensor_slices((X_tensor, Y_tensor, lengths_tensor))
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.map(pad_batch, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=100)

    ds = ds.prefetch(tf.data.AUTOTUNE)

    log_progress(f"Dataset được tạo với batch_size={batch_size}")
    return ds

def evaluate_model(model, dataset):
    """Evaluate model on a dataset"""
    total_loss = 0.0
    total_samples = 0
    
    for batch_X, batch_Y in dataset:
        batch_X = tf.cast(batch_X, tf.int32)
        batch_Y = tf.cast(batch_Y, tf.int32)
        
        loss = model.test_on_batch(batch_X, batch_Y)
        batch_size = tf.shape(batch_X)[0]
        
        total_loss += float(loss) * int(batch_size)
        total_samples += int(batch_size)
    
    return total_loss / max(total_samples, 1)

def train_model(model, train_ds, val_ds, test_ds, epochs, model_folder):

    # Callback để giảm learning rate
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )

    # Callback để log progress
    log_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: log_progress(
            f"Epoch {epoch+1}/{epochs} Train Loss: {logs['loss']:.4f} Val Loss: {logs['val_loss']:.4f}"
        )
    )

    # Callback để lưu model sau mỗi epoch
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_folder / "s_a_i.keras",
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[lr_scheduler, log_callback, checkpoint_cb],
        verbose=1
    )

    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    print("║                               ĐÁNH GIÁ TRÊN TEST SET                               ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    test_loss = model.evaluate(test_ds, verbose=0)
    log_progress(f"Final Test Loss: {test_loss:.4f}")
    print("╚════════════════════════════════════════════════════════════════════════════════════╝")

    return test_loss

def main():
    print("╔════════════════════════════════════════════════════════════════════════════════════╗")
    print("║                                 BẮT ĐẦU LOAD DATA                                  ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    X, Y, lengths = load_data()
    # log_memory_usage("Sau khi load data")

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
    learning_rate = config['learning_rate']

    X_train, Y_train, lengths_train, X_val, Y_val, lengths_val, X_test, Y_test, lengths_test = split_train_val_test(X, Y, lengths, train_ratio, val_ratio)
    log_progress(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    train_ds = create_dataset(X_train, Y_train, lengths_train, batch_size, shuffle=True)    
    val_ds = create_dataset(X_val, Y_val, lengths_val, batch_size, shuffle=False)    
    test_ds = create_dataset(X_test, Y_test, lengths_test, batch_size, shuffle=False)
    # log_memory_usage("Sau khi tạo train/val/test dataset")

    del X, Y, lengths
    del X_train, Y_train, lengths_train
    del X_val, Y_val, lengths_val  
    del X_test, Y_test, lengths_test
    gc.collect()

    model = Model(vocab_size, d_model, num_heads, num_layers, ff_dim, max_seq_len, dropout)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer)

    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    print("║                                 BẮT ĐẦU TRAINING                                   ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    final_test_loss = train_model(
        model, train_ds, val_ds, test_ds, 
        epochs=epochs, 
        model_folder=model_folder
    )

    print(f"\nHoàn thành training!")
    print(f"Đã lưu model cuối cùng vào: {model_folder / 's_a_i.keras'}")
    print(f"Test Loss cuối cùng: {final_test_loss:.4f}")

if __name__ == "__main__":
    main()







# class CustomLRScheduler:
#     """Custom learning rate scheduler with validation loss monitoring"""
#     def __init__(self, model, val_dataset, patience=3, factor=0.5, min_lr=1e-6):
#         self.model = model
#         self.val_dataset = val_dataset
#         self.patience = patience
#         self.factor = factor
#         self.min_lr = min_lr
#         self.best_val_loss = float('inf')
#         self.wait = 0
#         self.current_lr = float(model.optimizer.learning_rate)
        
#     def on_epoch_end(self, epoch):
#         """Called at the end of each epoch"""
#         val_loss = evaluate_model(self.model, self.val_dataset)
        
#         if val_loss < self.best_val_loss:
#             self.best_val_loss = val_loss
#             self.wait = 0
#         else:
#             self.wait += 1
            
#         if self.wait >= self.patience:
#             old_lr = self.current_lr
#             self.current_lr = max(self.current_lr * self.factor, self.min_lr)
#             if self.current_lr < old_lr:
#                 print(f"║ Giảm learning rate từ {old_lr:.6f} xuống {self.current_lr:.6f} ║")
#                 tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.current_lr)
#                 self.wait = 0
        
#         return val_loss
#
# def train_model_old(model, train_ds, val_ds, test_ds, epochs=10, model_folder=None, log_every_n_steps=10):
#     lr_scheduler = CustomLRScheduler(model, val_ds)

#     for epoch in range(1, epochs + 1):
#         log_progress(f" ===== Epoch {epoch}/{epochs} =====")
#         epoch_loss = 0.0
#         steps = 0

#         # Training phase
#         for step, (batch_X, batch_Y) in enumerate(train_ds, start=1):
#             # Memory logging
#             # if step % log_every_n_steps == 0:
#             #     log_memory_usage(f"Training step {step} (epoch {epoch})")
#             #     try:
#             #         details = tf.config.experimental.get_memory_info('GPU:0')
#             #         print(f"[GPU] Current: {details['current']/1024**2:.2f} MB, Peak: {details['peak']/1024**2:.2f} MB")
#             #     except Exception:
#             #         pass

#             # Ensure correct dtypes
#             batch_X = tf.cast(batch_X, tf.int32)
#             batch_Y = tf.cast(batch_Y, tf.int32)

#             loss = model.train_on_batch(batch_X, batch_Y)
#             epoch_loss += float(loss)
#             steps += 1

#             # Periodic garbage collection
#             if step % 50 == 0:
#                 gc.collect()

#         avg_train_loss = epoch_loss / max(1, steps)
#         log_progress(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}")

#         val_loss = lr_scheduler.on_epoch_end(epoch)
#         log_progress(f"Epoch {epoch} Val Loss: {val_loss:.4f}")

#         if model_folder:
#             model.save(model_folder / "s_a_i.keras")
#             log_progress(f"Model saved at epoch {epoch}")

#     print("╠════════════════════════════════════════════════════════════════╣")
#     print("║                     ĐÁNH GIÁ TRÊN TEST SET                     ║")
#     print("╠════════════════════════════════════════════════════════════════╣")
#     test_loss = evaluate_model(model, test_ds)
#     print(f"║ Final Test Loss: {test_loss:.4f}                               ║")
#     print("╚════════════════════════════════════════════════════════════════╝")
    
#     return test_loss
