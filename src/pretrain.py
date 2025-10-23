import sys
import numpy as np
import json
import tensorflow as tf
from model import Model
from pathlib import Path
import gc
from utils import load_data, log_progress, log_memory_usage

current_file = Path(__file__).resolve()
src_dir = current_file.parent
project_root = src_dir.parent
sys.path.append(str(project_root))
config_file = project_root / "config" / "config.json"
model_folder = project_root / "model"
model_folder.mkdir(parents=True, exist_ok=True)
data_processed_dir = project_root / "data" / "processed"
pretrain_tokenized_file = data_processed_dir / "pretrain_data_shorted_ids.npz"
continued_pretrain_tokenized_file = data_processed_dir / "continued_pretrain_ids.npz"

def split_train_val_test(X, Y, lengths, train_ratio, val_ratio, seed=54):
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
    
    if shuffle:
        total_samples = len(X)
        indices = tf.random.shuffle(tf.range(total_samples))
        
        X_tensor = tf.gather(X_tensor, indices)
        Y_tensor = tf.gather(Y_tensor, indices)
        lengths_tensor = tf.gather(lengths_tensor, indices)
    
    log_progress("Đã convert sang tensors, đang tạo dataset...")
    
    ds = tf.data.Dataset.from_tensor_slices((X_tensor, Y_tensor, lengths_tensor))
    
    bucket_size = 20

    def pad_batch(x_batch, y_batch, lengths):
        max_len = tf.reduce_max(lengths)
    
        bucket = tf.cast(
            tf.math.ceil(tf.cast(max_len, tf.float32) / tf.cast(bucket_size, tf.float32)) * tf.cast(bucket_size, tf.float32),
            tf.int32
        )
        
        x_dense = x_batch.to_tensor(default_value=0)
        y_dense = y_batch.to_tensor(default_value=0)
        
        current_len = tf.shape(x_dense)[1]
        pad_len = tf.maximum(bucket - current_len, 0)
        
        x_padded = tf.pad(x_dense, [[0, 0], [0, pad_len]], constant_values=0)
        y_padded = tf.pad(y_dense, [[0, 0], [0, pad_len]], constant_values=0)

        sample_weight = tf.cast(tf.not_equal(y_padded, 0), tf.float32)

        return x_padded, y_padded, sample_weight
    
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(pad_batch, num_parallel_calls=tf.data.AUTOTUNE, deterministic=not shuffle)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    log_progress(f"Dataset được tạo với batch_size={batch_size}")

    return ds

def pretrain(model, pretrain_tokenized_file, num_epochs, model_folder, train_ratio, val_ratio, batch_size):
    print("╔════════════════════════════════════════════════════════════════════════════════════╗")
    print("║                            BẮT ĐẦU LOAD PRETRAIN DATA                              ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    X, Y, lengths = load_data("pretrain", pretrain_tokenized_file)
    X_train, Y_train, lengths_train, X_val, Y_val, lengths_val, X_test, Y_test, lengths_test = split_train_val_test(X, Y, lengths, train_ratio, val_ratio)
    log_progress(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    train_ds = create_dataset(X_train, Y_train, lengths_train, batch_size, shuffle=True)    
    val_ds = create_dataset(X_val, Y_val, lengths_val, batch_size, shuffle=False)    
    test_ds = create_dataset(X_test, Y_test, lengths_test, batch_size, shuffle=False)

    del X, Y, lengths
    del X_train, Y_train, lengths_train
    del X_val, Y_val, lengths_val  
    del X_test, Y_test, lengths_test
    gc.collect()

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )

    log_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda ep, logs: log_progress(
            f"Epoch {ep+1}/{num_epochs} Train Loss: {logs['loss']:.4f} Val Loss: {logs['val_loss']:.4f}"
        )
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_folder / "s_a_i.keras",
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=num_epochs,
        callbacks=[lr_scheduler, log_callback, checkpoint_cb],
        verbose=1
    )

    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    print("║                               ĐÁNH GIÁ TRÊN TEST SET                               ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    test_loss = model.evaluate(test_ds, verbose=0)
    log_progress(f"Test Loss: {test_loss:.4f}")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    return test_loss

def continued_pretrain(model, continued_pretrain_tokenized_file, pretrain_tokenized_file, num_epochs, model_folder, train_ratio, val_ratio, batch_size):
    print("╔════════════════════════════════════════════════════════════════════════════════════╗")
    print("║                       BẮT ĐẦU LOAD CONTINUED PRETRAIN DATA                         ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    X, Y, lengths = load_data("continued_pretrain", continued_pretrain_tokenized_file, pretrain_tokenized_file)
    X_train, Y_train, lengths_train, X_val, Y_val, lengths_val, X_test, Y_test, lengths_test = split_train_val_test(X, Y, lengths, train_ratio, val_ratio)
    log_progress(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    train_ds = create_dataset(X_train, Y_train, lengths_train, batch_size, shuffle=True)    
    val_ds = create_dataset(X_val, Y_val, lengths_val, batch_size, shuffle=False)    
    test_ds = create_dataset(X_test, Y_test, lengths_test, batch_size, shuffle=False)

    del X, Y, lengths
    del X_train, Y_train, lengths_train
    del X_val, Y_val, lengths_val  
    del X_test, Y_test, lengths_test
    gc.collect()

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )

    log_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda ep, logs: log_progress(
            f"Epoch {ep+1}/{num_epochs} Train Loss: {logs['loss']:.4f} Val Loss: {logs['val_loss']:.4f}"
        )
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_folder / "s_a_i.keras",
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=num_epochs,
        callbacks=[lr_scheduler, log_callback, checkpoint_cb],
        verbose=1
    )

    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    print("║                               ĐÁNH GIÁ TRÊN TEST SET                               ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    test_loss = model.evaluate(test_ds, verbose=0)
    log_progress(f"Test Loss: {test_loss:.4f}")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    return test_loss

### MAIN
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

optimizer = tf.keras.optimizers.Adam(learning_rate)
model = Model(vocab_size, d_model, num_heads, num_layers, ff_dim, max_seq_len, dropout)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    optimizer=optimizer
)

print("╠════════════════════════════════════════════════════════════════════════════════════╣")
print("║                                 BẮT ĐẦU TRAINING                                   ║")
print("╠════════════════════════════════════════════════════════════════════════════════════╣")

pretrain_test_loss = pretrain(
    model, 
    pretrain_tokenized_file,
    num_epochs=epochs, 
    model_folder=model_folder,
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    batch_size=batch_size
)

continued_pretrain_test_loss = continued_pretrain(
    model, 
    continued_pretrain_tokenized_file, 
    pretrain_tokenized_file,
    num_epochs=epochs, 
    model_folder=model_folder,
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    batch_size=batch_size
)

log_progress(f"Hoàn thành training!")
log_progress(f"Pretrain Test Loss: {pretrain_test_loss:.4f}")
log_progress(f"Continued Pretrain Test Loss: {continued_pretrain_test_loss:.4f}")
log_progress(f"Đã lưu model cuối cùng vào: {model_folder / 's_a_i.keras'}")