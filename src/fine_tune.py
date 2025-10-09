from keras import models
import gc
import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from pretrain import log_progress

current_file = Path(__file__).resolve()
src_dir = current_file.parent
project_root = src_dir.parent
sys.path.append(str(project_root))
config_file = project_root / "config" / "config.json"
vocab_file = project_root/ "data" / "vocab.txt"
model_file = project_root / "model" / "s_a_i.keras"
processed_dir = project_root / "data" / "processed"
model_folder = project_root / "model"
model_folder.mkdir(parents=True, exist_ok=True)

def load_finetune_data():
    finetune_data_path = processed_dir / "finetune_data_ids.npz"
    data = np.load(finetune_data_path, allow_pickle=True)

    input = data["input"]
    response = data["response"]
    input_lengths = data["input_lengths"]
    response_lengths = data["response_lengths"]

    data.close()

    return input, response, input_lengths, response_lengths

def split_train_val_test(input, response, input_lengths, response_lenghts, train_ratio, val_ratio, seed=54):
    total_sample = len(input)
    indices = np.random.default_rng(seed).permutation(total_sample)

    train_end = int(total_sample * train_ratio)
    val_end = int(total_sample * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    input_train, response_train, input_lengths_train, response_lenghts_train = input[train_idx], response[train_idx], input_lengths[train_idx], response_lenghts[train_idx]
    input_val, response_val, input_lengths_val, response_lenghts_val = input[val_idx], response[val_idx], input_lengths[val_idx], response_lenghts[train_idx]
    input_test, response_test, input_lengths_test, response_lenghts_test = input[test_idx], response[test_idx], input_lengths[test_idx], response_lenghts[train_idx]
    
    return (
        input_train, response_train, input_lengths_train, response_lenghts_train,
        input_val, response_val, input_lengths_val, response_lenghts_val,
        input_test, response_test, input_lengths_test, response_lenghts_test
    )
    
def create_dataset(input, response, input_lengths, response_lengths, batch_size, shuffle=False):
    input_tensor = tf.ragged.constant([x for x in input], dtype=tf.int32)
    response_tensor = tf.ragged.constant([y for y in response], dtype=tf.int32)
    input_lengths_tensor = tf.ragged.constant(input_lengths, dtype=tf.int32)
    response_lengths_tensor = tf.ragged.constant(response_lengths, dtype=tf.int32)

    if shuffle:
        total_sample = len(input_tensor)
        indices = tf.random.shuffle(tf.range(total_sample))

        input_tensor = tf.gather(input_tensor, indices)
        response_tensor = tf.gather(response_tensor, indices)
        input_lengths_tensor = tf.gather(input_lengths_tensor, indices)
        response_lengths_tensor = tf.gather(response_lengths_tensor, indices)

    log_progress("Đã convert finetune data sang tensor, đang tạo finetune dataset")

    ds = tf.data.Dataset.from_tensor_slices((input_tensor, response_tensor, input_lengths_tensor, response_lengths_tensor))

    bucket_size = 20

    def pad_batch(input_batch, response_batch, input_lengths, response_lengths):
        max_input_lengths = tf.reduce_max(input_lengths)
        max_response_lengths = tf.reduce_max(response_lengths)

        bucket_input = tf.cast(
            tf.math.ceil(tf.cast(max_input_lengths, tf.float32) / tf.cast(bucket_size, tf.float32)) * tf.cast(bucket_size, tf.float32),
            tf.int32
        )
        bucket_response = tf.cast(
            tf.math.ceil(tf.cast(max_response_lengths, tf.float32) / tf.cast(bucket_size, tf.float32)) * tf.cast(bucket_size, tf.float32),
            tf.int32
        )

        input_dense = input_batch.to_tensor(default_value=0)
        response_dense = response_batch.to_tensor(default_value=0)

        current_input_len = tf.shape(input_dense)[1]
        current_response_len = tf.shape(response_dense)[1]

        pad_input_len = tf.maximum(bucket_input - current_input_len, 0)
        pad_response_len = tf.maximum(bucket_response - current_response_len, 0)

        input_padded = tf.pad(input_dense, [[0, 0], [0, pad_input_len]], constant_values=0)
        response_padded = tf.pad(response_dense, [[0, 0], [0, pad_response_len]], constant_values=0)

        sample_weight = tf.cast(tf.not_equal(response_padded, 0), tf.float32)

        return input_padded, response_padded, sample_weight
    
    ds = ds.batch(batch_size, drop_remainder=True)
    ds.map(pad_batch, num_parallel_calls=tf.data.AUTOTUNE, deterministic=not shuffle)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    log_progress(f"Dataset được tạo với batch_size={batch_size}")

    return ds

def finetune_model(model, train_ds, val_ds, test_ds, epochs, model_folder):
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )

    log_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: log_progress(
            f"Epoch {epoch+1}/{epochs} Train Loss: {logs['loss']:.4f} Val Loss: {logs['val_loss']:.4f}"
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

print("╔════════════════════════════════════════════════════════════════════════════════════╗")
print("║                                 BẮT ĐẦU LOAD DATA                                  ║")
print("╠════════════════════════════════════════════════════════════════════════════════════╣")
input, response, input_lengths, response_lengths = load_finetune_data()

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

input_train, response_train, input_train_lengths, response_train_lengths, input_val, response_val, input_val_lengths, response_val_lengths, input_test, response_test, input_test_lengths, response_test_lengths = split_train_val_test(input, response, input_lengths, response_lengths, train_ratio, val_ratio)
log_progress(f"Train: {len(input_train)}, Val: {len(input_val)}, Test: {len(input_test)}")
train_ds = create_dataset(input_train, response_train, input_train_lengths, response_train_lengths, batch_size, shuffle=True)
val_ds = create_dataset(input_val, response_val, input_val_lengths, response_val_lengths, batch_size, shuffle=False)
test_ds = create_dataset(input_test, response_test, input_test_lengths, response_test_lengths, batch_size, shuffle=False)

del input, response, input_lengths, response_lengths
del input_train, response_train, input_train_lengths, response_train_lengths
del input_val, response_val, input_val_lengths, response_val_lengths
del input_test, response_test, input_test_lengths, response_test_lengths
gc.colect()

model = models.load_model(model_file)


print("╠════════════════════════════════════════════════════════════════════════════════════╣")
print("║                                 BẮT ĐẦU TRAINING                                   ║")
print("╠════════════════════════════════════════════════════════════════════════════════════╣")
final_test_loss = finetune_model(
    model, train_ds, val_ds, test_ds, 
    epochs=epochs, 
    model_folder=model_folder
)

log_progress(f"Hoàn thành training!")
log_progress(f"Đã lưu model cuối cùng vào: {model_folder / 's_a_i.keras'}")
log_progress(f"Test Loss cuối cùng: {final_test_loss:.4f}")
