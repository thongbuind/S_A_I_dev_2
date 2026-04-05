from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import json
import gc
import sys
import argparse
from utils.utils import get_step_lr_lambda, log_progress
from utils.Dataset import Dataset, split_train_val_test, load_data
from model import TransformerModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model size: 35M or 100M")
args = parser.parse_args()

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))
config_dir = project_root / "config"
data_dir = project_root / "data"
model_dir = project_root / "model"
src_dir = project_root / "src"
base_config_file = config_dir / "base.json"
model_config_file = config_dir / f"{args.model}.json"
model_dir.mkdir(parents=True, exist_ok=True)
data_processed_dir = project_root / "data" / "processed"
pretrain_tokenized_file = data_processed_dir / "pretrain_data_ids.npz"
continued_pretrain_tokenized_file = data_processed_dir / "continued_pretrain_data_ids.npz"

def train_loop(data_type, tokenized_file, epochs, learning_rate, weight_decay, num_workers, extra_file=None):
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    print("║                                BAT ĐAU LOAD DATA                                   ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    if extra_file is None:
        X, Y, lengths = load_data(data_type, tokenized_file)
    else:
        X, Y, lengths = load_data(data_type, tokenized_file, extra_file)

    X_train, Y_train, _, lengths_train, X_val, Y_val, _, lengths_val, X_test, Y_test, _, lengths_test = split_train_val_test(X, Y, None, lengths, train_ratio, val_ratio)
    log_progress(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    train_ds = Dataset.create_dataloader(X_train, Y_train, lengths_train, batch_size, num_workers, shuffle=True)
    val_ds = Dataset.create_dataloader(X_val, Y_val, lengths_val, batch_size, num_workers, shuffle=False)
    test_ds = Dataset.create_dataloader(X_test, Y_test, lengths_test, batch_size, num_workers, shuffle=False)

    del X_train, Y_train, lengths_train, X_val, Y_val, lengths_val, X_test, Y_test, lengths_test
    gc.collect()

    global optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_steps = (len(train_ds) // accumulation_steps) * epochs
    warmup_steps = len(train_ds) // 4

    lr_lambda = get_step_lr_lambda(warmup_steps, total_steps)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss(reduction='none')

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        batch_count = 0
        total_batches = len(train_ds)
        optimizer.zero_grad()

        for batch_idx, (X_batch, Y_batch, sample_weight, attention_mask) in enumerate(train_ds):
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            with autocast(device_type='cuda', enabled=(scaler is not None)):
                outputs = model(X_batch, attention_mask=attention_mask)

                loss_per_token = criterion(outputs.view(-1, outputs.size(-1)), Y_batch.view(-1))
                num_valid_tokens = sample_weight.sum()
                loss = ((loss_per_token * sample_weight.view(-1)).sum() / (num_valid_tokens + 1e-8))

            scaled_loss = loss / accumulation_steps

            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == total_batches:
                if scaler is not None:
                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()

                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0
                    )
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            train_loss += loss.item()
            batch_count += 1
            current_lr = optimizer.param_groups[0]['lr']

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                avg_loss = train_loss / batch_count
                print(f"\rEpoch {epoch+1}/{epochs} - Step {global_step}/{total_steps} - loss: {avg_loss:.4f} - lr: {current_lr:.2e}", end='')

            del X_batch, Y_batch, outputs, loss, sample_weight, attention_mask

        print()

        train_loss /= len(train_ds)
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, Y_batch, sample_weight, attention_mask in val_ds:
                X_batch = X_batch.to(device, non_blocking=True)
                Y_batch = Y_batch.to(device, non_blocking=True)

                sample_weight = sample_weight.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)

                with autocast(device_type='cuda', enabled=(scaler is not None)):
                    outputs = model(X_batch, attention_mask=attention_mask)

                loss_per_token = criterion(outputs.view(-1, outputs.size(-1)), Y_batch.view(-1))

                num_valid_tokens = sample_weight.sum()
                loss = ((loss_per_token * sample_weight.view(-1)).sum() / (num_valid_tokens + 1e-8))

                val_loss += loss.item()

                del X_batch, Y_batch, outputs, loss, sample_weight, attention_mask

        val_loss /= len(val_ds)
        log_progress(f"Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_dir / "pretrained.pt")
            print(f"Epoch {epoch+1}: val_loss improved to {val_loss:.5f}, saving model")

    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    print("║                               ĐÁNH GIÁ TRÊN TEST SET                               ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for X_batch, Y_batch, sample_weight, attention_mask in test_ds:
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)

            sample_weight = sample_weight.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            with autocast(device_type='cuda', enabled=(scaler is not None)):
                outputs = model(X_batch, attention_mask=attention_mask)

            loss_per_token = criterion(
                outputs.view(-1, outputs.size(-1)),
                Y_batch.view(-1)
            )

            num_valid_tokens = sample_weight.sum()
            loss = ((loss_per_token * sample_weight.view(-1)).sum() / (num_valid_tokens + 1e-8))
            test_loss += loss.item()

            del X_batch, Y_batch, outputs, loss, sample_weight, attention_mask

    test_loss /= len(test_ds)
    log_progress(f"Test Loss: {test_loss:.4f}")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    return test_loss

with open(base_config_file, 'r') as f:
    config = json.load(f)
with open(model_config_file, 'r') as f:
    config.update(json.load(f))
    
vocab_size = config['vocab_size']
max_seq_len = config['max_seq_len']
d_model = config['d_model']
num_heads = config['num_heads']
num_layers = config['num_layers']
ff_dim = config['ff_dim']
dropout = config['dropout']
pretrain_epochs = config['pretrain_epochs']
continued_pretrain_epochs = config['continued_pretrain_epochs']
batch_size = config['batch_size']
train_ratio = config['train_ratio']
val_ratio = config['val_ratio']
pretrain_learning_rate = config['pretrain_learning_rate']
continued_pretrain_learning_rate = config['continued_pretrain_learning_rate']
accumulation_steps = config['accumulation_steps']
pretrain_weight_decay = config['pretrain_weight_decay']
continued_pretrain_weight_decay = config['continued_pretrain_weight_decay']
num_workers = config['num_workers']

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
model = TransformerModel(vocab_size, d_model, num_heads, num_layers, ff_dim, max_seq_len, dropout).to(device)
optimizer = optim.AdamW(model.parameters(), lr=pretrain_learning_rate, weight_decay=pretrain_weight_decay)

print("╠════════════════════════════════════════════════════════════════════════════════════╣")
print("║                                 BẮT ĐẦU TRAINING                                   ║")
print("╠════════════════════════════════════════════════════════════════════════════════════╣")

pretrain_test_loss = train_loop(
    data_type="pretrain",
    tokenized_file=pretrain_tokenized_file,
    epochs=pretrain_epochs,
    learning_rate=pretrain_learning_rate,
    weight_decay=pretrain_weight_decay,
    num_workers=num_workers
)

log_progress("Đang load best model từ pretrain để tiếp tục training...")
model.load_state_dict(torch.load(model_dir / "pretrained.pt", map_location=device))
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=continued_pretrain_learning_rate, weight_decay=continued_pretrain_weight_decay)
log_progress(f"Reset optimizer với learning rate: {continued_pretrain_learning_rate}")

continued_pretrain_test_loss = train_loop(
    data_type="continued_pretrain",
    tokenized_file=continued_pretrain_tokenized_file,
    epochs=continued_pretrain_epochs,
    learning_rate=continued_pretrain_learning_rate,
    weight_decay=continued_pretrain_weight_decay,
    num_workers=num_workers,
    extra_file=pretrain_tokenized_file
)

log_progress(f"Hoàn thành training!")
log_progress(f"Pretrain Test Loss: {pretrain_test_loss:.4f}")
log_progress(f"Continued Pretrain Test Loss: {continued_pretrain_test_loss:.4f}")
