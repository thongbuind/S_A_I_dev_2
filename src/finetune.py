from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import json
import gc
import sys
from utils import split_train_val_test, get_step_lr_lambda, create_dataset, log_progress, load_data
from model import TransformerModel

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))
config_dir = project_root / "config"
data_dir = project_root / "data"
model_dir = project_root / "model"
src_dir = project_root / "src"
config_file = config_dir / "config.json"
model_dir.mkdir(parents=True, exist_ok=True)
data_processed_dir = project_root / "data" / "processed"
finetune_tokenized_file = data_processed_dir / "finetune_data_ids.npz"


def finetune(model, optimizer, device, finetune_tokenized_file, num_epochs, model_folder, train_ratio, val_ratio, batch_size):
    print("╔════════════════════════════════════════════════════════════════════════════════════╗")
    print("║                              BẮT ĐẦU LOAD FINETUNE DATA                            ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    X, Y, loss_mask, lengths = load_data("finetune", finetune_tokenized_file)
    
    X_train, Y_train, mask_train, len_train, \
    X_val, Y_val, mask_val, len_val, \
    X_test, Y_test, mask_test, len_test = split_train_val_test(
        X, Y, loss_mask, lengths, train_ratio, val_ratio
    )

    log_progress(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    train_ds = create_dataset(
        X_train, Y_train, len_train,
        batch_size,
        shuffle=True,
        loss_masks=mask_train
    )
    val_ds = create_dataset(
        X_val, Y_val, len_val,
        batch_size,
        shuffle=False,
        loss_masks=mask_val
    )
    test_ds = create_dataset(
        X_test, Y_test, len_test,
        batch_size,
        shuffle=False,
        loss_masks=mask_test
    )

    del X_train, Y_train, mask_train, len_train
    del X_val, Y_val, mask_val, len_val
    del X_test, Y_test, mask_test, len_test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_steps = len(train_ds) * num_epochs
    warmup_steps = len(train_ds) // 5
    lr_lambda = get_step_lr_lambda(warmup_steps, total_steps)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log_progress(f"Step-based LR: warmup={warmup_steps} steps, total={total_steps} steps")

    criterion = nn.CrossEntropyLoss(reduction="none")
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        batch_count = 0
        total_batches = len(train_ds)

        for batch_idx, (X_batch, Y_batch, sample_weight, attention_mask) in enumerate(train_ds):
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(X_batch, attention_mask=attention_mask)
            loss_per_token = criterion(
                outputs.view(-1, outputs.size(-1)),
                Y_batch.view(-1)
            )

            num_valid_tokens = sample_weight.sum()
            loss = (loss_per_token * sample_weight.view(-1)).sum() / (num_valid_tokens + 1e-8)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            batch_count += 1
            global_step += 1

            current_lr = optimizer.param_groups[0]["lr"]

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                avg_loss = train_loss / batch_count
                print(
                    f"\rEpoch {epoch+1}/{num_epochs} "
                    f"- Step {global_step}/{total_steps} "
                    f"- loss: {avg_loss:.4f} "
                    f"- lr: {current_lr:.2e}",
                    end=""
                )

            del X_batch, Y_batch, outputs, loss, sample_weight, attention_mask
            if (batch_idx + 1) % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

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

                outputs = model(X_batch, attention_mask=attention_mask)
                loss_per_token = criterion(
                    outputs.view(-1, outputs.size(-1)),
                    Y_batch.view(-1)
                )

                num_valid_tokens = sample_weight.sum()
                loss = (loss_per_token * sample_weight.view(-1)).sum() / (num_valid_tokens + 1e-8)
                val_loss += loss.item()

                del X_batch, Y_batch, outputs, loss, sample_weight, attention_mask

        val_loss /= len(val_ds)
        log_progress(
            f"Epoch {epoch+1}/{num_epochs} "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_folder / "finetune.pt")
            print(f"Epoch {epoch+1}: val_loss improved to {val_loss:.5f}, saving model")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    print("║                               ĐÁNH GIÁ TRÊN TEST SET                               ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, Y_batch, sample_weight, attention_mask in test_ds:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            sample_weight = sample_weight.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(X_batch, attention_mask=attention_mask)
            loss_per_token = criterion(
                outputs.view(-1, outputs.size(-1)),
                Y_batch.view(-1)
            )

            num_valid_tokens = sample_weight.sum()
            loss = (loss_per_token * sample_weight.view(-1)).sum() / (num_valid_tokens + 1e-8)
            test_loss += loss.item()

            del X_batch, Y_batch, outputs, loss, sample_weight, attention_mask

    test_loss /= len(test_ds)
    log_progress(f"Test Loss: {test_loss:.4f}")
    print("╚════════════════════════════════════════════════════════════════════════════════════╝")

    return test_loss


# ===== MAIN =====
if __name__ == "__main__":
    with open(config_file, "r") as f:
        config = json.load(f)

    vocab_size = config["vocab_size"]
    max_seq_len = config["max_seq_len"]
    d_model = config["d_model"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    ff_dim = config["ff_dim"]
    dropout = config["dropout"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    train_ratio = config["train_ratio"]
    val_ratio = config["val_ratio"]
    finetune_learning_rate = config["finetune_learning_rate"]
    finetune_weight_decay = config["finetune_weight_decay"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_progress(f"Sử dụng device: {device}")
    log_progress("Finetune với masked loss (assistant-only)")

    model = TransformerModel(
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        ff_dim,
        max_seq_len,
        dropout
    ).to(device)

    log_progress("Load model từ continued-pretrain...")
    model.load_state_dict(torch.load(model_dir / "pretrained.pt", map_location=device))
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=finetune_learning_rate * 0.1,
        weight_decay=finetune_weight_decay
    )

    print("╔════════════════════════════════════════════════════════════════════════════════════╗")
    print("║                                 BẮT ĐẦU FINETUNE                                   ║")
    print("╚════════════════════════════════════════════════════════════════════════════════════╝")

    finetune_test_loss = finetune(
        model,
        optimizer,
        device,
        finetune_tokenized_file,
        num_epochs=epochs,
        model_folder=model_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        batch_size=batch_size
    )

    log_progress(f"Finetune Test Loss: {finetune_test_loss:.4f}")
    log_progress(f"Model finetune lưu tại: {model_dir / 'finetune.pt'}")
