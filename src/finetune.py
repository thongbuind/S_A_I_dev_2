from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import json
import gc
import sys
import argparse
from utils.utils import get_step_lr_lambda, freeze_layers, unfreeze_all_layers, log_progress, load_checkpoint, save_checkpoint
from utils.Dataset import Dataset, split_train_val_test, load_data
from PenaltyEngine import PenaltyEngine, WrongTokenMarginPenalty, WrongTokenEntropyPenalty, FocalOverconfidencePenalty
from model import TransformerModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model size: 35M or 100M")
parser.add_argument(
    "--phase", type=str, required=True,
    choices=["sft1", "sft2", "sft1_resume", "sft2_resume", "full"],
    help="Training phase: sft1 | sft2 | sft1_resume | sft2_resume | full"
)
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
SFT1_data_ids_file = data_processed_dir / "SFT1_data_ids.npz"
SFT2_data_ids_file = data_processed_dir / "SFT2_data_ids.npz"

def _build_val_test_loaders(phase_name, main_data, sub_data, train_ratio, val_ratio, batch_size, num_workers):
    X, Y, loss_mask, lengths = load_data(phase_name, main_data, sub_data)

    X_train, Y_train, mask_train, len_train, \
    X_val, Y_val, mask_val, len_val, \
    X_test, Y_test, mask_test, len_test = split_train_val_test(
        X, Y, loss_mask, lengths, train_ratio, val_ratio
    )

    val_ds = Dataset.create_dataloader(X_val, Y_val, len_val, batch_size, num_workers, shuffle=False, loss_masks=mask_val)
    test_ds = Dataset.create_dataloader(X_test, Y_test, len_test, batch_size, num_workers, shuffle=False, loss_masks=mask_test)

    train_size = len(X_train)
    val_size = len(X_val)
    test_size = len(X_test)

    del X_train, Y_train, mask_train, len_train
    del X_val, Y_val, mask_val, len_val
    del X_test, Y_test, mask_test, len_test
    del X, Y, loss_mask, lengths
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return val_ds, test_ds, train_size, val_size, test_size

def _build_train_loader_epoch(phase_name, main_data, sub_data, train_ratio, val_ratio, batch_size, num_workers, epoch=0):
    X, Y, loss_mask, lengths = load_data(phase_name, main_data, sub_data, seed=epoch)

    X_train, Y_train, mask_train, len_train, \
    X_val, Y_val, mask_val, len_val, \
    X_test, Y_test, mask_test, len_test = split_train_val_test(
        X, Y, loss_mask, lengths, train_ratio, val_ratio
    )

    train_ds = Dataset.create_dataloader(X_train, Y_train, len_train, batch_size, num_workers, shuffle=True, loss_masks=mask_train)

    del X_val, Y_val, mask_val, len_val
    del X_test, Y_test, mask_test, len_test
    del X_train, Y_train, mask_train, len_train
    del X, Y, loss_mask, lengths
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return train_ds

def finetune(model, optimizer, device, main_data, sub_data, num_epochs, model_save_path, train_ratio, val_ratio, batch_size, num_workers, phase_name, penalty_engine, resample_per_epoch=False, resume_checkpoint_path: Path = None):
    print(f"╠════════════════════════════════════════════════════════════════════════════════════╣")
    print(f"║                               BẮT ĐẦU LOAD {phase_name.upper():<4} DATA                               ║")
    print(f"╠════════════════════════════════════════════════════════════════════════════════════╣")

    if not resample_per_epoch:
        X, Y, loss_mask, lengths = load_data(phase_name, main_data, sub_data)

        X_train, Y_train, mask_train, len_train, \
        X_val, Y_val, mask_val, len_val, \
        X_test, Y_test, mask_test, len_test = split_train_val_test(
            X, Y, loss_mask, lengths, train_ratio, val_ratio
        )

        log_progress(f"[{phase_name}] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        train_ds = Dataset.create_dataloader(X_train, Y_train, len_train, batch_size, num_workers, shuffle=True, loss_masks=mask_train)
        val_ds = Dataset.create_dataloader(X_val, Y_val, len_val, batch_size, num_workers, shuffle=False, loss_masks=mask_val)
        test_ds = Dataset.create_dataloader(X_test, Y_test, len_test, batch_size, num_workers, shuffle=False, loss_masks=mask_test)

        del X_train, Y_train, mask_train, len_train
        del X_val, Y_val, mask_val, len_val
        del X_test, Y_test, mask_test, len_test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        total_steps = len(train_ds) * num_epochs

    else:
        val_ds, test_ds, train_size, val_size, test_size = _build_val_test_loaders(
            phase_name, main_data, sub_data, train_ratio, val_ratio, batch_size, num_workers
        )
        log_progress(f"[{phase_name}] Train: ~{train_size}, Val: {val_size}, Test: {test_size}")

        train_ds = _build_train_loader_epoch(phase_name, main_data, sub_data, train_ratio, val_ratio, batch_size, num_workers, epoch=0)
        steps_per_epoch = len(train_ds)
        total_steps = steps_per_epoch * num_epochs

    warmup_steps = int(total_steps * 0.15)
    lr_lambda = get_step_lr_lambda(warmup_steps, total_steps)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    log_progress(f"Step-based LR: warmup={warmup_steps} steps, total={total_steps} steps")

    criterion = nn.CrossEntropyLoss(reduction="none")
    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 0
    use_penalty = penalty_engine is not None and len(penalty_engine.rules) > 0
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    checkpoint_path = model_save_path.with_suffix(".ckpt.pt")
    if resume_checkpoint_path is not None:
        if resume_checkpoint_path.exists():
            start_epoch, global_step, best_val_loss = load_checkpoint(
                resume_checkpoint_path, model, optimizer, scheduler, scaler, device
            )
        else:
            log_progress(f"[WARNING] Checkpoint not found at {resume_checkpoint_path}. Starting from scratch.")

    for epoch in range(start_epoch, num_epochs):

        if resample_per_epoch and epoch > 0:
            log_progress(f"[{phase_name}] Epoch {epoch+1}: Re-sampling train data...")
            train_ds = _build_train_loader_epoch(
                phase_name, main_data, sub_data, train_ratio, val_ratio, batch_size, num_workers, epoch
            )

        model.train()
        train_loss = 0.0
        batch_count = 0
        total_batches = len(train_ds)

        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (X_batch, Y_batch, sample_weight, attention_mask) in enumerate(train_ds):
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                outputs = model(X_batch, attention_mask=attention_mask)
                loss_per_token = criterion(outputs.view(-1, outputs.size(-1)), Y_batch.view(-1))

                num_valid_tokens = sample_weight.sum()
                ce_loss = (loss_per_token * sample_weight.view(-1)).sum() / (num_valid_tokens + 1e-8)

                if use_penalty:
                    penalty = penalty_engine(logits=outputs, inputs=X_batch, targets=Y_batch, loss_mask=sample_weight)
                    loss = ce_loss + penalty
                else:
                    loss = ce_loss

            scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == total_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            train_loss += loss.item()
            batch_count += 1
            current_lr = optimizer.param_groups[0]["lr"]

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                avg_loss = train_loss / batch_count
                penalty_str = f" | penalty: {penalty.item():.4f}" if use_penalty else ""
                print(
                    f"\r{phase_name} | Epoch {epoch+1}/{num_epochs} "
                    f"- Step {global_step}/{total_steps} "
                    f"- loss: {avg_loss:.4f}{penalty_str} "
                    f"- lr: {current_lr:.2e}",
                    end=""
                )

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

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                    outputs = model(X_batch, attention_mask=attention_mask)
                    loss_per_token = criterion(outputs.view(-1, outputs.size(-1)), Y_batch.view(-1))
                num_valid_tokens = sample_weight.sum()
                loss = (loss_per_token * sample_weight.view(-1)).sum() / (num_valid_tokens + 1e-8)
                val_loss += loss.item()

        val_loss /= len(val_ds)
        log_progress(f"{phase_name} Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss (CE only): {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Epoch {epoch+1}: val_loss improved to {val_loss:.5f}, saving model")

        save_checkpoint(
            checkpoint_path, epoch, global_step,
            model, optimizer, scheduler, scaler, best_val_loss
        )
        log_progress(f"Checkpoint saved → {checkpoint_path}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"╠════════════════════════════════════════════════════════════════════════════════════╣")
    print(f"║                            ĐÁNH GIÁ {phase_name.upper():<4} TRÊN TEST SET                             ║")
    print(f"╠════════════════════════════════════════════════════════════════════════════════════╣")

    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, Y_batch, sample_weight, attention_mask in test_ds:
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                outputs = model(X_batch, attention_mask=attention_mask)
                loss_per_token = criterion(outputs.view(-1, outputs.size(-1)), Y_batch.view(-1))
            num_valid_tokens = sample_weight.sum()
            loss = (loss_per_token * sample_weight.view(-1)).sum() / (num_valid_tokens + 1e-8)
            test_loss += loss.item()

    test_loss /= len(test_ds)
    log_progress(f"{phase_name} Test Loss (CE only): {test_loss:.4f}")
    print(f"╠════════════════════════════════════════════════════════════════════════════════════╣")

    return test_loss

with open(base_config_file, 'r') as f:
    config = json.load(f)
with open(model_config_file, 'r') as f:
    config.update(json.load(f))

vocab_size = config["vocab_size"]
max_seq_len = config["max_seq_len"]
d_model = config["d_model"]
num_heads = config["num_heads"]
num_layers = config["num_layers"]
ff_dim = config["ff_dim"]
dropout = config["dropout"]
sft1_epochs = config["sft1_epochs"]
sft2_epochs = config["sft2_epochs"]
batch_size = config["batch_size"]
train_ratio = config["train_ratio"]
val_ratio = config["val_ratio"]
sft1_learning_rate = config["sft1_learning_rate"]
sft1_learning_weight_decay = config["sft1_learning_weight_decay"]
sft2_learning_rate = config["sft2_learning_rate"]
accumulation_steps = config["accumulation_steps"]
sft2_learning_weight_decay = config["sft2_learning_weight_decay"]
freeze = config["freeze"]
num_workers = config["num_workers"]
penalty_engine = (PenaltyEngine()
    .add_rule(WrongTokenMarginPenalty(weight=config["penalty_margin_weight"], detach_max=config["penalty_margin_detach_max"]))
    .add_rule(WrongTokenEntropyPenalty(weight=config["penalty_entropy_weight"], min_entropy=config["penalty_entropy_min_entropy"]))
    .add_rule(FocalOverconfidencePenalty(weight=config["penalty_focal_weight"], gamma=config["penalty_focal_gamma"]))
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_progress(f"Sử dụng device: {device}")

model = TransformerModel(vocab_size, d_model, num_heads, num_layers, ff_dim, max_seq_len, dropout).to(device)

phase = args.phase
if phase == "sft1":
    log_progress("Load model từ continued-pretrain...")
    model.load_state_dict(torch.load(model_dir / "continued_pretrained.pt", map_location=device))
    freeze_layers(model, freeze)

    optimizer_sft1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=sft1_learning_rate, weight_decay=sft1_learning_weight_decay
    )

    test_loss = finetune(
        model, optimizer_sft1, device, SFT1_data_ids_file, sub_data=None,
        num_epochs=sft1_epochs,
        model_save_path=model_dir / "sft1.pt",
        train_ratio=train_ratio, val_ratio=val_ratio,
        batch_size=batch_size, phase_name="sft1",
        num_workers=num_workers,
        penalty_engine=penalty_engine,
        resample_per_epoch=False,
        resume_checkpoint_path=None,
    )
    log_progress(f"SFT1 Test Loss: {test_loss:.4f}")

elif phase == "sft1_resume":
    log_progress("SFT1 resume: khởi tạo model skeleton trước khi load checkpoint...")
    freeze_layers(model, freeze)

    optimizer_sft1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=sft1_learning_rate, weight_decay=sft1_learning_weight_decay
    )

    test_loss = finetune(
        model, optimizer_sft1, device, SFT1_data_ids_file, sub_data=None,
        num_epochs=sft1_epochs,
        model_save_path=model_dir / "sft1.pt",
        train_ratio=train_ratio, val_ratio=val_ratio,
        batch_size=batch_size, phase_name="sft1",
        num_workers=num_workers,
        penalty_engine=penalty_engine,
        resample_per_epoch=False,
        resume_checkpoint_path=model_dir / "sft1.ckpt.pt",
    )
    log_progress(f"SFT1 Test Loss: {test_loss:.4f}")

elif phase == "sft2":
    log_progress("Load model từ sft1...")
    model.load_state_dict(torch.load(model_dir / "sft1.pt", map_location=device))
    unfreeze_all_layers(model)
    freeze_layers(model, freeze)

    optimizer_sft2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=sft2_learning_rate, weight_decay=sft2_learning_weight_decay
    )

    test_loss = finetune(
        model, optimizer_sft2, device, SFT2_data_ids_file, SFT1_data_ids_file,
        num_epochs=sft2_epochs,
        model_save_path=model_dir / "sft2.pt",
        train_ratio=train_ratio, val_ratio=val_ratio,
        batch_size=batch_size, phase_name="sft2",
        num_workers=num_workers,
        penalty_engine=penalty_engine,
        resample_per_epoch=True,
        resume_checkpoint_path=None,
    )
    log_progress(f"SFT2 Test Loss: {test_loss:.4f}")

elif phase == "sft2_resume":
    log_progress("SFT2 resume: khởi tạo model skeleton trước khi load checkpoint...")
    unfreeze_all_layers(model)
    freeze_layers(model, freeze)

    optimizer_sft2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=sft2_learning_rate, weight_decay=sft2_learning_weight_decay
    )

    test_loss = finetune(
        model, optimizer_sft2, device, SFT2_data_ids_file, SFT1_data_ids_file,
        num_epochs=sft2_epochs,
        model_save_path=model_dir / "sft2.pt",
        train_ratio=train_ratio, val_ratio=val_ratio,
        batch_size=batch_size, phase_name="sft2",
        num_workers=num_workers,
        penalty_engine=penalty_engine,
        resample_per_epoch=True,
        resume_checkpoint_path=model_dir / "sft2.ckpt.pt",
    )
    log_progress(f"SFT2 Test Loss: {test_loss:.4f}")

elif phase == "full":
    # ── SFT1 ──
    log_progress("Load model từ continued-pretrain...")
    model.load_state_dict(torch.load(model_dir / "continued_pretrained.pt", map_location=device))
    freeze_layers(model, freeze)

    optimizer_sft1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=sft1_learning_rate, weight_decay=sft1_learning_weight_decay
    )

    test_loss = finetune(
        model, optimizer_sft1, device, SFT1_data_ids_file, sub_data=None,
        num_epochs=sft1_epochs,
        model_save_path=model_dir / "sft1.pt",
        train_ratio=train_ratio, val_ratio=val_ratio,
        batch_size=batch_size, phase_name="sft1",
        num_workers=num_workers,
        penalty_engine=penalty_engine,
        resample_per_epoch=False,
        resume_checkpoint_path=None,
    )
    log_progress(f"SFT1 Test Loss: {test_loss:.4f}")

    # ── SFT2 ──
    log_progress("Load model từ sft1...")
    model.load_state_dict(torch.load(model_dir / "sft1.pt", map_location=device))
    unfreeze_all_layers(model)
    freeze_layers(model, freeze)

    optimizer_sft2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=sft2_learning_rate, weight_decay=sft2_learning_weight_decay
    )

    test_loss = finetune(
        model, optimizer_sft2, device, SFT2_data_ids_file, SFT1_data_ids_file,
        num_epochs=sft2_epochs,
        model_save_path=model_dir / "sft2.pt",
        train_ratio=train_ratio, val_ratio=val_ratio,
        batch_size=batch_size, phase_name="sft2",
        num_workers=num_workers,
        penalty_engine=penalty_engine,
        resample_per_epoch=True,
        resume_checkpoint_path=None,
    )
    log_progress(f"SFT2 Test Loss: {test_loss:.4f}")

log_progress(f"Model cuối cùng lưu tại: {model_dir / (phase.split('_')[0] + '.pt')}")
