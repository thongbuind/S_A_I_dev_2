import torch
from pathlib import Path

def get_step_lr_lambda(warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < total_steps * 0.4:
            return 1.0
        else:
            progress = (current_step - total_steps * 0.4) / (total_steps * 0.3)
            return max(0.1, 1.0 - 0.9 * progress)
    return lr_lambda

def freeze_layers(model, layers_to_freeze):
    for idx in layers_to_freeze:
        for param in model.decoder_blocks[idx].parameters():
            param.requires_grad = False
    
def unfreeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = True

def log_progress(text):
    fixed_width = 82
    formatted_text = f"║ {text:<{fixed_width}} ║"
    print(formatted_text)

def save_checkpoint(path: Path, epoch: int, global_step: int, model, optimizer, scheduler, scaler, best_val_loss: float):
    """Save full training state so training can be resumed exactly."""
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_loss": best_val_loss,
    }, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, scaler, device):
    """Load full training state. Returns (start_epoch, global_step, best_val_loss)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    global_step = ckpt["global_step"]
    best_val_loss = ckpt["best_val_loss"]
    log_progress(f"Resumed from checkpoint: epoch {ckpt['epoch']+1}, step {global_step}, best_val_loss {best_val_loss:.5f}")
    return start_epoch, global_step, best_val_loss

