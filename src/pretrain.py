from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import gc
import sys
from utils import log_progress, load_data
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
pretrain_tokenized_file = data_processed_dir / "pretrain_data_shorted_ids.npz"
continued_pretrain_tokenized_file = data_processed_dir / "continued_pretrain_data_ids.npz"

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, lengths, indices):
        self.X = X
        self.Y = Y
        self.lengths = lengths
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return (
            self.X[real_idx],
            self.Y[real_idx],
            self.lengths[real_idx]
        )

def split_train_val_test(X, Y, lengths, train_ratio, val_ratio, seed=54):
    """Giống hệt TF version"""
    total_sample = len(X)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total_sample)

    train_end = int(total_sample * train_ratio)
    val_end = int(total_sample * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train, Y_train, lengths_train = X[train_idx], Y[train_idx], lengths[train_idx]
    X_val, Y_val, lengths_val = X[val_idx], Y[val_idx], lengths[val_idx]
    X_test, Y_test, lengths_test = X[test_idx], Y[test_idx], lengths[test_idx]
    
    return (X_train, Y_train, lengths_train, 
            X_val, Y_val, lengths_val, 
            X_test, Y_test, lengths_test)

def create_dataset(X, Y, lengths, batch_size, shuffle):
    """
    Chuyển từ tf.data.Dataset sang torch.utils.data.DataLoader
    Giữ nguyên logic: RaggedTensor → Bucket padding → Prefetch
    """
    log_progress(f"Đang tạo dataset từ {len(X)} samples...")
    indices = np.arange(len(X))
    log_progress("Đã convert sang tensors, đang tạo dataset...")

    dataset = Dataset(X, Y, lengths, indices)

    bucket_size = 20

    def collate_fn(batch):
        """
        Tương đương TF's pad_batch function
        Input: list of (x, y, length)
        Output: (x_padded, y_padded, sample_weight, attention_mask)
        """
        X_batch = [item[0] for item in batch]
        Y_batch = [item[1] for item in batch]
        lengths_batch = [item[2] for item in batch]
        
        # Tương đương tf.reduce_max(lengths)
        max_len = max(len(x) for x in X_batch)
        
        # Tương đương TF's bucket calculation
        bucket = int(np.ceil(max_len / bucket_size) * bucket_size)
        
        # Convert sang tensor và pad - tương đương x_batch.to_tensor() + tf.pad()
        X_padded = torch.stack([
            torch.nn.functional.pad(
                torch.tensor(x, dtype=torch.long),
                (0, bucket - len(x)),
                value=0
            ) for x in X_batch
        ])
        
        Y_padded = torch.stack([
            torch.nn.functional.pad(
                torch.tensor(y, dtype=torch.long),
                (0, bucket - len(y)),
                value=0
            ) for y in Y_batch
        ])
        
        sample_weight = (Y_padded != 0).float()
        
        attention_mask = (X_padded != 0).float()
        
        return X_padded, Y_padded, sample_weight, attention_mask
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )

    log_progress(f"Dataset được tạo với batch_size={batch_size}")
    return dataloader

def pretrain(model, optimizer, scheduler, device, pretrain_tokenized_file, num_epochs, model_folder, train_ratio, val_ratio, batch_size):
    """
    Chuyển từ TF's model.fit() sang PyTorch manual training loop
    Giữ nguyên logic callbacks: ReduceLROnPlateau, LambdaCallback, ModelCheckpoint
    """
    print("╔════════════════════════════════════════════════════════════════════════════════════╗")
    print("║                            BẮT ĐẦU LOAD PRETRAIN DATA                              ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    X, Y, lengths = load_data("pretrain", pretrain_tokenized_file)
    X_train, Y_train, lengths_train, X_val, Y_val, lengths_val, X_test, Y_test, lengths_test = split_train_val_test(X, Y, lengths, train_ratio, val_ratio)
    log_progress(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    train_ds = create_dataset(X_train, Y_train, lengths_train, batch_size, shuffle=True)    
    val_ds = create_dataset(X_val, Y_val, lengths_val, batch_size, shuffle=False)    
    test_ds = create_dataset(X_test, Y_test, lengths_test, batch_size, shuffle=False)

    del X_train, Y_train, lengths_train
    del X_val, Y_val, lengths_val  
    del X_test, Y_test, lengths_test
    gc.collect()

    criterion = nn.CrossEntropyLoss(reduction='none')
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        batch_count = 0
        total_batches = len(train_ds)
        
        for batch_idx, (X_batch, Y_batch, sample_weight, attention_mask) in enumerate(train_ds):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            sample_weight = sample_weight.to(device)
            attention_mask = attention_mask.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(X_batch, attention_mask=attention_mask)
            
            loss_per_token = criterion(outputs.view(-1, outputs.size(-1)), Y_batch.view(-1))
            
            assert loss_per_token.shape[0] == sample_weight.view(-1).shape[0], \
                f"Shape mismatch: loss {loss_per_token.shape} vs weight {sample_weight.view(-1).shape}"
            
            num_valid_tokens = sample_weight.sum()
            loss = (loss_per_token * sample_weight.view(-1)).sum() / (num_valid_tokens + 1e-8)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                avg_loss = train_loss / batch_count
                if batch_idx == 0:
                    log_progress(f"[DEBUG] Batch 1: valid_tokens={num_valid_tokens.item():.0f}, loss={loss.item():.4f}")
                print(f"\rEpoch {epoch+1}/{num_epochs} - {batch_idx+1}/{total_batches} - loss: {avg_loss:.4f}", end='')
            
            del X_batch, Y_batch, outputs, loss, sample_weight, attention_mask
        
        print()
        train_loss /= len(train_ds)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch, sample_weight, attention_mask in val_ds:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                sample_weight = sample_weight.to(device)
                attention_mask = attention_mask.to(device)
                
                outputs = model(X_batch, attention_mask=attention_mask)
                loss_per_token = criterion(outputs.view(-1, outputs.size(-1)), Y_batch.view(-1))
                
                num_valid_tokens = sample_weight.sum()
                loss = (loss_per_token * sample_weight.view(-1)).sum() / (num_valid_tokens + 1e-8)
                val_loss += loss.item()
                
                del X_batch, Y_batch, outputs, loss, sample_weight, attention_mask
        
        val_loss /= len(val_ds)
        
        log_progress(f"Epoch {epoch+1}/{num_epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")
        
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"Epoch {epoch+1}: ReduceLROnPlateau reducing learning rate to {new_lr}.")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_folder / "s_a_i.pt")
            print(f"Epoch {epoch+1}: val_loss improved from {best_val_loss:.5f} to {val_loss:.5f}, saving model to {model_folder / 's_a_i.pt'}")

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
            loss_per_token = criterion(outputs.view(-1, outputs.size(-1)), Y_batch.view(-1))
            
            num_valid_tokens = sample_weight.sum()
            loss = (loss_per_token * sample_weight.view(-1)).sum() / (num_valid_tokens + 1e-8)
            test_loss += loss.item()
            
            del X_batch, Y_batch, outputs, loss, sample_weight, attention_mask
    
    test_loss /= len(test_ds)
    log_progress(f"Test Loss: {test_loss:.4f}")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    return test_loss

def continued_pretrain(model, optimizer, scheduler, device, continued_pretrain_tokenized_file, pretrain_tokenized_file, num_epochs, model_folder, train_ratio, val_ratio, batch_size):
    """Giống hệt pretrain, chỉ khác data source"""
    print("╔════════════════════════════════════════════════════════════════════════════════════╗")
    print("║                       BẮT ĐẦU LOAD CONTINUED PRETRAIN DATA                         ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    X, Y, lengths = load_data("continued_pretrain", continued_pretrain_tokenized_file, pretrain_tokenized_file)
    X_train, Y_train, lengths_train, X_val, Y_val, lengths_val, X_test, Y_test, lengths_test = split_train_val_test(X, Y, lengths, train_ratio, val_ratio)
    log_progress(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    train_ds = create_dataset(X_train, Y_train, lengths_train, batch_size, shuffle=True)    
    val_ds = create_dataset(X_val, Y_val, lengths_val, batch_size, shuffle=False)    
    test_ds = create_dataset(X_test, Y_test, lengths_test, batch_size, shuffle=False)

    del X_train, Y_train, lengths_train
    del X_val, Y_val, lengths_val  
    del X_test, Y_test, lengths_test
    gc.collect()

    criterion = nn.CrossEntropyLoss(reduction='none')
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        batch_count = 0
        total_batches = len(train_ds)
        
        for batch_idx, (X_batch, Y_batch, sample_weight, attention_mask) in enumerate(train_ds):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            sample_weight = sample_weight.to(device)
            attention_mask = attention_mask.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch, attention_mask=attention_mask)
            
            loss_per_token = criterion(outputs.view(-1, outputs.size(-1)), Y_batch.view(-1))
            
            num_valid_tokens = sample_weight.sum()
            loss = (loss_per_token * sample_weight.view(-1)).sum() / (num_valid_tokens + 1e-8)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                avg_loss = train_loss / batch_count
                print(f"\rEpoch {epoch+1}/{num_epochs} - {batch_idx+1}/{total_batches} - loss: {avg_loss:.4f}", end='')
            
            del X_batch, Y_batch, outputs, loss, sample_weight, attention_mask
        
        print()
        train_loss /= len(train_ds)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch, sample_weight, attention_mask in val_ds:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                sample_weight = sample_weight.to(device)
                attention_mask = attention_mask.to(device)
                
                outputs = model(X_batch, attention_mask=attention_mask)
                loss_per_token = criterion(outputs.view(-1, outputs.size(-1)), Y_batch.view(-1))
                
                num_valid_tokens = sample_weight.sum()
                loss = (loss_per_token * sample_weight.view(-1)).sum() / (num_valid_tokens + 1e-8)
                val_loss += loss.item()
                
                del X_batch, Y_batch, outputs, loss, sample_weight, attention_mask
        
        val_loss /= len(val_ds)
        
        log_progress(f"Epoch {epoch+1}/{num_epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")
        
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"Epoch {epoch+1}: ReduceLROnPlateau reducing learning rate to {new_lr}.")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_folder / "s_a_i.pt")
            print(f"Epoch {epoch+1}: val_loss improved from {best_val_loss:.5f} to {val_loss:.5f}, saving model to {model_folder / 's_a_i.pt'}")

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
            loss_per_token = criterion(outputs.view(-1, outputs.size(-1)), Y_batch.view(-1))
            
            num_valid_tokens = sample_weight.sum()
            loss = (loss_per_token * sample_weight.view(-1)).sum() / (num_valid_tokens + 1e-8)
            test_loss += loss.item()
            
            del X_batch, Y_batch, outputs, loss, sample_weight, attention_mask
    
    test_loss /= len(test_ds)
    log_progress(f"Test Loss: {test_loss:.4f}")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    return test_loss

### MAIN
if __name__ == "__main__":
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_progress(f"Sử dụng device: {device}")
    
    model = TransformerModel(vocab_size, d_model, num_heads, num_layers, ff_dim, max_seq_len, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-6
    )

    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    print("║                                 BẮT ĐẦU TRAINING                                   ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    pretrain_test_loss = pretrain(
        model,
        optimizer,
        scheduler,
        device,
        pretrain_tokenized_file,
        num_epochs=epochs, 
        model_folder=model_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        batch_size=batch_size
    )

    log_progress(f"Đang load best model từ pretrain để tiếp tục training...")
    model.load_state_dict(torch.load(model_dir / "s_a_i.pt", map_location=device))
    model.to(device)

    continued_pretrain_test_loss = continued_pretrain(
        model,
        optimizer,
        scheduler,
        device,
        continued_pretrain_tokenized_file, 
        pretrain_tokenized_file,
        num_epochs=epochs, 
        model_folder=model_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        batch_size=batch_size
    )

    log_progress(f"Hoàn thành training!")
    log_progress(f"Pretrain Test Loss: {pretrain_test_loss:.4f}")
    log_progress(f"Continued Pretrain Test Loss: {continued_pretrain_test_loss:.4f}")
    log_progress(f"Đã lưu model cuối cùng vào: {model_dir / 's_a_i.pt'}")
    