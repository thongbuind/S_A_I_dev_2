import numpy as np
import gc
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, lengths, indices, loss_masks=None):
        self.X = X
        self.Y = Y
        self.lengths = lengths
        self.indices = indices
        self.loss_masks = loss_masks

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        if self.loss_masks is not None:
            return (
                self.X[real_idx],
                self.Y[real_idx],
                self.lengths[real_idx],
                real_idx  # Trả về index để collate_fn lấy loss_mask
            )
        else:
            return (
                self.X[real_idx],
                self.Y[real_idx],
                self.lengths[real_idx],
                real_idx
            )

def split_train_val_test(X, Y, loss_masks, lengths, train_ratio, val_ratio, seed=54):
    """
    Sửa lỗi: Thêm loss_masks vào input và output
    """
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
    
    # Xử lý loss_masks nếu có
    if loss_masks is not None:
        mask_train = loss_masks[train_idx]
        mask_val = loss_masks[val_idx]
        mask_test = loss_masks[test_idx]
        
        return (X_train, Y_train, mask_train, lengths_train, 
                X_val, Y_val, mask_val, lengths_val, 
                X_test, Y_test, mask_test, lengths_test)
    else:
        # Pretrain không có loss_mask
        return (X_train, Y_train, None, lengths_train,
                X_val, Y_val, None, lengths_val,
                X_test, Y_test, None, lengths_test)

def create_dataset(X, Y, lengths, batch_size, shuffle, loss_masks=None):
    log_progress(f"Đang tạo dataset từ {len(X)} samples...")
    indices = np.arange(len(X))
    dataset = Dataset(X, Y, lengths, indices, loss_masks)
    bucket_size = 20
    PAD_ID = 0

    def collate_fn(batch):
        X_batch = [item[0] for item in batch]
        Y_batch = [item[1] for item in batch]
        idx_batch = [item[3] for item in batch]

        max_len = max(len(x) for x in X_batch)
        if max_len > 1000:
            bucket = 1024
        else:
            bucket = int(np.ceil(max_len / bucket_size) * bucket_size)

        bsz = len(X_batch)
        X_padded = torch.zeros((bsz, bucket), dtype=torch.long)
        Y_padded = torch.zeros((bsz, bucket), dtype=torch.long)

        if loss_masks is not None:
            loss_mask_padded = torch.zeros((bsz, bucket), dtype=torch.float)

        for i, (x, y) in enumerate(zip(X_batch, Y_batch)):
            x_len = len(x)
            y_len = len(y)

            X_padded[i, :x_len] = torch.as_tensor(x, dtype=torch.long)
            Y_padded[i, :y_len] = torch.as_tensor(y, dtype=torch.long)

            if loss_masks is not None:
                lm = loss_masks[idx_batch[i]]
                lm_len = len(lm)
                loss_mask_padded[i, :lm_len] = torch.as_tensor(lm, dtype=torch.float)

        attention_mask = (X_padded != PAD_ID).float()

        if loss_masks is None:
            sample_weight = (Y_padded != PAD_ID).float()
        else:
            sample_weight = loss_mask_padded

        return X_padded, Y_padded, sample_weight, attention_mask

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=1,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=False
    )

    log_progress(f"Dataset được tạo với batch_size={batch_size}")
    return dataloader

def get_step_lr_lambda(warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < total_steps * 0.7:
            return 1.0
        else:
            progress = (current_step - total_steps * 0.7) / (total_steps * 0.3)
            return max(0.1, 1.0 - 0.9 * progress)
    return lr_lambda

def load_data(data_type, path_1, path_2=None):
    if data_type == "pretrain":
        data = np.load(path_1, allow_pickle=True)
        X, Y, lengths = data["X"], data["Y"], data["lengths"]
        data.close()
        return X, Y, lengths

    elif data_type == "continued_pretrain":
        cont_data = np.load(path_1, allow_pickle=True)
        X_c, Y_c, L_c = cont_data["X"], cont_data["Y"], cont_data["lengths"]
        cont_data.close()

        if path_2 is not None:
            pre_data = np.load(path_2, allow_pickle=True)
            X_p, Y_p, L_p = pre_data["X"], pre_data["Y"], pre_data["lengths"]
            pre_data.close()

            n_continued = len(X_c)
            n_pretrain_needed = 5 * n_continued
            total_pretrain = len(X_p)
            
            rng = np.random.default_rng(54)
            shuffled_indices = rng.permutation(total_pretrain)
            
            n_samples = min(n_pretrain_needed, total_pretrain)
            selected_indices = shuffled_indices[:n_samples]
            
            X_p_sampled = X_p[selected_indices]
            Y_p_sampled = Y_p[selected_indices]
            L_p_sampled = L_p[selected_indices]
            
            del X_p, Y_p, L_p, shuffled_indices, selected_indices
            gc.collect()
            
            X_combined = np.concatenate([X_c, X_p_sampled])
            Y_combined = np.concatenate([Y_c, Y_p_sampled])
            L_combined = np.concatenate([L_c, L_p_sampled])
            
            combined_indices = rng.permutation(len(X_combined))
            X = X_combined[combined_indices]
            Y = Y_combined[combined_indices]
            lengths = L_combined[combined_indices]
            
        else:
            X, Y, lengths = X_c, Y_c, L_c

        return X, Y, lengths

    elif data_type == "finetune":
        data = np.load(path_1, allow_pickle=True)
        X = data["X"]
        Y = data["Y"]
        loss_mask = data["loss_mask"]
        lengths = data["lengths"]
        data.close()
        return X, Y, loss_mask, lengths

    else:
        raise ValueError(f"Unknown data_type: {data_type}")

def log_progress(text):
    fixed_width = 82
    formatted_text = f"║ {text:<{fixed_width}} ║"
    print(formatted_text)
