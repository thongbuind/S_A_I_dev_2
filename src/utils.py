import numpy as np
import tensorflow as tf
import psutil, os, gc
from pympler import asizeof

import numpy as np

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

            n = int(0.7 * len(X_p))
            X = np.concatenate([X_c, X_p[:n]])
            Y = np.concatenate([Y_c, Y_p[:n]])
            lengths = np.concatenate([L_c, L_p[:n]])
        else:
            X, Y, lengths = X_c, Y_c, L_c

        return X, Y, lengths

    elif data_type == "finetune":
        data = np.load(path_1, allow_pickle=True)
        input = data["input"]
        response = data["response"]
        input_lengths = data["input_lengths"]
        response_lengths = data["response_lengths"]
        data.close()
        return input, response, input_lengths, response_lengths

    else:
        raise ValueError(f"Unknown data_type: {data_type}")

# def load_pretrain_data(pretrain_tokenized_file):
#     data = np.load(pretrain_tokenized_file, allow_pickle=True)

#     X = data["X"]
#     Y = data["Y"]
#     lengths = data["lengths"]

#     data.close()
    
#     return X, Y, lengths

# def load_finetune_data(finetune_data_file):
#     data = np.load(finetune_data_file, allow_pickle=True)

#     input = data["input"]
#     response = data["response"]
#     input_lengths = data["input_lengths"]
#     response_lengths = data["response_lengths"]

#     data.close()

#     return input, response, input_lengths, response_lengths

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
