import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import json
import torch
from pathlib import Path
from sklearn.cluster import KMeans
from collections import defaultdict
from tokenizers import Tokenizer
from model import TransformerModel

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

config_file = project_root / "config" / "config.json"
tokenizer_file = project_root / "data" / "processed" / "tokenizer.json"
model_file = project_root / "model" / "s_a_i.pt"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
with open(config_file, "r") as f:
    config = json.load(f)

model = TransformerModel(
    vocab_size=config["vocab_size"],
    d_model=config["d_model"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"],
    ff_dim=config["ff_dim"],
    max_seq_len=config["max_seq_len"],
    dropout=config["dropout"]
)

model.load_state_dict(torch.load(model_file, map_location=DEVICE))
model.to(DEVICE)
model.eval()

tokenizer = Tokenizer.from_file(str(tokenizer_file))

def encode(text):
    return tokenizer.encode(text.lower()).ids

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

@torch.no_grad()
def embed_groups(model, groups):
    group_vectors = {}

    for g in groups:
        ids = encode(g)
        input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)

        hidden = model.forward_hidden(input_ids)
        vec = hidden.mean(dim=1).squeeze(0)   # [D]

        group_vectors[g] = vec

    return group_vectors

@torch.no_grad()
def assign_texts_to_groups(model, texts, group_vectors):
    results = {}

    for text in texts:
        ids = encode(text)
        input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)

        hidden = model.forward_hidden(input_ids)
        text_vec = hidden.mean(dim=1).squeeze(0)

        best_group = None
        best_score = -1.0

        for group_name, group_vec in group_vectors.items():
            score = cosine_similarity(text_vec, group_vec)
            if score > best_score:
                best_score = score
                best_group = group_name

        results[text] = (best_group, best_score)

    return results

from collections import defaultdict

def group_logs(assignments):
    grouped = defaultdict(list)

    for text, (group, score) in assignments.items():
        grouped[group].append((text, score))

    return grouped

def print_group_logs(grouped, all_groups, title="TEXT GROUPING RESULT"):
    for group in all_groups:
        print(f"[{group}]")
        if group in grouped and grouped[group]:
            for text, score in sorted(grouped[group], key=lambda x: -x[1]):
                print(f"  - {text:30s} | score = {score:.3f}")
        else:
            print(f"  (không có text nào)")
        print()

def run_classification(model, texts, groups, title):
    """Run a single classification test"""
    print("=" * 70)
    print(f"Running: {title}")
    print("=" * 70)
    
    group_vectors = embed_groups(model, groups)
    assignments = assign_texts_to_groups(model, texts, group_vectors)
    grouped_logs = group_logs(assignments)
    print_group_logs(grouped_logs, groups, title)

# =========================
# TEST CASES CONFIGURATION
# =========================

TEST_CASES = [
    {
        "title": "Tổng quan",
        "texts": [
            "đất nước",
            "đại cồ việt",
            "nhật bản",
            "hàn quốc",
            "nhà minh",
            "nhà thanh",
            "hoàng đế",
            "vua",
            "đại việt",
            "việt nam",
            "lịch sử",
            "hoàng đế lý công uẩn",
            "thái tổ cao hoàng đế",
            "đinh tiên hoàng đế",
            "triều đại nhà trần",
            "triều đại nhà lý",
            "lê lợi",
            "lê thái tổ",
            "lê nguyên long",
            "thái tổ trần thừa",
            "bún",
            "phở",
            "bánh mì",
            "cháo",
            "ronaldo",
            "messi",
            "quả bóng",
            "bàn thắng",
            "mưa",
            "đám mây",
            "bao nhiêu tuổi",
            "max verstappen",
            "charles leclerc",
            "red bull",
            "f1",
            "công thức 1",
            "giải đua xe",
            "buổi sáng tốt lành",
            "hello",
            "chào buổi sáng"
        ],
        "groups": [
            "quốc gia",
            "hoàng đế",
            "triều đại",
            "đồ ăn",
            "bóng đá",
            "đua xe",
            "thời tiết",
            "chào hỏi"
        ]
    },
    {
        "title": "Test 2: Phân loại triều đại",
        "texts": [
            "đinh tiên hoàng",
            "đinh tiên hoàng đế",
            "đại thắng minh hoàng đế",
            "đinh bộ lĩnh",
            "lý thái tổ",
            "lý công uẩn",
            "lý thái tông",
            "thái tông lý phật mã",
            "lý phật mã",
            "lý nhật tôn",
            "lý thánh tông",
            "lý càn đức",
            "lý nhân tông",
            "thái tổ trần thừa",
            "thái tông trần cảnh",
            "trần thái tông",
            "trần thánh tông",
            "thái tổ cao hoàng đế",
            "lê lợi",
            "lê thái tổ",
            "lê thái tông",
            "thái tông hoàng đế lê nguyên long",
            "lê thánh tông",
            "nhà trần",
            "nhà lý",
            "nhà lê",
            "nhà đinh"
        ],
        "groups": [
            "triều đại nhà đinh",
            "triều đại nhà lý",
            "triều đại hậu lê",
            "triều đại nhà trần"
        ]
    },
    {
        "title": "Test 3: Phân loại nhân vật",
        "texts": [
            "đại thắng minh hoàng đế",
            "đinh tiên hoàng đế",
            "đinh tiên hoàng",
            "lý thái tổ",
            "thái tổ trần thừa",
            "trần thái tổ",
            "trần thái tông",
            "thái tông trần cảnh",
            "trần thánh tông",
            "thái tông hoàng đế lê nguyên long",
            "thái tổ cao hoàng đế",
            "thái tổ cao hoàng đế lê lợi",
            "lê thái tổ",
            "lê thái tông",
            "lê thánh tông",
            "thánh tông thuần hoàng đế",
            "thái tông văn hoàng đế",
            "lý thái tông",
            "thái tông lý phật mã",
            "lý thánh tông",
            "lý nhân tông",
            "quang trung hoàng đế",
            "lê nhân tông",
        ],
        "groups": [
            "đinh bộ lĩnh",
            "lý công uẩn",
            "lê lợi",
            "trần cảnh",
            "trần thừa",
            "trần hoảng",
            "lê nguyên long",
            "lê bang cơ",
            "lê tư thành",
            "lý càn đức",
            "lý nhật tôn",
            "lý phật mã",
            "nguyễn huệ"
        ]
    },
    {
        "title": "Phân loại sự kiện cho các nhân vật",
        "texts": [
            "dẹp loạn 12 sứ quân",
            "khởi nghĩa lam sơn",
            "vô địch f1",
        ],
        "groups": [
            "lê lợi",
            "đinh bộ lĩnh",
            "max verstappen"
        ]
    },
    {
        "title": "Phân loại môn thể thao",
        "texts": [
            "messi",
            "ronaldo",
            "max verstappen",
            "chalers leclerc",
            "hamilton",
            "khung thành",
            "f1",
            "đua xe",
            "fomula one",
            "fomula 1",
            "bàn thắng",
            "grand prix",
            "manchester united",
            "red bull",
            "ferrari",
            "mclaren"
        ],
        "groups": [
            "bóng đá",
            "công thức 1",
        ]
    },
    {
        "title": "Phân loại yêu cầu giao tiếp",
        "texts": [
            "hello",
            "hi",
            "xin chào",
            "chào buổi sáng",
            "ngủ ngon",
            "buổi sáng tốt lành",
            "bạn tên là gì",
            "bạn năm nay bao nhiêu tuổi",
            "huý của thái tổ cao hoàng đế là gì",
            "ai là người đã sáng lập triều đại nhà lý",
            "messi là người nước nào",
            "max verstappen là tay đua của đội nào"
        ],
        "groups": [
            "chào hỏi cơ bản",
            "hỏi thông tin cá nhân",
            "hỏi thông tin lịch sử",
            "hỏi thông tin bóng đá",
            "hỏi thông tin f1"
        ]
    }
]

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    
    print("\n" + "=" * 70)
    print(f"RUNNING {len(TEST_CASES)} CLASSIFICATION TESTS")
    print("=" * 70 + "\n")
    
    for i, test_case in enumerate(TEST_CASES, 1):
        run_classification(
            model=model,
            texts=test_case["texts"],
            groups=test_case["groups"],
            title=test_case["title"]
        )
        
        # Add spacing between tests
        if i < len(TEST_CASES):
            print("\n" * 2)
    
    print("\n" + "=" * 70)
    print("ALL CLASSIFICATIONS COMPLETED")
    print("=" * 70)
