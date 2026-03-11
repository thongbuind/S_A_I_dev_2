import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import json
import sys
from pathlib import Path
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent
project_root = src_dir.parent
sys.path.append(str(project_root))
from tokenizers import Tokenizer
from src.model import TransformerModel
from src.utils.utils import render_chat_box

config_file = project_root / "config" / "config.json"
sft1_file = project_root / "model" / "sft1.pt"
sft2_file = project_root / "model" / "sft2.pt"
data_dir = project_root / "data"

with open(config_file, 'r') as f:
    config = json.load(f)
vocab_size = config['vocab_size']
max_seq_len = config['max_seq_len']
d_model = config['d_model']
num_heads = config['num_heads']
num_layers = config['num_layers']
ff_dim = config['ff_dim']
dropout = config['dropout']

device = torch.device('cuda' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
print(f"🖥️  Sử dụng device: {device}")

tokenizer_file = data_dir / "tokenizer.json"
tokenizer = Tokenizer.from_file(str(tokenizer_file))
vocab = tokenizer.get_vocab()

USER = vocab["<|user|>"]
SAI = vocab["<|s.a.i|>"]
BOS = vocab["[BOS]"]
EOS = vocab["[EOS]"]
PAD = vocab["[PAD]"]

def load_model(model_file):
    model = TransformerModel(vocab_size, d_model, num_heads, num_layers, ff_dim, max_seq_len, dropout)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()
    return model

models = {
    "sft1": {"model": load_model(sft1_file)}
    # "SAI": {"model": load_model(sft2_file)}
}
print(f"✅ Đã load {len(models)} models: {', '.join(models.keys())}")
total_params = 0
for param_name, param in models["sft1"]["model"].named_parameters():
    total_params += param.numel()
        
print(f"👉 Tổng số tham số của model: {total_params:,}")

def pad_sequence(sequence, max_len, padding_value=0):
    if len(sequence) >= max_len:
        return sequence[:max_len]
    return sequence + [padding_value] * (max_len - len(sequence))

def generate_response_prev(model, user_input, max_new_tokens=50, beam_size=10):
    model.eval()
    prompt = " Input: " + user_input
    prompt_ids = tokenizer.encode(prompt).ids
    input = [BOS] + [USER] + prompt_ids + [SAI]
    output_start_idx = len(input)

    beams = [{"seq": list(input), "log_prob": 0.0, "done": False}]
    completed_beams = []

    def normalized_score(beam):
        out_len = max(len(beam["seq"]) - output_start_idx, 1)
        return beam["log_prob"] / (out_len ** 1.0)

    for step in range(max_new_tokens):
        active_beams = [b for b in beams if not b["done"]]
        if not active_beams:
            break

        batch_inputs = [pad_sequence(b["seq"], max_seq_len, padding_value=PAD) for b in active_beams]
        batch_tensor = torch.tensor(batch_inputs, dtype=torch.long, device=device)

        with torch.no_grad():
            logits_batch = model(batch_tensor)

        all_candidates = []
        for b_idx, beam in enumerate(active_beams):
            cur_pos = len(beam["seq"]) - 1
            logits = logits_batch[b_idx, cur_pos, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs = torch.clamp(log_probs, -1e9, 0.0)

            topk_log_probs, topk_tokens = torch.topk(log_probs, beam_size)
            for log_p, token in zip(topk_log_probs.cpu().numpy(), topk_tokens.cpu().numpy()):
                token = int(token)
                new_seq = beam["seq"] + [token]
                new_log_prob = beam["log_prob"] + float(log_p)
                done = token in [EOS, PAD] or len(new_seq) >= max_seq_len
                all_candidates.append({"seq": new_seq, "log_prob": new_log_prob, "done": done})

        all_candidates.sort(key=normalized_score, reverse=True)
        kept  = all_candidates[:beam_size]
        beams = []
        for cand in kept:
            if cand["done"]:
                completed_beams.append(cand)
            else:
                beams.append(cand)

        if len(completed_beams) >= beam_size:
            break

    final_pool = completed_beams if completed_beams else beams
    best_beam  = max(final_pool, key=normalized_score)
    output_tokens = best_beam["seq"][output_start_idx:]
    while output_tokens and output_tokens[-1] in [EOS, PAD]:
        output_tokens.pop()

    return tokenizer.decode(output_tokens)


def generate_response(model, user_input, max_new_tokens=200, beam_size=10, no_repeat_ngram_size=3, repetition_penalty=1.2):
    model.eval()
    prompt = " Input: " + user_input
    prompt_ids = tokenizer.encode(prompt).ids
    input = [BOS] + [USER] + prompt_ids + [SAI]
    output_start_idx = len(input)

    beams = [{"seq": list(input), "log_prob": 0.0, "done": False}]
    completed_beams = []

    def get_banned_tokens(seq):
        """Lấy danh sách token bị cấm do tạo ra n-gram lặp lại."""
        banned = set()
        if no_repeat_ngram_size > 0 and len(seq) >= no_repeat_ngram_size:
            # Lấy (n-1) token cuối làm prefix để kiểm tra
            ngram_prefix = tuple(seq[-(no_repeat_ngram_size - 1):])
            for i in range(len(seq) - no_repeat_ngram_size + 1):
                if tuple(seq[i:i + no_repeat_ngram_size - 1]) == ngram_prefix:
                    banned.add(seq[i + no_repeat_ngram_size - 1])
        return banned

    def apply_repetition_penalty(log_probs, seq):
        """Phạt các token đã xuất hiện trong chuỗi hiện tại."""
        if repetition_penalty == 1.0:
            return log_probs
        log_probs = log_probs.clone()
        unique_tokens = set(seq)
        for token_id in unique_tokens:
            if log_probs[token_id] < 0:
                # log_prob âm → nhân penalty làm âm hơn (phạt nặng hơn)
                log_probs[token_id] *= repetition_penalty
            else:
                log_probs[token_id] /= repetition_penalty
        return log_probs

    def normalized_score(beam):
        out_len = max(len(beam["seq"]) - output_start_idx, 1)
        return beam["log_prob"] / (out_len ** 1.0)

    for step in range(max_new_tokens):
        active_beams = [b for b in beams if not b["done"]]
        if not active_beams:
            break

        batch_inputs = [pad_sequence(b["seq"], max_seq_len, padding_value=PAD) for b in active_beams]
        batch_tensor = torch.tensor(batch_inputs, dtype=torch.long, device=device)

        with torch.no_grad():
            logits_batch = model(batch_tensor)

        all_candidates = []
        for b_idx, beam in enumerate(active_beams):
            cur_pos = len(beam["seq"]) - 1
            logits = logits_batch[b_idx, cur_pos, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs = torch.clamp(log_probs, -1e9, 0.0)

            # Áp dụng repetition penalty
            log_probs = apply_repetition_penalty(log_probs, beam["seq"])

            # Lấy danh sách token bị cấm (no_repeat_ngram)
            banned_tokens = get_banned_tokens(beam["seq"])

            topk_log_probs, topk_tokens = torch.topk(log_probs, beam_size + len(banned_tokens))
            count = 0
            for log_p, token in zip(topk_log_probs.cpu().numpy(), topk_tokens.cpu().numpy()):
                if count >= beam_size:
                    break
                token = int(token)

                # Bỏ qua token bị cấm
                if token in banned_tokens:
                    continue

                new_seq = beam["seq"] + [token]
                new_log_prob = beam["log_prob"] + float(log_p)
                done = token in [EOS, PAD] or len(new_seq) >= max_seq_len
                all_candidates.append({"seq": new_seq, "log_prob": new_log_prob, "done": done})
                count += 1

        all_candidates.sort(key=normalized_score, reverse=True)
        kept  = all_candidates[:beam_size]
        beams = []
        for cand in kept:
            if cand["done"]:
                completed_beams.append(cand)
            else:
                beams.append(cand)

        if len(completed_beams) >= beam_size:
            break

    final_pool = completed_beams if completed_beams else beams
    best_beam  = max(final_pool, key=normalized_score)
    output_tokens = best_beam["seq"][output_start_idx:]
    while output_tokens and output_tokens[-1] in [EOS, PAD]:
        output_tokens.pop()

    return tokenizer.decode(output_tokens)

# if __name__ == "__main__":
    # test_cases = [
        # "chào",
        # "3+5 bằng mấy",
        # "9+8",
        # "hiện hồn lên đây",
        # "formula 1 là gì",
        # "cách làm bánh mì việt nam",
        # "cho tôi thêm thông tin về đội đua f1 ferrari",
        # "max verstappen là ai",
        # "ê sai",
        # "đinh tiên hoàng đế có huý danh là gì",
        # "tên thật của trần thái tổ là gì",
        # "trần nhân tông có tên thật là gì",
        # "giỏi thế",
        # "lê lợi có miếu hiệu là gì",
        # "huý danh của lê thái tông là gì",
        # "ê dậy đi",
        # "hoàng đế lê thánh tông có huý là gì",
        # "miếu hiệu của trần hoảng là gì",
        # "hello sai",
        # "ẩm thực việt nam",
        # "cách nấu phở",
        # "hướng dẫn cho tôi cách nấu cháo gà",
        # "oke cảm ơn",
        # "ừ, ổn rồi đấy"
    # ]

# render_chat_box(test_cases, models, generate_response)



test_cases = [
# ================= CÂU TRÁI NGHĨA =================
"Viết một câu trái nghĩa với câu sau: nam học rất giỏi",
"Hãy viết lại câu có ý nghĩa ngược lại: hôm nay trời rất nóng",
"Tạo câu trái nghĩa với câu sau: cô ấy rất vui",
"Đổi câu sau thành câu trái nghĩa: bài này rất dễ",
"Hãy viết câu mang ý nghĩa ngược lại: căn phòng rất sáng",

# ================= CÂU PHỦ ĐỊNH =================
"Chuyển câu sau sang dạng phủ định: tôi thích ăn cá",
"Hãy viết lại câu sau ở dạng phủ định: anh ấy hiểu bài",
"Đổi câu sau thành câu phủ định: chúng tôi muốn đi chơi",
"Viết lại câu sau nhưng mang nghĩa phủ định: lan đang đọc sách",
"Hãy biến câu sau thành câu phủ định: trời đang mưa",

# ================= CHỦ ĐỘNG → BỊ ĐỘNG =================
"Chuyển câu sau sang thể bị động: Nam giúp mẹ nấu ăn buổi tối",
"Viết lại câu sau ở dạng bị động: Lan đang tưới cây",
"Đổi câu chủ động sau sang bị động: Mẹ nấu cơm",
"Câu sau ở dạng chủ động, hãy đổi sang bị động: Thầy giáo sửa bài",
"Hãy viết lại câu sau theo dạng bị động: Nam trồng cây",

# ================= TỪ ĐỒNG NGHĨA =================
"Viết lại câu sau nhưng dùng từ đồng nghĩa: anh ấy rất thông minh",
"Hãy thay một số từ để câu sau vẫn giữ nguyên ý: cô ấy rất xinh",
"Viết lại câu sau với từ gần nghĩa: bài này rất khó",
"Diễn đạt lại câu sau bằng từ khác nhưng cùng nghĩa: hôm nay tôi rất vui",
"Hãy viết lại câu sau bằng từ đồng nghĩa: căn nhà rất nhỏ",

# ================= NGUYÊN NHÂN → KẾT QUẢ =================
"Viết lại câu sau để thể hiện rõ quan hệ nguyên nhân và kết quả: tôi ở nhà vì trời mưa",
"Diễn đạt lại câu sau theo quan hệ nguyên nhân - kết quả: tôi hiểu bài vì tôi chăm học",
"Viết lại câu sau theo dạng kết quả - nguyên nhân: nam tập thể dục nên nam khỏe",
"Hãy viết lại câu sau nhưng vẫn giữ quan hệ nguyên nhân: trời lạnh nên tôi mặc áo ấm",
"Diễn đạt lại câu sau theo cấu trúc nguyên nhân - kết quả: tôi không trễ học vì tôi dậy sớm",

# ================= NHƯỢNG BỘ =================
"Viết lại câu sau theo quan hệ nhượng bộ: trời mưa nhưng tôi vẫn đi học",
"Diễn đạt lại câu sau nhưng vẫn giữ ý nhượng bộ: bài khó nhưng tôi vẫn làm",
"Hãy viết lại câu sau theo kiểu nhượng bộ: trời lạnh nhưng anh ấy vẫn chạy bộ",
"Đổi cách nói của câu sau nhưng vẫn giữ nghĩa nhượng bộ: tôi mệt nhưng tôi vẫn học",
"Viết lại câu sau với quan hệ nhượng bộ: đường xa nhưng chúng tôi vẫn đi",

# ================= TRỰC TIẾP → GIÁN TIẾP =================
"Chuyển lời nói trực tiếp sau thành gián tiếp: Nam nói: tôi mệt",
"Viết lại câu sau dưới dạng lời nói gián tiếp: Lan nói: tôi thích đọc sách",
"Hãy đổi lời nói sau sang gián tiếp: Mẹ nói: con đi học đi",
"Chuyển câu nói trực tiếp thành gián tiếp: Thầy nói: các em làm bài",
"Viết lại câu sau thành lời gián tiếp: Bạn tôi nói: tôi rất vui",

# ================= MỞ RỘNG NGỮ CẢNH =================
"Hãy viết thêm để câu sau dài hơn: trời đang mưa",
"Mở rộng câu sau bằng cách thêm thông tin: tôi đang ăn cơm",
"Viết thêm vài chi tiết cho câu sau: nam đang học bài",
"Phát triển câu sau thành câu dài hơn: mẹ đang nấu ăn",
"Hãy mở rộng nội dung của câu sau: chúng tôi đang chơi",

# ================= LỖI LOGIC =================
"Câu sau có lỗi logic, hãy sửa lại: trời đang mưa nên đất rất khô",
"Tìm lỗi sai trong câu và viết lại cho hợp lý: vì rất nghèo nên anh ấy có nhiều tiền",
"Câu sau có gì không hợp lý? Hãy sửa lại: trời rất lạnh nên tôi đổ nhiều mồ hôi",
"Phát hiện lỗi logic trong câu sau và viết lại: vì không ăn nên tôi rất no",
"Câu sau sai về ý nghĩa, hãy chỉnh lại: trời rất tối nên mặt trời rất sáng",

# ================= NỐI CÂU NGUYÊN NHÂN - KẾT QUẢ =================
"Dùng từ nối để ghép hai vế sau: trời mưa ; tôi mang áo mưa",
"Kết hợp hai vế sau thành một câu hoàn chỉnh: tôi học chăm ; tôi hiểu bài",
"Hãy nối hai vế sau thành một câu: trời lạnh ; tôi mặc áo ấm",
"Dùng từ nối thích hợp để ghép: tôi đói ; tôi ăn cơm",
"Tạo một câu hoàn chỉnh từ hai vế sau: nam tập thể dục ; nam khỏe",

# ================= ĐIỀU KIỆN - KẾT QUẢ =================
"Nối hai vế sau thành câu điều kiện: bạn học chăm ; bạn sẽ giỏi",
"Hãy ghép hai vế sau bằng quan hệ điều kiện: trời mưa ; tôi sẽ ở nhà",
"Dùng từ nối điều kiện để ghép: bạn đến sớm ; chúng ta đi chơi",
"Tạo câu điều kiện từ hai vế sau: bạn đọc sách ; bạn hiểu bài",
"Kết hợp hai vế sau theo dạng điều kiện: tôi có tiền ; tôi mua sách",

# ================= TƯƠNG PHẢN =================
"Hãy nối hai vế sau theo quan hệ tương phản: trời mưa ; tôi vẫn đi học",
"Ghép hai vế sau bằng từ nối tương phản: bài khó ; tôi vẫn làm",
"Dùng quan hệ tương phản để nối: trời lạnh ; anh ấy mặc áo mỏng",
"Hãy tạo câu tương phản từ hai vế sau: tôi mệt ; tôi vẫn làm việc",
"Ghép hai vế sau theo ý đối lập: đường xa ; chúng tôi vẫn đi",

# ================= TĂNG TIẾN =================
"Nối hai vế sau theo quan hệ tăng tiến: trời không chỉ mưa ; còn có gió",
"Ghép hai vế sau thành câu tăng tiến: anh ấy không chỉ học giỏi ; còn chăm chỉ",
"Dùng cấu trúc tăng tiến để nối: lan không chỉ xinh ; còn tốt bụng",
"Hãy kết hợp hai vế sau theo kiểu tăng tiến: tôi không chỉ đọc sách ; còn viết bài",
"Tạo câu tăng tiến từ hai vế sau: nam không chỉ chạy nhanh ; còn khỏe",

# ================= LỰA CHỌN =================
"Hãy nối hai vế sau thành câu lựa chọn: bạn uống trà ; bạn uống cà phê",
"Ghép hai vế sau theo quan hệ lựa chọn: bạn đi học ; bạn ở nhà",
"Tạo câu có nghĩa lựa chọn từ hai vế sau: chúng ta ăn cơm ; chúng ta ăn phở",
"Hãy kết hợp hai vế sau bằng từ 'hoặc': bạn đọc sách ; bạn xem phim",
"Ghép hai vế sau thành câu lựa chọn: bạn đi xe đạp ; bạn đi bộ",

]

# ================= CHẠY TEST VÀ LƯU MARKDOWN =================

results = []

for i, test in enumerate(test_cases, 1):
    print(f"\n🧪 Test {i}: {test}")
    response_prev = generate_response_prev(models["sft1"]["model"], test)
    response = generate_response(models["sft1"]["model"], test)

    print(f"🤖 response_prev: {response_prev}")
    print(f"🤖 response: {response}")

    results.append({
        "id": i,
        "input": test,
        "response_prev": response_prev,
        "response": response,
    })

# ================= LƯU FILE MARKDOWN =================
output_md = current_file.parent / "test.md"

print("🚀 Bắt đầu ghi file markdown...")

with open(output_md, "w", encoding="utf-8") as f:
    f.write("# Kết quả đánh giá model SFT1\n\n")
    f.write(f"**Tổng số test:** {len(results)}\n\n")

    for i, r in enumerate(results, 1):
        print(f"✍️ Đang ghi Test {i}/{len(results)} (id={r['id']})")

        f.write(f"## Test {r['id']}\n")
        f.write(f"**Input:** {r['input']}\n\n")
        f.write(f"**Response 1:** {r['response_prev']}\n")
        f.write(f"**Response 2:** {r['response']}\n\n")
        f.write("---\n\n")

print(f"\n📄 Đã lưu kết quả vào: {output_md}")
print("✅ Hoàn thành ghi markdown")
