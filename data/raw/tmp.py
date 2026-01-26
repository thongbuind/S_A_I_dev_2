import json
from pathlib import Path
import sys

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

raw_dir = project_root / "data" / "raw"
input_path = raw_dir / "shorted_data.jsonl"
output_path = raw_dir / "history_data.jsonl"

keywords = [
    "đinh tiên hoàng đế",
    "lê hoàn",
    "lê đại hành",
    "lý thái tổ",
    "lý công uẩn",
    "lý thái tông",
    "lý phật mã",
    "lý thánh tông",
    "lý nhật tôn",
    "lý nhân tông",
    "lý càn đức",

    "trần thái tổ",
    "trần thừa",
    "trần thái tông",
    "trần cảnh",
    "trần thánh tông",
    "trần hoảng",

    "lê thái tổ",
    "thái tổ cao hoàng đế",
    "lê lợi",
    "lê thái tông",
    "lê nguyên long",
    "lê nhân tông",
    "lê bang cơ",
    "lê nghi dân",
    "thiên hưng đế",
    "lê thánh tông",
    "lê tư thành",

    "quang trung",
    "nguyễn huệ",
]
keywords = [k.lower() for k in keywords]

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "a", encoding="utf-8") as fout:

    for line in fin:
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        text = obj.get("text", "").lower()
        if not text:
            continue

        if any(k in text for k in keywords):
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
