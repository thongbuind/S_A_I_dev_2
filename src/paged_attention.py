"""
PagedAttention — quản lý KV cache theo block/page thay vì pre-allocate liên tục.

Ý tưởng gốc từ vLLM (Kwon et al., 2023):
  - Bộ nhớ GPU được chia thành các "physical block" có kích thước cố định.
  - Mỗi sequence được gán một danh sách "logical block" → map sang physical block.
  - Nhiều sequence có thể share physical block (prefix caching / beam search).

Các thành phần:
  BlockTable      — ánh xạ logical → physical block cho một sequence
  PagedKVPool     — bộ nhớ vật lý chứa toàn bộ K/V
"""
import torch
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class BlockTable:
    """
    Ánh xạ logical block index → physical block index cho một sequence.
    logical_blocks[i] = physical_block_id
    """
    logical_blocks: List[int] = field(default_factory=list)

    def num_blocks(self) -> int:
        return len(self.logical_blocks)

    def append_block(self, physical_id: int):
        self.logical_blocks.append(physical_id)

    def physical_ids(self) -> List[int]:
        return self.logical_blocks


class PagedKVPool:
    """
    Pre-allocate một pool gồm `num_blocks` physical block.
    Mỗi block chứa `block_size` token slot cho tất cả các layer.

    Layout tensor:
        k_cache[layer_id] : (num_blocks, num_heads, block_size, head_dim)
        v_cache[layer_id] : (num_blocks, num_heads, block_size, head_dim)

    free_blocks : set các physical block id chưa được dùng.
    """

    def __init__(
        self,
        num_layers:  int,
        num_heads:   int,
        head_dim:    int,
        block_size:  int,
        num_blocks:  int,
        device:      torch.device,
        dtype:       torch.dtype = torch.float16,
    ):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_layers = num_layers

        shape = (num_blocks, num_heads, block_size, head_dim)
        self.k_cache = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)]
        self.v_cache = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)]

        self.free_blocks: set = set(range(num_blocks))

    def allocate_block(self) -> int:
        """Cấp phát một physical block. Raise nếu hết."""
        if not self.free_blocks:
            raise RuntimeError("PagedKVPool: hết physical block — cần evict hoặc tăng num_blocks")
        return self.free_blocks.pop()

    def free_block(self, physical_id: int):
        """Trả block về pool."""
        self.free_blocks.add(physical_id)

    def free_sequence(self, block_table: BlockTable):
        """Giải phóng toàn bộ block của một sequence."""
        for pid in block_table.physical_ids():
            self.free_block(pid)
        block_table.logical_blocks.clear()

    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def copy_blocks(self, src_physical_ids: List[int], dst_physical_ids: List[int]):
        """
        Copy nội dung K/V từ src sang dst cho tất cả các layer.
        Dùng cho beam reorder: khi beam search chọn lại beam src,
        cần nhân bản physical block sang beam khác.
        src_physical_ids[i] → dst_physical_ids[i]
        """
        for layer_id in range(self.num_layers):
            for src, dst in zip(src_physical_ids, dst_physical_ids):
                self.k_cache[layer_id][dst].copy_(self.k_cache[layer_id][src])
                self.v_cache[layer_id][dst].copy_(self.v_cache[layer_id][src])

    def write_slot(
        self,
        layer_id:    int,
        block_table: BlockTable,
        token_pos:   int,       # vị trí tuyệt đối trong sequence
        k:           torch.Tensor,  # (num_heads, head_dim)
        v:           torch.Tensor,
    ):
        """Ghi K/V của một token vào đúng physical slot."""
        logical_block  = token_pos // self.block_size
        slot_in_block  = token_pos %  self.block_size

        # Tự động cấp phát block mới nếu cần
        while logical_block >= block_table.num_blocks():
            pid = self.allocate_block()
            block_table.append_block(pid)

        physical_block = block_table.logical_blocks[logical_block]
        self.k_cache[layer_id][physical_block, :, slot_in_block, :] = k
        self.v_cache[layer_id][physical_block, :, slot_in_block, :] = v

    def gather_kv(
        self,
        layer_id:    int,
        block_table: BlockTable,
        seq_len:     int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Thu thập toàn bộ K/V của một sequence theo block_table.
        Trả về:
            k : (1, num_heads, seq_len, head_dim)
            v : (1, num_heads, seq_len, head_dim)
        """
        kc = self.k_cache[layer_id]
        vc = self.v_cache[layer_id]

        k_list, v_list = [], []
        tokens_left = seq_len

        for pid in block_table.physical_ids():
            slots = min(self.block_size, tokens_left)
            k_list.append(kc[pid, :, :slots, :])   # (H, slots, d_k)
            v_list.append(vc[pid, :, :slots, :])
            tokens_left -= slots
            if tokens_left <= 0:
                break

        k_full = torch.cat(k_list, dim=1).unsqueeze(0)   # (1, H, seq_len, d_k)
        v_full = torch.cat(v_list, dim=1).unsqueeze(0)
        return k_full, v_full
