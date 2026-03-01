import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, NamedTuple

"""
WrongTokenMarginPenalty: Khi mô hình dự đoán sai
    - Ép mô hình tăng logit của token đúng (nếu detach_max = False thì trực tiếp giảm thêm logit của token sai nhưng không nên dùng)

WrongTokenEntropyPenalty: Khi mô hình đã dự đoán sai, mà lại còn tự tin vào cái token sai đấy
    - Dấu hiệu: entropy thấp
    - Công thức: penalty = max(0, min_entropy - H) * WrongMask
    - Không quan tâm đến token đúng, chỉ cần gradient kéo toàn bộ phân phối xác suất phẳng hơn là được.

FocalOverconfidencePenalty: Khi mô hình dự đoán sai
    - Khuếch đại cái sai theo hệ số gamma, phạt cho xác suất của token sai giảm xuống
    - Xác suất mô hình dành cho token sai càng lơn, phạt càng mạnh

Các trường hợp:
    - tất cả z đều thấp, và model dự đoán đúng: chưa xử lý
    - tất cả z đều thấp, và model dự đoán sai: dùng WrongTokenMarginPenalty để kéo logit của token đúng lên
    - tất cả z đều cao, và model dự đoán sai: dùng WrongTokenMarginPenalty để kéo logit của token đúng lên và FocalOverconfidencePenalty để giảm xác suất của token sai
    - tất cả z đều cao, và model dự đoán đúng: chưa xử lý
    - entropy thấp, model dự đoán đúng: chưa xử lý
    - entropy thấp, model dự đoán sai: dùng WrongTokenMarginPenalty để kéo logit của token đúng lên và WrongTokenEntropyPenalty để kéo phẳng phân phối và FocalOverconfidencePenalty để giảm xác suất của token sai
"""

class PenaltyContext(NamedTuple):
    logits: torch.Tensor
    inputs: torch.Tensor
    targets: torch.Tensor
    loss_mask: Optional[torch.Tensor]
    max_logits: torch.Tensor
    predicted: torch.Tensor
    target_logits: torch.Tensor
    wrong_mask: torch.Tensor
    log_probs: Optional[torch.Tensor]

def _safe_masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is not None:
        tensor = tensor * mask
        denom = mask.sum().clamp(min=1e-6)
    else:
        denom = torch.tensor(tensor.numel(), dtype=tensor.dtype, device=tensor.device).clamp(min=1e-6)
    return tensor.sum() / denom

class PenaltyRule(nn.Module, ABC):
    needs_log_probs: bool = False

    @abstractmethod
    def forward(self, ctx: PenaltyContext) -> torch.Tensor:
        """Trả về scalar penalty."""

class WrongTokenMarginPenalty(PenaltyRule):
    needs_log_probs = False
    def __init__(self, weight: float = 1.0, detach_max: bool = True):
        super().__init__()
        self.weight = weight
        self.detach_max = detach_max

    def forward(self, ctx: PenaltyContext) -> torch.Tensor:
        max_l = ctx.max_logits.detach() if self.detach_max else ctx.max_logits
        margin = F.relu(max_l - ctx.target_logits)
        penalty = margin * ctx.wrong_mask

        return self.weight * _safe_masked_mean(penalty, ctx.loss_mask)

class WrongTokenEntropyPenalty(PenaltyRule):
    needs_log_probs = True
    def __init__(self, weight: float = 0.01, min_entropy: float = 0.5):
        super().__init__()
        self.weight = weight
        self.min_entropy = min_entropy

    def forward(self, ctx: PenaltyContext) -> torch.Tensor:
        assert ctx.log_probs is not None, \
            "WrongTokenEntropyPenalty cần log_probs; thêm needs_log_probs=True"

        probs = ctx.log_probs.exp()
        entropy = -(probs * ctx.log_probs).sum(dim=-1)

        low_entropy = F.relu(self.min_entropy - entropy)

        combined_mask = ctx.wrong_mask * (ctx.loss_mask if ctx.loss_mask is not None else 1.0)
        penalty = low_entropy * combined_mask

        return self.weight * _safe_masked_mean(penalty, ctx.loss_mask)

class FocalOverconfidencePenalty(PenaltyRule):
    """
    Lấy cảm hứng từ Focal Loss nhưng đảo chiều:
    phạt nặng khi model quá chắc vào token SAI.

        penalty = (p_wrong)^gamma * log(p_wrong + ε)  khi sai

    Với gamma > 1: gradient TĂNG khi overconfident
    (ngược lại với p(1-p) trong softmax thông thường).

    Cần log_probs để lấy p_max một cách hiệu quả.
    """
    needs_log_probs = True

    def __init__(self, weight: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, ctx: PenaltyContext) -> torch.Tensor:
        assert ctx.log_probs is not None

        # xác suất của token được predict (token sai)
        log_p_wrong = ctx.log_probs.gather(
            dim=-1,
            index=ctx.predicted.unsqueeze(-1)
        ).squeeze(-1)
        p_wrong = log_p_wrong.exp()

        # focal penalty: nặng khi p cao, nhẹ khi p thấp
        penalty = (p_wrong ** self.gamma) * (-log_p_wrong)
        penalty = penalty * ctx.wrong_mask

        return self.weight * _safe_masked_mean(penalty, ctx.loss_mask)

class PenaltyEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.rules = nn.ModuleList()

    @classmethod
    def build_default(cls) -> "PenaltyEngine":
        return (
            cls()
            .add_rule(WrongTokenMarginPenalty(weight=0.1, detach_max=True))
            .add_rule(WrongTokenEntropyPenalty(weight=0.001, min_entropy=0.5))
            .add_rule(FocalOverconfidencePenalty(weight=0.05, gamma=2.0))
        )

    def add_rule(self, rule: PenaltyRule) -> "PenaltyEngine":
        self.rules.append(rule)
        return self

    def _build_context(
        self,
        logits: torch.Tensor,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: Optional[torch.Tensor],
    ) -> PenaltyContext:

        max_logits, predicted = logits.max(dim=-1)
        target_logits = logits.gather(
            dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)

        wrong_mask = (predicted != targets).float()

        needs_log = any(getattr(r, "needs_log_probs", False) for r in self.rules)
        log_probs = F.log_softmax(logits, dim=-1) if needs_log else None

        return PenaltyContext(
            logits=logits,
            inputs=inputs,
            targets=targets,
            loss_mask=loss_mask,
            max_logits=max_logits,
            predicted=predicted,
            target_logits=target_logits,
            wrong_mask=wrong_mask,
            log_probs=log_probs,
        )

    def forward(
        self,
        logits: torch.Tensor,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.rules:
            return torch.tensor(0.0, device=logits.device)

        ctx = self._build_context(logits, inputs, targets, loss_mask)

        total = torch.tensor(0.0, device=logits.device)
        for rule in self.rules:
            total = total + rule(ctx)
        return total
