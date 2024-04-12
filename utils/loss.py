import torch
import torch.nn as nn


class OhemCELoss(nn.Module):
    def __init__(self, thresh: float, min_kept: int) -> None:
        super().__init__()
        self.thresh = torch.log(torch.tensor(1 / thresh, dtype=torch.float))
        self.min_kept = min_kept
        self.criteria = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.min_kept] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.min_kept]
        return torch.mean(loss)


class OhemLossWrapper:
    def __init__(self, thresh: float, min_kept: int) -> None:
        self.loss = OhemCELoss(thresh=thresh, min_kept=min_kept)

    def __call__(self, output, labels):
        out, out16, out32 = output

        loss1 = self.loss(out, labels)
        loss2 = self.loss(out16, labels)
        loss3 = self.loss(out32, labels)

        loss = loss1 + loss2 + loss3
        return loss
