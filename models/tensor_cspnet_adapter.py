from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn


def get_tensor_cspnet_reference_root() -> str:
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "archive",
            "reference_code",
            "Tensor-CSPNet-and-Graph-CSPNet",
        )
    )


def ensure_tensor_cspnet_reference_on_path() -> str:
    ref_root = get_tensor_cspnet_reference_root()
    if ref_root not in sys.path:
        sys.path.insert(0, ref_root)
    return ref_root


ensure_tensor_cspnet_reference_on_path()

from utils.model import Tensor_CSPNet_Basic  # noqa: E402


@dataclass
class TensorCSPNetForwardOutputs:
    x_csp: torch.Tensor
    x_log: torch.Tensor
    latent: torch.Tensor
    base_logit: torch.Tensor
    final_logit: torch.Tensor


class TensorCSPNetAdapter(nn.Module):
    def __init__(
        self,
        *,
        channel_num: int,
        mlp: bool = False,
        dataset: str = "BCIC",
        base_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.channel_num = int(channel_num)
        self.dataset = str(dataset)
        self.base_model = base_model or Tensor_CSPNet_Basic(
            channel_num=int(channel_num),
            mlp=bool(mlp),
            dataset=str(dataset),
        )
        # Keep the SPD pipeline in float64, but move the downstream temporal/readout
        # path to float32 so RTX-class GPUs do not spend unnecessary time on FP64 conv/MLP.
        self.base_model.Temporal_Block = self.base_model.Temporal_Block.float()
        self.base_model.Classifier = self.base_model.Classifier.float()
        self.feature_dim = int(getattr(self.base_model, "tcn_channels"))
        self.num_classes = int(self._infer_num_classes())

    def _infer_num_classes(self) -> int:
        classifier = self.base_model.Classifier
        if isinstance(classifier, nn.Linear):
            return int(classifier.out_features)
        if isinstance(classifier, nn.Sequential):
            last = classifier[-1]
            if isinstance(last, nn.Linear):
                return int(last.out_features)
        raise TypeError("unable to infer class count from Tensor-CSPNet classifier")

    def forward_features(self, x: torch.Tensor) -> TensorCSPNetForwardOutputs:
        x = x.double()
        batch_size = int(x.shape[0])
        window_num = int(x.shape[1])
        band_num = int(x.shape[2])

        x_flat = x.reshape(batch_size, window_num * band_num, x.shape[3], x.shape[4])
        x_csp = self.base_model.BiMap_Block(x_flat)
        x_log = self.base_model.LogEig(x_csp)
        x_vec = x_log.float().reshape(batch_size, 1, window_num, -1)
        latent = self.base_model.Temporal_Block(x_vec).reshape(batch_size, -1)
        base_logit = self.base_model.Classifier(latent)
        return TensorCSPNetForwardOutputs(
            x_csp=x_csp,
            x_log=x_log,
            latent=latent,
            base_logit=base_logit,
            final_logit=base_logit,
        )

    def forward(self, x: torch.Tensor, *, return_features: bool = False):
        outputs = self.forward_features(x)
        if return_features:
            return outputs
        return outputs.final_logit
