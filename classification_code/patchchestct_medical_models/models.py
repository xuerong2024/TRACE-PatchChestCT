"""Medical model registry for PatchChestCT case-level fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from classification_code.deterministic_ops import (
    replace_adaptive_global_avg_pool3d,
    replace_max_pool3d,
)


Normalization = Literal["unit", "minus_one_one", "tapct", "zscore"]


@dataclass(frozen=True)
class MedicalModelSpec:
    name: str
    family: str
    default_model_id: str | None
    target_shape: tuple[int, int, int]
    clip_hu: tuple[float, float]
    normalization: Normalization
    source: str
    notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


MODEL_SPECS: dict[str, MedicalModelSpec] = {
    "medicalnet_resnet18_23": MedicalModelSpec(
        name="medicalnet_resnet18_23",
        family="medicalnet_resnet18",
        default_model_id="resnet_18_23dataset.pth",
        target_shape=(64, 128, 128),
        clip_hu=(-1000.0, 1000.0),
        normalization="zscore",
        source="Tencent MedicalNet / Med3D",
        notes="3D-ResNet18 MedicalNet checkpoint pretrained on 23 medical datasets.",
    ),
    "medicalnet_resnet50_23": MedicalModelSpec(
        name="medicalnet_resnet50_23",
        family="medicalnet_resnet50",
        default_model_id="resnet_50_23dataset.pth",
        target_shape=(64, 128, 128),
        clip_hu=(-1000.0, 1000.0),
        normalization="zscore",
        source="Tencent MedicalNet / Med3D",
        notes="3D-ResNet50 MedicalNet checkpoint pretrained on 23 medical datasets.",
    ),
    "modelgenesis_chest_ct": MedicalModelSpec(
        name="modelgenesis_chest_ct",
        family="modelgenesis",
        default_model_id="https://huggingface.co/MrGiovanni/ModelsGenesis/resolve/main/Genesis_Chest_CT.pt?download=true",
        target_shape=(64, 128, 128),
        clip_hu=(-1000.0, 1000.0),
        normalization="unit",
        source="Models Genesis Chest CT",
        notes="Self-supervised Models Genesis 3D U-Net encoder pretrained on chest CT.",
    ),
    "tapct_b_3d": MedicalModelSpec(
        name="tapct_b_3d",
        family="hf_tapct",
        default_model_id="fomofo/tap-ct-b-3d",
        target_shape=(12, 224, 224),
        clip_hu=(-1008.0, 822.0),
        normalization="tapct",
        source="Hugging Face / TAP-CT",
        notes="TAP-CT-B-3D ViT-Base CT foundation model pretrained with 3D DINOv2-style SSL.",
    ),
}


def get_model_spec(
    name: str,
    *,
    model_id: str | None = None,
    target_shape: tuple[int, int, int] | None = None,
) -> MedicalModelSpec:
    if name not in MODEL_SPECS:
        valid = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model {name!r}. Valid choices: {valid}")
    spec = MODEL_SPECS[name]
    if model_id is not None:
        spec = replace(spec, default_model_id=model_id)
    if target_shape is not None:
        spec = replace(spec, target_shape=tuple(int(v) for v in target_shape))
    return spec


def conv3x3x3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1) -> nn.Conv3d:
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False,
    )


def downsample_basic_block(x: torch.Tensor, planes: int, stride: int, no_cuda: bool = False) -> torch.Tensor:
    # avg_pool3d(kernel_size=1) is exactly strided subsampling, while its CUDA
    # backward has no deterministic implementation in PyTorch 2.5.
    out = x[:, :, ::stride, ::stride, ::stride]
    zero_pads = torch.zeros(
        out.size(0),
        planes - out.size(1),
        out.size(2),
        out.size(3),
        out.size(4),
        dtype=out.dtype,
        device=out.device,
    )
    return torch.cat([out, zero_pads], dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, dilation: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, dilation: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class MedicalNetResNetClassifier(nn.Module):
    def __init__(self, block: type[BasicBlock] | type[Bottleneck], layers: list[int], num_classes: int, shortcut_type: str = "A") -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=4)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(512 * block.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def _make_layer(
        self,
        block: type[BasicBlock] | type[Bottleneck],
        planes: int,
        blocks: int,
        shortcut_type: str,
        stride: int = 1,
        dilation: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(downsample_basic_block, planes=planes * block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion),
                )
        layers = [block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input: torch.Tensor) -> None:
        if input.dim() != 5:
            raise ValueError(f"expected 5D input, got {input.dim()}D")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)
        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )


class LUConv(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, act: str) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)
        if act == "relu":
            self.activation: nn.Module = nn.ReLU(inplace=True)
        elif act == "prelu":
            self.activation = nn.PReLU(out_chan)
        elif act == "elu":
            self.activation = nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation {act!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn1(self.conv1(x)))


def _make_nconv(in_channel: int, depth: int, act: str) -> nn.Sequential:
    layer1 = LUConv(in_channel, 32 * (2**depth), act)
    layer2 = LUConv(32 * (2**depth), 32 * (2**depth) * 2, act)
    return nn.Sequential(layer1, layer2)


class DownTransition(nn.Module):
    def __init__(self, in_channel: int, depth: int, act: str) -> None:
        super().__init__()
        self.ops = _make_nconv(in_channel, depth, act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.current_depth == 3:
            out = self.ops(x)
            return out, out
        out_before_pool = self.ops(x)
        return self.maxpool(out_before_pool), out_before_pool


class ModelGenesisClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.2, act: str = "relu") -> None:
        super().__init__()
        self.down_tr64 = DownTransition(1, 0, act)
        self.down_tr128 = DownTransition(64, 1, act)
        self.down_tr256 = DownTransition(128, 2, act)
        self.down_tr512 = DownTransition(256, 3, act)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 1024), nn.ReLU(inplace=True), nn.Linear(1024, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out64, _skip64 = self.down_tr64(x)
        out128, _skip128 = self.down_tr128(out64)
        out256, _skip256 = self.down_tr256(out128)
        out512, _skip512 = self.down_tr512(out256)
        return self.head(self.pool(out512).flatten(1))


class TAPCTClassifier(nn.Module):
    def __init__(self, model_id: str, num_classes: int, pretrained: bool, dropout: float) -> None:
        super().__init__()
        try:
            import transformers
        except ModuleNotFoundError as e:
            raise RuntimeError("TAP-CT requires transformers in the pcct environment.") from e

        if pretrained:
            self.backbone = transformers.AutoModel.from_pretrained(model_id, trust_remote_code=True)
        else:
            config = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            self.backbone = transformers.AutoModel.from_config(config, trust_remote_code=True)

        hidden_size = int(getattr(self.backbone.config, "hidden_size", 768))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(x)
        if getattr(outputs, "pooler_output", None) is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state.mean(dim=1)
        return self.head(pooled)


def _state_dict_from_checkpoint(path_or_url: str) -> dict[str, torch.Tensor]:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        checkpoint = torch.hub.load_state_dict_from_url(path_or_url, map_location="cpu", progress=True)
    else:
        checkpoint = torch.load(Path(path_or_url), map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unsupported checkpoint format from {path_or_url}")
    return {str(k).removeprefix("module."): v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}


def _load_partial_state_dict(model: nn.Module, path_or_url: str, *, allowed_prefixes: tuple[str, ...] | None = None) -> dict[str, int]:
    state_dict = _state_dict_from_checkpoint(path_or_url)
    if allowed_prefixes is not None:
        state_dict = {k: v for k, v in state_dict.items() if k.startswith(allowed_prefixes)}

    current = model.state_dict()
    loadable = {k: v for k, v in state_dict.items() if k in current and tuple(v.shape) == tuple(current[k].shape)}
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    return {"loaded": len(loadable), "missing": len(missing), "unexpected": len(unexpected)}


def _medicalnet_weight_path(filename: str) -> Path:
    return Path("nnunet_data/Bronchidata/PatchChestCT/pretrained_medical_models/MedicalNet") / filename


def build_model(
    spec: MedicalModelSpec,
    num_classes: int,
    *,
    pretrained: bool = True,
    pretrained_path: str | None = None,
    freeze_backbone: bool = False,
    dropout: float = 0.2,
    deterministic: bool = False,
) -> nn.Module:
    load_report: dict[str, int] | None = None
    if spec.family == "medicalnet_resnet18":
        model = MedicalNetResNetClassifier(BasicBlock, [2, 2, 2, 2], num_classes, shortcut_type="A")
        if pretrained:
            path = pretrained_path or str(_medicalnet_weight_path(str(spec.default_model_id)))
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"Missing MedicalNet weights: {path}. Download MedicalNet_pytorch_files.zip from the official "
                    "MedicalNet README and place resnet_18_23dataset.pth there, or pass --pretrained-path."
                )
            load_report = _load_partial_state_dict(model, path, allowed_prefixes=("conv1", "bn1", "layer"))
    elif spec.family == "medicalnet_resnet50":
        model = MedicalNetResNetClassifier(Bottleneck, [3, 4, 6, 3], num_classes, shortcut_type="B")
        if pretrained:
            path = pretrained_path or str(_medicalnet_weight_path(str(spec.default_model_id)))
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"Missing MedicalNet weights: {path}. Download MedicalNet_pytorch_files.zip from the official "
                    "MedicalNet README and place resnet_50_23dataset.pth there, or pass --pretrained-path."
                )
            load_report = _load_partial_state_dict(model, path, allowed_prefixes=("conv1", "bn1", "layer"))
    elif spec.family == "modelgenesis":
        model = ModelGenesisClassifier(num_classes, dropout=dropout)
        if pretrained:
            load_report = _load_partial_state_dict(
                model,
                pretrained_path or str(spec.default_model_id),
                allowed_prefixes=("down_tr64", "down_tr128", "down_tr256", "down_tr512"),
            )
    elif spec.family == "hf_tapct":
        model = TAPCTClassifier(str(pretrained_path or spec.default_model_id), num_classes, pretrained=pretrained, dropout=dropout)
    else:
        raise ValueError(f"Unsupported model family {spec.family!r}")

    if load_report is not None:
        setattr(model, "_pretrained_load_report", load_report)
    if deterministic:
        deterministic_replacements = {
            "max_pool3d": replace_max_pool3d(model),
            "adaptive_global_avg_pool3d": replace_adaptive_global_avg_pool3d(model),
            "medicalnet_kernel1_avg_pool3d": spec.family.startswith("medicalnet_"),
        }
        setattr(model, "_deterministic_replacements", deterministic_replacements)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for module in model.modules():
            if isinstance(module, nn.Linear) and module.out_features == num_classes:
                for parameter in module.parameters():
                    parameter.requires_grad = True
    return model


def trainable_parameter_count(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
