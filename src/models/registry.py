from typing import Any, Dict

try:
    from .cnn_bilstm import CNNBiLSTM
except Exception:
    CNNBiLSTM = None  # type: ignore

try:
    from .tcn import TCN
except Exception:
    TCN = None  # type: ignore


def build_model(name: str, **kwargs: Dict[str, Any]):
    name = name.lower()
    if name == "cnn_bilstm":
        if CNNBiLSTM is None:
            raise RuntimeError("PyTorch not available for CNNBiLSTM")
        return CNNBiLSTM(num_classes=kwargs.get("num_classes", 3),
                         conv_blocks=kwargs.get("conv_blocks", 3),
                         hidden=kwargs.get("lstm_hidden", 128),
                         dropout=kwargs.get("dropout", 0.3))
    if name == "tcn":
        if TCN is None:
            raise RuntimeError("PyTorch not available for TCN")
        return TCN(num_classes=kwargs.get("num_classes", 3),
                   layers=kwargs.get("layers", 4),
                   ch=kwargs.get("ch", 64),
                   dropout=kwargs.get("dropout", 0.2))
    raise ValueError(f"Unknown model: {name}")
