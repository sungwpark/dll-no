import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple


class FieldNormalizer:
    """Global per-channel normalization for field tensors."""

    def __init__(
        self, tensor: torch.Tensor, mode: str = "gaussian", epsilon: float = 1e-6
    ):
        self.mode = mode.lower()
        self.epsilon = epsilon

        self.reduce_dims = (0,) + tuple(range(2, tensor.ndim))

        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.range = None

        self._compute_stats(tensor)

    def _compute_stats(self, tensor: torch.Tensor):
        if self.mode == "gaussian":
            self.mean = torch.mean(tensor, dim=self.reduce_dims, keepdim=True)
            self.std = torch.std(tensor, dim=self.reduce_dims, keepdim=True)

            assert self.mean.numel() == tensor.size(
                1
            ), f"Error: Stats shape {self.mean.shape} implies pointwise normalization. Expected global channel stats."

        elif self.mode == "minmax":
            self.min = torch.amin(tensor, dim=self.reduce_dims, keepdim=True)
            self.max = torch.amax(tensor, dim=self.reduce_dims, keepdim=True)
            self.range = self.max - self.min

            assert self.min.numel() == tensor.size(
                1
            ), f"Error: Stats shape {self.min.shape} implies pointwise normalization."
        else:
            raise ValueError(f"Unknown normalization mode: {self.mode}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "gaussian":
            return (x - self.mean) / (self.std + self.epsilon)
        elif self.mode == "minmax":
            return (x - self.min) / (self.range + self.epsilon)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "gaussian":
            return x * (self.std + self.epsilon) + self.mean
        elif self.mode == "minmax":
            return x * (self.range + self.epsilon) + self.min

    def to(self, device):
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.std is not None:
            self.std = self.std.to(device)
        if self.min is not None:
            self.min = self.min.to(device)
        if self.max is not None:
            self.max = self.max.to(device)
        if self.range is not None:
            self.range = self.range.to(device)
        return self

    def cuda(self):
        return self.to("cuda")

    def cpu(self):
        return self.to("cpu")


def create_dataloaders(
    train_set: TensorDataset,
    val_set: TensorDataset,
    test_set: TensorDataset,
    batch_size: int = 256,
    num_workers: int = 8,
    normalization_mode: str = "gaussian",
    share_xy_normalizer: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, FieldNormalizer, FieldNormalizer]:

    print(
        f"[DataLoaders] Processing dataset for Operator Learning ({normalization_mode})..."
    )

    def extract_and_fix(dataset: TensorDataset) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = dataset.tensors
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B, L) -> (B, 1, L)
        if y.ndim == 2:
            y = y.unsqueeze(1)
        return x, y

    train_x, train_y = extract_and_fix(train_set)
    val_x, val_y = extract_and_fix(val_set)
    test_x, test_y = extract_and_fix(test_set)

    print(f"    Input Shape: {tuple(train_x.shape)}")

    x_normalizer = FieldNormalizer(train_x, mode=normalization_mode)
    if share_xy_normalizer:
        y_normalizer = x_normalizer
    else:
        y_normalizer = FieldNormalizer(train_y, mode=normalization_mode)

    if normalization_mode == "gaussian":
        print(f"    X Norm Mean: {tuple(x_normalizer.mean.shape)}")
    else:
        print(f"    X Norm Min:  {tuple(x_normalizer.min.shape)}")

    train_ds = TensorDataset(x_normalizer.encode(train_x), y_normalizer.encode(train_y))
    val_ds = TensorDataset(x_normalizer.encode(val_x), y_normalizer.encode(val_y))
    test_ds = TensorDataset(x_normalizer.encode(test_x), y_normalizer.encode(test_y))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, x_normalizer, y_normalizer


def create_last_layer_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int = 256,
    num_workers: int = 8,
    normalization_mode: str = "gaussian",
) -> Tuple[
    DataLoader, DataLoader, DataLoader, FieldNormalizer, FieldNormalizer, object
]:
    """Create dataloaders for cached last-layer tuples without re-normalization."""

    print(
        f"[DataLoaders] Processing LastLayer Dataset (Norm: {normalization_mode}; no re-normalization)..."
    )

    def extract_tensors(dataset):
        xs = torch.stack([item[0] for item in dataset.data])
        ys = torch.stack([item[1] for item in dataset.data])
        ws = torch.stack([item[2] for item in dataset.data])

        if xs.ndim == 2:
            xs = xs.unsqueeze(1)
        if ys.ndim == 2:
            ys = ys.unsqueeze(1)
        if ws.ndim == 2:
            ws = ws.unsqueeze(2)

        return xs, ys, ws

    train_x, train_y, train_w = extract_tensors(train_dataset)
    val_x, val_y, val_w = extract_tensors(val_dataset)
    test_x, test_y, test_w = extract_tensors(test_dataset)

    print(
        f"    Train Shapes -> X: {tuple(train_x.shape)}, Y: {tuple(train_y.shape)}, W: {tuple(train_w.shape)}"
    )

    def create_ds(x, y, w):
        return TensorDataset(x, y, w)

    train_ds = create_ds(train_x, train_y, train_w)
    val_ds = create_ds(val_x, val_y, val_w)
    test_ds = create_ds(test_x, test_y, test_w)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, None, None, None
