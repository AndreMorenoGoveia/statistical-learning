# PCS5024 - Aprendizado Estatístico - 2026
# EP - Séries Temporais
# Autor: Marcel Rodrigues de Barros (marcel.barros@usp.br)

# Objetivo:
# Implementar e comparar modelos para previsão de séries temporais com dados faltantes,
# combinando codificação temporal no estilo de Vaswani et al. (2017) com o modelo IQN
# descrito em Gouttes et al. (2021), para avaliar o impacto dessas abordagens no desempenho.

# O que os alunos devem implementar:
# 1. Codificação temporal inspirada em Vaswani et al. (2017), com as features temporais
#    codificadas concatenadas à feature de SSH, resultando em T+1 features de entrada.
# 2. IQN (Implicit Quantile Networks) conforme Gouttes et al. (2021), integrado ao pipeline
#    de previsão para produzir estimativas probabilísticas da série temporal.
# 3. Comparação entre o modelo base e o modelo com codificação temporal,
#    tanto com dados completos quanto com diferentes níveis de dados faltantes.
# 4. Teste de cobertura nos resultados da IQN para avaliar a qualidade dos quantis estimados.

# Entregáveis:
# 1. Código-fonte atualizado com a implementação solicitada.
# 2. Relatório em PDF descrevendo a implementação, os desafios enfrentados e os resultados obtidos (plots).

# Priorize boas visualizações!
# Dúvidas devem ser enviadas via fórum no e-Disciplinas.

import argparse
import datetime
import json
import math
import pathlib
import random
import time
from dataclasses import dataclass, field
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# --- Configuration ---
DEFAULT_PAST_LEN = 10 * 800
DEFAULT_FUTURE_LEN = 10 * 200
DEFAULT_SLIDING_WINDOW_STEP = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_EPOCHS = 1000
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_DATA_FILENAME = "data/santos_ssh.csv"
DEFAULT_TRAIN_TEST_SPLIT_DATE = "2020-06-01 00:00:00"
DEFAULT_PAST_PLOT_VIEW_SIZE = 200
DATA_REMOVAL_RATIO = 0.3

DEFAULT_TIME_DIM = 16
DEFAULT_IQN_EMBEDDING_DIM = 64
DEFAULT_NUM_EVAL_SAMPLES = 100

CRPS_TAUS = np.round(np.arange(0.05, 0.96, 0.05), 4)
COVERAGE_LEVELS = [0.5, 0.8, 0.9, 0.95]
RESULTS_DIR = pathlib.Path("results")
FIGURES_DIR = pathlib.Path("figures")

SEED = 100


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(SEED)


@dataclass(slots=True)
class PreparedData:
    feature_names: list[str]
    norm_statistics: tuple[torch.Tensor, torch.Tensor]
    train_dataloader: DataLoader
    test_dataloader: DataLoader


@dataclass(slots=True)
class EvaluationResult:
    avg_loss: float
    contexts: list[np.ndarray]
    context_timestamps: list[np.ndarray]
    predictions: list[np.ndarray]
    targets: list[np.ndarray]
    target_timestamps: list[np.ndarray]
    quantile_taus: np.ndarray = field(default_factory=lambda: np.empty(0))
    quantile_preds: list[np.ndarray] = field(default_factory=list)


def load_data(file_path: pathlib.Path) -> pl.DataFrame:
    """Loads data from CSV, and sets datetime and feature types."""

    df = pl.read_csv(file_path)
    df = df.with_columns(
        [
            pl.col("datetime").str.to_datetime(
                time_unit="ms",
                strict=True,
                exact=True,
                format="%Y-%m-%d %H:%M:%S+00:00",
            ),
        ]
        + [pl.col(f).cast(pl.Float32) for f in df.columns if f != "datetime"]
    )

    return df


def split_data(
    df: pl.DataFrame, split_date: datetime.datetime
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Splits the data into training and testing sets based on the split date."""

    train_df = df.filter(pl.col("datetime") < split_date)
    test_df = df.filter(pl.col("datetime") >= split_date)

    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

    return train_df, test_df


def create_sequences(
    df: pl.DataFrame,
    past_len: int,
    future_len: int,
    step: int = 1,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Creates windows using a sliding window approach.

    Args:
        data (pl.DataFrame): DataFrame containing the data.
        past_len (int): Length of the past sequence in minutes.
        future_len (int): Length of the future sequence in minutes.
        step (int): Step size for the sliding window in minutes.

    Returns:
        tuple: Arrays of past and future sequences of features and timestamps
    """

    xs, xs_timestamps, xs_lengths, ys, ys_timestamps, ys_lengths = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    datetime_values = df.get_column("datetime").to_numpy().astype(np.float32)
    start_time = float(np.min(datetime_values)) + float(past_len)
    stop_time = float(np.max(datetime_values)) - float(future_len)
    observer_times = np.arange(start_time, stop_time, step)
    for ot in observer_times:
        lb = df["datetime"].search_sorted(ot - past_len, side="left")
        obs = df["datetime"].search_sorted(ot, side="left")
        ub = df["datetime"].search_sorted(ot + future_len, side="left")
        x = df[lb:obs].select(pl.exclude("datetime")).to_numpy()
        x_timestamps = df[lb:obs].select(pl.col("datetime")).to_numpy()
        x_length = x.shape[0]
        y = df[obs:ub].select(pl.exclude("datetime")).to_numpy()
        y_timestamps = df[obs:ub].select(pl.col("datetime")).to_numpy()
        y_length = y.shape[0]

        if x_length == 0 or y_length == 0:
            continue
        xs.append(torch.tensor(x))
        xs_timestamps.append(torch.tensor(x_timestamps))
        xs_lengths.append(torch.tensor(x_length))
        ys.append(torch.tensor(y))
        ys_timestamps.append(torch.tensor(y_timestamps))
        ys_lengths.append(torch.tensor(y_length))
    return (
        torch.nn.utils.rnn.pad_sequence(xs, batch_first=True),
        torch.nn.utils.rnn.pad_sequence(xs_timestamps, batch_first=True),
        torch.stack(xs_lengths),
    ), (
        torch.nn.utils.rnn.pad_sequence(ys, batch_first=True),
        torch.nn.utils.rnn.pad_sequence(ys_timestamps, batch_first=True),
        torch.stack(ys_lengths),
    )


def prepare_dataloaders(
    train_df_features: pl.DataFrame,
    test_df_features: pl.DataFrame,
    past_len: int,
    future_len: int,
    batch_size: int,
    sliding_window_step: int,
) -> tuple[DataLoader, DataLoader]:
    """Creates sequences and prepares PyTorch DataLoaders."""

    print("Creating sequences and dataloaders...")
    (x_train, x_train_timestamps, x_train_lengths), (
        y_train,
        y_train_timestamps,
        y_train_lengths,
    ) = create_sequences(
        df=train_df_features,
        past_len=past_len,
        future_len=future_len,
        step=sliding_window_step,
    )
    (x_test, x_test_timestamps, x_test_lengths), (
        y_test,
        y_test_timestamps,
        y_test_lengths,
    ) = create_sequences(
        df=test_df_features,
        past_len=past_len,
        future_len=future_len,
        step=sliding_window_step,
    )

    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    train_dataset = TensorDataset(
        x_train,
        x_train_timestamps,
        x_train_lengths,
        y_train,
        y_train_timestamps,
        y_train_lengths,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(
        x_test,
        x_test_timestamps,
        x_test_lengths,
        y_test,
        y_test_timestamps,
        y_test_lengths,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_dataloader,
        test_dataloader,
    )


def data_preparation(
    args: argparse.Namespace, missing_ratio: float | None = None
) -> PreparedData:
    """Loads, splits, scales, and converts the data into dataloaders."""

    if missing_ratio is None:
        missing_ratio = DATA_REMOVAL_RATIO

    file_path = pathlib.Path(args.data_filename)
    split_date = datetime.datetime.fromisoformat(args.split_date)
    df = load_data(file_path=file_path)
    feature_names = list(df.drop("datetime").columns)

    origin = df.get_column("datetime").min()

    train_df, test_df = split_data(df=df, split_date=split_date)

    train_mean = train_df.select(feature_names).mean()
    train_std = train_df.select(feature_names).std()
    norm_statistics = (
        torch.tensor(train_mean.row(0), dtype=torch.float32),
        torch.tensor(train_std.row(0), dtype=torch.float32),
    )

    print(f"Scaling data using Train Mean: {train_mean}, Train Std: {train_std}")

    def scale_and_clock(frame: pl.DataFrame) -> pl.DataFrame:
        frame = frame.with_columns(
            [
                (pl.col(f) - train_mean.select([f]).item())
                / train_std.select([f]).item()
                for f in feature_names
            ]
        )
        frame = frame.with_columns(
            [
                (pl.col("datetime") - origin)
                .dt.total_minutes()
                .cast(pl.Float32)
                .alias("datetime")
            ]
        )
        return frame

    train_data_scaled = scale_and_clock(train_df)
    test_data_scaled = scale_and_clock(test_df)

    if missing_ratio > 0.0:
        sample_train = sorted(
            random.sample(
                range(train_data_scaled.height),
                int((1 - missing_ratio) * train_data_scaled.height),
            )
        )
        train_data_scaled = train_data_scaled[sample_train]

        sample_test = sorted(
            random.sample(
                range(test_data_scaled.height),
                int((1 - missing_ratio) * test_data_scaled.height),
            )
        )
        test_data_scaled = test_data_scaled[sample_test]

    print(
        f"After removing {missing_ratio:.0%} of points: "
        f"train={train_data_scaled.height}, test={test_data_scaled.height}"
    )

    train_dataloader, test_dataloader = prepare_dataloaders(
        train_df_features=train_data_scaled,
        test_df_features=test_data_scaled,
        past_len=int(args.past_len),
        future_len=int(args.future_len),
        batch_size=int(args.batch_size),
        sliding_window_step=int(args.sliding_window_step),
    )

    return PreparedData(
        feature_names=feature_names,
        norm_statistics=norm_statistics,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
    )


# --- Model Definition ---
class TemporalEncoding(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"TemporalEncoding dim must be even, got {dim}")
        self.dim = dim
        exponents = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        div_term = torch.pow(torch.tensor(max_period), -exponents)
        self.register_buffer("div_term", div_term)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t * self.div_term
        enc = torch.zeros(*t.shape[:-1], self.dim, device=t.device, dtype=t.dtype)
        enc[..., 0::2] = torch.sin(angles)
        enc[..., 1::2] = torch.cos(angles)
        return enc


class IQNHead(nn.Module):
    def __init__(self, hidden_size: int, embedding_dim: int = DEFAULT_IQN_EMBEDDING_DIM):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.phi = nn.Linear(embedding_dim, hidden_size)
        self.q = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        i_pi = math.pi * torch.arange(0, embedding_dim, dtype=torch.float32)
        self.register_buffer("i_pi", i_pi)

    def forward(self, psi: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        cos_basis = torch.cos(tau.unsqueeze(-1) * self.i_pi)
        phi = F.relu(self.phi(cos_basis))
        out = psi * (1.0 + phi)
        return self.q(out)


class ForecastModel(nn.Module):
    def __init__(
        self,
        ssh_dim: int,
        hidden_size: int,
        use_temporal_encoding: bool = False,
        time_dim: int = DEFAULT_TIME_DIM,
        head: str = "iqn",
        iqn_embedding_dim: int = DEFAULT_IQN_EMBEDDING_DIM,
    ):
        super().__init__()
        self.ssh_dim = ssh_dim
        self.hidden_size = hidden_size
        self.use_temporal_encoding = use_temporal_encoding
        self.head = head

        if use_temporal_encoding:
            self.temporal = TemporalEncoding(time_dim)
            encoder_input = ssh_dim + time_dim
            decoder_input = time_dim
        else:
            self.temporal = None
            encoder_input = ssh_dim
            decoder_input = 1

        self.encoder = nn.GRU(encoder_input, hidden_size, batch_first=True)
        self.decoder = nn.GRU(decoder_input, hidden_size, batch_first=True)

        if head == "mse":
            self.linear = nn.Linear(hidden_size, ssh_dim)
            self.iqn = None
        elif head == "iqn":
            self.linear = None
            self.iqn = IQNHead(hidden_size, embedding_dim=iqn_embedding_dim)
        else:
            raise ValueError(f"Unknown head: {head}")

    @staticmethod
    def forecast_origin(y_timestamps: torch.Tensor) -> torch.Tensor:
        return y_timestamps[:, :1, :]

    def encode(
        self,
        x: torch.Tensor,
        x_timestamps: torch.Tensor,
        x_lengths: torch.Tensor,
        origin: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_temporal_encoding:
            x = torch.cat([self.temporal(x_timestamps - origin), x], dim=-1)
        x_packed = nn.utils.rnn.pack_padded_sequence(
            x, x_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.encoder(x_packed)
        return h_n

    def decode_states(
        self,
        h_n: torch.Tensor,
        y_timestamps: torch.Tensor,
        y_lengths: torch.Tensor,
        origin: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = h_n.size(1)
        max_target_length = int(y_lengths.max().item())

        if self.use_temporal_encoding:
            y_rel = (y_timestamps - origin)[:, :max_target_length]
            decoder_input = self.temporal(y_rel)
        else:
            decoder_input = torch.zeros(
                batch_size, max_target_length, 1, device=h_n.device
            )

        packed = nn.utils.rnn.pack_padded_sequence(
            decoder_input, y_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.decoder(packed, h_n)
        out = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0]
        return out

    def predict(self, psi: torch.Tensor, tau: torch.Tensor | None = None) -> torch.Tensor:
        if self.head == "mse":
            return self.linear(psi)
        if tau is None:
            tau = torch.rand(psi.size(0), psi.size(1), device=psi.device)
        return self.iqn(psi, tau)

    def forward(
        self,
        x: torch.Tensor,
        x_timestamps: torch.Tensor,
        x_lengths: torch.Tensor,
        y_timestamps: torch.Tensor,
        y_lengths: torch.Tensor,
        tau: torch.Tensor | None = None,
    ) -> torch.Tensor:
        origin = self.forecast_origin(y_timestamps)
        h_n = self.encode(x, x_timestamps, x_lengths, origin)
        psi = self.decode_states(h_n, y_timestamps, y_lengths, origin)
        return self.predict(psi, tau)


def length_mask(lengths: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(max_len, device=device).unsqueeze(0) < lengths.to(
        device
    ).unsqueeze(1)


def masked_mse(
    predictions: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    mask = length_mask(lengths, predictions.size(1), predictions.device)
    return F.mse_loss(predictions[mask], targets[mask])


def quantile_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    tau: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    error = (targets.squeeze(-1) - predictions.squeeze(-1))
    pinball = torch.maximum(tau * error, (tau - 1.0) * error)
    mask = length_mask(lengths, predictions.size(1), predictions.device)
    return pinball[mask].mean()


# --- Training and Evaluation ---
def _unpack_batch(batch, device):
    (xf, xt, xl, yf, yt, yl) = batch
    max_tl = int(yl.max().item())
    inputs = xf.to(device)
    in_ts = xt.to(device).float()
    targets = yf.to(device)[:, :max_tl]
    tgt_ts = yt.to(device).float()
    return inputs, in_ts, xl, targets, tgt_ts, yl


def run_train_epoch(
    model: ForecastModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    show_progress: bool = True,
) -> float:
    model.train()
    iterator = tqdm(dataloader, desc="Training") if show_progress else dataloader
    losses = []
    for batch in iterator:
        inputs, in_ts, in_len, targets, tgt_ts, tgt_len = _unpack_batch(batch, device)
        optimizer.zero_grad()
        if model.head == "iqn":
            tau = torch.rand(targets.size(0), targets.size(1), device=device)
            outputs = model(inputs, in_ts, in_len, tgt_ts, tgt_len, tau=tau)
            loss = quantile_loss(outputs, targets, tau, tgt_len)
        else:
            outputs = model(inputs, in_ts, in_len, tgt_ts, tgt_len)
            loss = masked_mse(outputs, targets, tgt_len)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().item())
    return float(np.mean(losses))


@torch.no_grad()
def run_eval_loss(
    model: ForecastModel, dataloader: DataLoader, device: torch.device
) -> float:
    model.eval()
    losses = []
    for batch in dataloader:
        inputs, in_ts, in_len, targets, tgt_ts, tgt_len = _unpack_batch(batch, device)
        if model.head == "iqn":
            tau = torch.rand(targets.size(0), targets.size(1), device=device)
            outputs = model(inputs, in_ts, in_len, tgt_ts, tgt_len, tau=tau)
            loss = quantile_loss(outputs, targets, tau, tgt_len)
        else:
            outputs = model(inputs, in_ts, in_len, tgt_ts, tgt_len)
            loss = masked_mse(outputs, targets, tgt_len)
        losses.append(loss.cpu().item())
    return float(np.mean(losses))


def denormalize_batch(
    batch: np.ndarray, norm_statistics: tuple[torch.Tensor, torch.Tensor]
) -> np.ndarray:
    """Converts a normalized batch back to the original data scale."""

    norm_mean, norm_std = norm_statistics
    return batch * norm_std.cpu().numpy() + norm_mean.cpu().numpy()


def build_eval_taus() -> np.ndarray:
    taus = set(np.round(CRPS_TAUS, 4).tolist())
    taus.add(0.5)
    for c in COVERAGE_LEVELS:
        taus.add(round((1 - c) / 2, 4))
        taus.add(round((1 + c) / 2, 4))
    return np.array(sorted(taus), dtype=np.float32)


@torch.no_grad()
def evaluate_model(
    model: ForecastModel,
    dataloader: DataLoader,
    device: torch.device,
    norm_statistics: tuple[torch.Tensor, torch.Tensor],
    num_samples: int = DEFAULT_NUM_EVAL_SAMPLES,
) -> EvaluationResult:
    model.eval()
    eval_taus = build_eval_taus() if model.head == "iqn" else np.array([], dtype=np.float32)
    median_idx = int(np.where(np.isclose(eval_taus, 0.5))[0][0]) if model.head == "iqn" else 0

    all_contexts, all_context_ts = [], []
    all_targets, all_target_ts = [], []
    all_predictions, all_quantiles = [], []
    total_loss, num_batches = 0.0, 0

    for batch in dataloader:
        inputs, in_ts, in_len, targets, tgt_ts, tgt_len = _unpack_batch(batch, device)
        origin = model.forecast_origin(tgt_ts)
        h_n = model.encode(inputs, in_ts, in_len, origin)
        psi = model.decode_states(h_n, tgt_ts, tgt_len, origin)
        max_tl = psi.size(1)

        if model.head == "iqn":
            tau_grid = torch.from_numpy(eval_taus).to(device)
            preds_per_tau = []
            for tval in tau_grid:
                tau = tval.expand(psi.size(0), max_tl)
                preds_per_tau.append(model.predict(psi, tau).squeeze(-1))
            quant = torch.stack(preds_per_tau, dim=-1)
            point = quant[..., median_idx]
            err = targets.squeeze(-1).unsqueeze(-1) - quant
            pin = torch.maximum(tau_grid * err, (tau_grid - 1.0) * err)
            mask = length_mask(tgt_len, max_tl, device)
            total_loss += pin[mask].mean().item()
        else:
            point = model.predict(psi).squeeze(-1)
            quant = None
            total_loss += masked_mse(point.unsqueeze(-1), targets, tgt_len).item()
        num_batches += 1

        bs = inputs.size(0)
        for i in range(bs):
            il = int(in_len[i].item())
            tl = int(tgt_len[i].item())
            all_contexts.append(
                denormalize_batch(inputs[i, :il].cpu().numpy(), norm_statistics)
            )
            all_context_ts.append(in_ts[i, :il].cpu().numpy())
            all_targets.append(
                denormalize_batch(targets[i, :tl].cpu().numpy(), norm_statistics)
            )
            all_target_ts.append(tgt_ts[i, :tl].cpu().numpy())
            all_predictions.append(
                denormalize_batch(point[i, :tl].cpu().numpy().reshape(-1, 1), norm_statistics)
            )
            if quant is not None:
                q = quant[i, :tl].cpu().numpy()
                all_quantiles.append(denormalize_batch(q, norm_statistics))

    return EvaluationResult(
        avg_loss=total_loss / max(num_batches, 1),
        contexts=all_contexts,
        context_timestamps=all_context_ts,
        predictions=all_predictions,
        targets=all_targets,
        target_timestamps=all_target_ts,
        quantile_taus=eval_taus,
        quantile_preds=all_quantiles,
    )


def compute_metrics(result: EvaluationResult, has_quantiles: bool) -> dict:
    y = np.concatenate([t[:, 0] for t in result.targets])
    yhat = np.concatenate([p[:, 0] for p in result.predictions])
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    mae = float(np.mean(np.abs(y - yhat)))
    metrics = {"rmse": rmse, "mae": mae}

    if has_quantiles and len(result.quantile_preds) > 0:
        taus = result.quantile_taus
        q = np.concatenate(result.quantile_preds, axis=0)
        yy = y[:, None]
        err = yy - q
        pinball = np.maximum(taus[None, :] * err, (taus[None, :] - 1.0) * err)

        denom = np.sum(np.abs(y))
        def ql(level: float) -> float:
            j = int(np.argmin(np.abs(taus - level)))
            return float(2.0 * np.sum(pinball[:, j]) / denom)

        metrics["ql50"] = ql(0.5)
        metrics["ql90"] = ql(0.9)

        crps_cols = [int(np.argmin(np.abs(taus - t))) for t in CRPS_TAUS]
        metrics["crps"] = float(2.0 * np.mean(pinball[:, crps_cols]))

        cov = {}
        for c in COVERAGE_LEVELS:
            lo = int(np.argmin(np.abs(taus - (1 - c) / 2)))
            hi = int(np.argmin(np.abs(taus - (1 + c) / 2)))
            inside = (y >= q[:, lo]) & (y <= q[:, hi])
            cov[f"{int(c * 100)}"] = float(np.mean(inside))
        metrics["coverage"] = cov
    return metrics


def coverage_curve(result: EvaluationResult) -> tuple[np.ndarray, np.ndarray]:
    taus = result.quantile_taus
    y = np.concatenate([t[:, 0] for t in result.targets])
    q = np.concatenate(result.quantile_preds, axis=0)
    nominal, empirical = [], []
    for t in taus:
        if t >= 0.5:
            continue
        lo = int(np.argmin(np.abs(taus - t)))
        hi = int(np.argmin(np.abs(taus - (1 - t))))
        inside = (y >= q[:, lo]) & (y <= q[:, hi])
        nominal.append(1 - 2 * t)
        empirical.append(float(np.mean(inside)))
    order = np.argsort(nominal)
    return np.array(nominal)[order], np.array(empirical)[order]


def _q_col(taus: np.ndarray, level: float) -> int:
    return int(np.argmin(np.abs(taus - level)))


def plot_loss_curve(history: dict, path: pathlib.Path, title: str) -> None:
    epochs = range(1, len(history["train"]) + 1)
    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, history["train"], label="Train loss")
    plt.plot(epochs, history["test"], label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (pinball / MSE)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_forecast_example(
    result: EvaluationResult,
    path: pathlib.Path,
    view_size: int,
    title: str,
    window_id: int | None = None,
) -> None:
    if window_id is None:
        window_id = int(np.argmax([len(t) for t in result.targets]))

    ctx = result.contexts[window_id][-view_size:, 0]
    ctx_t = result.context_timestamps[window_id][-view_size:, 0]
    tgt = result.targets[window_id][:, 0]
    tgt_t = result.target_timestamps[window_id][:, 0]
    med = result.predictions[window_id][:, 0]

    origin = tgt_t[0]
    ctx_h = (ctx_t - origin) / 60.0
    tgt_h = (tgt_t - origin) / 60.0

    plt.figure(figsize=(11, 5))
    plt.plot(ctx_h, ctx, color="0.4", lw=1.2, label="Context (past SSH)")
    plt.plot(tgt_h, tgt, color="black", lw=1.8, label="Target")

    has_q = len(result.quantile_preds) > 0
    if has_q:
        taus = result.quantile_taus
        q = result.quantile_preds[window_id]
        q05, q95 = q[:, _q_col(taus, 0.05)], q[:, _q_col(taus, 0.95)]
        q25, q75 = q[:, _q_col(taus, 0.25)], q[:, _q_col(taus, 0.75)]
        plt.fill_between(tgt_h, q05, q95, color="tab:blue", alpha=0.18, label="90% interval")
        plt.fill_between(tgt_h, q25, q75, color="tab:blue", alpha=0.30, label="50% interval")
        plt.plot(tgt_h, med, color="tab:blue", lw=1.6, label="Median forecast")
    else:
        plt.plot(tgt_h, med, color="tab:blue", lw=1.6, label="Point forecast")

    plt.axvline(0.0, color="red", ls="--", lw=1, alpha=0.6)
    plt.xlabel("Hours from forecast start")
    plt.ylabel("SSH (m)")
    plt.title(title)
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_coverage_diagram(
    curves: dict[str, tuple[np.ndarray, np.ndarray]], path: pathlib.Path, title: str
) -> None:
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")
    for label, (nominal, empirical) in curves.items():
        plt.plot(nominal, empirical, marker="o", ms=4, label=label)
    plt.xlabel("Nominal coverage")
    plt.ylabel("Empirical coverage")
    plt.title(title)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_metric_vs_missing(
    grid: dict, metric: str, ylabel: str, path: pathlib.Path, title: str
) -> None:
    plt.figure(figsize=(7.5, 4.5))
    for label, runs in grid.items():
        ratios = sorted(runs.keys())
        values = [runs[r]["metrics"].get(metric) for r in ratios]
        if any(v is None for v in values):
            continue
        plt.plot([r * 100 for r in ratios], values, marker="o", label=label)
    plt.xlabel("Missing data (%)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_coverage90_vs_missing(grid: dict, path: pathlib.Path) -> None:
    plt.figure(figsize=(7.5, 4.5))
    for label, runs in grid.items():
        ratios = sorted(runs.keys())
        vals = []
        for r in ratios:
            cov = runs[r]["metrics"].get("coverage")
            vals.append(cov["90"] if cov else None)
        if any(v is None for v in vals):
            continue
        plt.plot([r * 100 for r in ratios], vals, marker="o", label=label)
    plt.axhline(0.90, color="red", ls="--", lw=1, label="Nominal 90%")
    plt.xlabel("Missing data (%)")
    plt.ylabel("Empirical coverage of 90% interval")
    plt.title("Calibration of the 90% interval vs missing data")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_data_overview(args: argparse.Namespace, path: pathlib.Path) -> None:
    df = load_data(pathlib.Path(args.data_filename))
    t = df.get_column("datetime").to_numpy()
    ssh = df.get_column("ssh").to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    axes[0].plot(t, ssh, lw=0.4, color="tab:blue")
    axes[0].axvline(
        np.datetime64(datetime.datetime.fromisoformat(args.split_date)),
        color="red", ls="--", lw=1, label="Train/test split",
    )
    axes[0].set_title("Santos SSH — full series (10-min cadence)")
    axes[0].set_ylabel("SSH (m)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    zoom = df.head(3 * 144)
    axes[1].plot(zoom.get_column("datetime").to_numpy(), zoom.get_column("ssh").to_numpy(),
                 lw=1.0, color="tab:blue")
    axes[1].set_title("3-day zoom — dominant semi-diurnal tide")
    axes[1].set_ylabel("SSH (m)")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def training_loop(
    model: ForecastModel,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    show_progress: bool = False,
) -> dict:
    train_losses, test_losses = [], []
    for epoch in range(1, num_epochs + 1):
        train_loss = run_train_epoch(
            model, train_dataloader, optimizer, device, show_progress=show_progress
        )
        test_loss = run_eval_loss(model, test_dataloader, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"  epoch {epoch:3d}/{num_epochs} | train {train_loss:.4f} | test {test_loss:.4f}")
    return {"train": train_losses, "test": test_losses}


def run_single(
    label: str,
    prepared: PreparedData,
    device: torch.device,
    *,
    head: str,
    use_temporal_encoding: bool,
    hidden_size: int,
    time_dim: int,
    learning_rate: float,
    num_epochs: int,
    num_eval_samples: int,
) -> dict:
    set_seed(SEED)
    model = ForecastModel(
        ssh_dim=len(prepared.feature_names),
        hidden_size=hidden_size,
        use_temporal_encoding=use_temporal_encoding,
        time_dim=time_dim,
        head=head,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\n=== Training [{label}] (head={head}, temporal_encoding={use_temporal_encoding}) ===")
    t0 = time.time()
    history = training_loop(
        model, prepared.train_dataloader, prepared.test_dataloader, optimizer,
        device, num_epochs,
    )
    train_time = time.time() - t0

    eval_result = evaluate_model(
        model, prepared.test_dataloader, device, prepared.norm_statistics,
        num_samples=num_eval_samples,
    )
    metrics = compute_metrics(eval_result, has_quantiles=(head == "iqn"))
    metrics["train_time_s"] = round(train_time, 1)
    print(f"  metrics: {metrics}")
    return {"label": label, "metrics": metrics, "history": history, "result": eval_result}


def make_run_args(args: argparse.Namespace) -> argparse.Namespace:
    run_args = argparse.Namespace(**vars(args))
    run_args.past_len = args.exp_past_len
    run_args.future_len = args.exp_future_len
    run_args.sliding_window_step = args.exp_step
    run_args.batch_size = args.batch_size
    return run_args


def run_experiments(args: argparse.Namespace, device: torch.device) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    run_args = make_run_args(args)
    ratios = args.exp_missing

    plot_data_overview(args, FIGURES_DIR / "data_overview.png")

    configs = [
        ("IQN (no temporal enc.)", dict(head="iqn", use_temporal_encoding=False)),
        ("IQN + temporal enc.", dict(head="iqn", use_temporal_encoding=True)),
    ]

    grid: dict[str, dict] = {label: {} for label, _ in configs}
    grid["MSE point baseline"] = {}
    summary_rows = []

    for ratio in ratios:
        print(f"\n########## Missing ratio = {ratio:.0%} ##########")
        set_seed(SEED)
        prepared = data_preparation(run_args, missing_ratio=ratio)

        runs_here = list(configs)
        if ratio == 0.0:
            runs_here = runs_here + [
                ("MSE point baseline", dict(head="mse", use_temporal_encoding=False))
            ]

        for label, cfg in runs_here:
            run = run_single(
                f"{label} @ {ratio:.0%}", prepared, device,
                hidden_size=args.hidden_size,
                time_dim=args.time_dim,
                learning_rate=args.exp_lr,
                num_epochs=args.exp_epochs,
                num_eval_samples=args.num_eval_samples,
                **cfg,
            )
            grid[label][ratio] = run

            m = run["metrics"]
            summary_rows.append({
                "config": label, "missing": ratio,
                "rmse": m["rmse"], "mae": m["mae"],
                "crps": m.get("crps"), "ql50": m.get("ql50"), "ql90": m.get("ql90"),
                "cov50": (m.get("coverage") or {}).get("50"),
                "cov90": (m.get("coverage") or {}).get("90"),
                "train_time_s": m["train_time_s"],
            })

            tag = label.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
            plot_loss_curve(
                run["history"], FIGURES_DIR / f"loss_{tag}_{int(ratio*100)}.png",
                f"Loss — {label} @ {ratio:.0%} missing",
            )
            plot_forecast_example(
                run["result"], FIGURES_DIR / f"forecast_{tag}_{int(ratio*100)}.png",
                view_size=args.past_view_size,
                title=f"Forecast — {label} @ {ratio:.0%} missing",
            )

    plot_metric_vs_missing(
        {k: v for k, v in grid.items() if k != "MSE point baseline"},
        "rmse", "RMSE (m)", FIGURES_DIR / "rmse_vs_missing.png",
        "Point accuracy vs missing data",
    )
    plot_metric_vs_missing(
        {k: v for k, v in grid.items() if k != "MSE point baseline"},
        "crps", "CRPS (m)", FIGURES_DIR / "crps_vs_missing.png",
        "Probabilistic accuracy (CRPS) vs missing data",
    )
    plot_coverage90_vs_missing(
        {k: v for k, v in grid.items() if k != "MSE point baseline"},
        FIGURES_DIR / "coverage90_vs_missing.png",
    )

    hardest = max(ratios)
    cov_curves = {}
    for label, _ in configs:
        if hardest in grid[label]:
            cov_curves[label] = coverage_curve(grid[label][hardest]["result"])
    if cov_curves:
        plot_coverage_diagram(
            cov_curves, FIGURES_DIR / "coverage_diagram.png",
            f"Coverage reliability @ {hardest:.0%} missing",
        )

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(summary_rows, f, indent=2)
    write_metrics_markdown(summary_rows, RESULTS_DIR / "metrics.md")
    print(f"\nDone. Figures in {FIGURES_DIR}/, metrics in {RESULTS_DIR}/")


def write_metrics_markdown(rows: list[dict], path: pathlib.Path) -> None:
    header = (
        "| Config | Missing | RMSE | MAE | CRPS | QL50 | QL90 | Cov50 | Cov90 |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    def fmt(v):
        return "—" if v is None else f"{v:.4f}"
    lines = []
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['missing']:.0%} | {r['rmse']:.4f} | {r['mae']:.4f} | "
            f"{fmt(r['crps'])} | {fmt(r['ql50'])} | {fmt(r['ql90'])} | "
            f"{fmt(r['cov50'])} | {fmt(r['cov90'])} |"
        )
    path.write_text(header + "\n".join(lines) + "\n")


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.mode == "experiment":
        run_experiments(args, device)
        return

    prepared = data_preparation(args, missing_ratio=args.missing_ratio)
    run = run_single(
        "single", prepared, device,
        head=args.head,
        use_temporal_encoding=args.use_temporal_encoding,
        hidden_size=args.hidden_size,
        time_dim=args.time_dim,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        num_eval_samples=args.num_eval_samples,
    )
    FIGURES_DIR.mkdir(exist_ok=True)
    plot_loss_curve(run["history"], FIGURES_DIR / "loss_curve.png", "Training and Test Loss")
    plot_forecast_example(
        run["result"], FIGURES_DIR / "forecast_example.png",
        view_size=args.past_view_size, title="Forecast example",
    )
    print("Script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series EP: IQN-RNN + temporal encoding")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "experiment"])

    parser.add_argument("--past_len", type=int, default=DEFAULT_PAST_LEN)
    parser.add_argument("--future_len", type=int, default=DEFAULT_FUTURE_LEN)
    parser.add_argument("--sliding_window_step", type=int, default=DEFAULT_SLIDING_WINDOW_STEP)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--hidden_size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--data_filename", type=str, default=DEFAULT_DATA_FILENAME)
    parser.add_argument("--split_date", type=str, default=DEFAULT_TRAIN_TEST_SPLIT_DATE)
    parser.add_argument("--past_view_size", type=int, default=DEFAULT_PAST_PLOT_VIEW_SIZE)

    parser.add_argument("--head", type=str, default="iqn", choices=["iqn", "mse"])
    parser.add_argument("--use_temporal_encoding", action="store_true")
    parser.add_argument("--time_dim", type=int, default=DEFAULT_TIME_DIM)
    parser.add_argument("--missing_ratio", type=float, default=DATA_REMOVAL_RATIO)
    parser.add_argument("--num_eval_samples", type=int, default=DEFAULT_NUM_EVAL_SAMPLES)

    parser.add_argument("--exp_past_len", type=int, default=2880)
    parser.add_argument("--exp_future_len", type=int, default=720)
    parser.add_argument("--exp_step", type=int, default=120)
    parser.add_argument("--exp_epochs", type=int, default=40)
    parser.add_argument("--exp_lr", type=float, default=1e-3)
    parser.add_argument("--exp_missing", type=float, nargs="+", default=[0.0, 0.3, 0.6])

    args = parser.parse_args()
    main(args)
