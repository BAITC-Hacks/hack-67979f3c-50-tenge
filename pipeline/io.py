"""Чтение исходных parquet и проверки согласованности без изменения данных."""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype


def load_data(data_dir):
    """Return (nodes, edges, transactions); validation is an explicit next step."""
    data_dir = Path(data_dir)
    frames = []
    for name in ("nodes", "edges", "transactions"):
        path = data_dir / f"{name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Не найден файл данных: {path}")
        frames.append(pd.read_parquet(path))
    nodes, edges, transactions = frames
    if "date" not in transactions:
        raise ValueError("transactions: отсутствует колонка date")
    try:
        transactions["date"] = pd.to_datetime(transactions["date"], format="ISO8601", errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("transactions.date: некорректная дата") from exc
    return nodes, edges, transactions


def _require_columns(frame, name, columns):
    if not frame.columns.is_unique:
        raise ValueError(f"{name}: повторяющиеся имена колонок")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: отсутствуют колонки {missing}")
    if frame[list(columns)].isna().any().any():
        raise ValueError(f"{name}: пропуски в обязательных колонках")


def _integer_column(frame, name, column):
    values = frame[column]
    if not is_integer_dtype(values.dtype) or is_bool_dtype(values.dtype):
        raise ValueError(f"{name}.{column}: требуется целочисленный тип")
    bounds = np.iinfo(np.int64)
    if ((values < bounds.min) | (values > bounds.max)).any():
        raise ValueError(f"{name}.{column}: значение вне диапазона int64")


def _positive_amounts(frame, name):
    values = frame["sum_kzt"]
    if (not is_numeric_dtype(values.dtype) or is_bool_dtype(values.dtype)
            or not np.isfinite(values.to_numpy(dtype=float)).all()
            or (values <= 0).any()):
        raise ValueError(f"{name}.sum_kzt: нужны конечные положительные суммы")


def validate_inputs(nodes, edges, transactions) -> None:
    """Fail on invalid schemas, endpoints or transaction aggregates.

    Floating sums are compared with rtol=1e-12, atol=1e-6 KZT, sufficient
    for floating aggregation noise at this dataset's scale.
    Identical transaction rows are allowed: they need not be duplicate events.
    """
    _require_columns(nodes, "nodes", ("gid", "depth", "is_seed"))
    _require_columns(edges, "edges", ("src", "dst", "sum_kzt", "n_tx", "depth"))
    _require_columns(transactions, "transactions", ("src", "dst", "date", "sum_kzt"))
    for name, frame, columns in (
        ("nodes", nodes, ("gid", "depth")),
        ("edges", edges, ("src", "dst", "n_tx", "depth")),
        ("transactions", transactions, ("src", "dst")),
    ):
        for column in columns:
            _integer_column(frame, name, column)
    if not is_bool_dtype(nodes["is_seed"].dtype):
        raise ValueError("nodes.is_seed: требуется булев тип")
    if nodes["gid"].duplicated().any():
        raise ValueError("nodes: повторяющиеся gid")
    if edges.duplicated(["src", "dst"]).any():
        raise ValueError("edges: повторяющиеся пары src, dst")
    for name, frame in (("nodes", nodes), ("edges", edges)):
        if not frame["depth"].between(0, 4).all():
            raise ValueError(f"{name}.depth: допустимый диапазон 0..4")
    if (edges["n_tx"] <= 0).any():
        raise ValueError("edges.n_tx: нужны положительные целые количества")
    gids = set(nodes["gid"])
    for name, frame in (("edges", edges), ("transactions", transactions)):
        _positive_amounts(frame, name)
        if not (set(frame["src"]) | set(frame["dst"])) <= gids:
            raise ValueError(f"{name}: src/dst отсутствуют в nodes.gid")
    try:
        dates = pd.to_datetime(transactions["date"], format="ISO8601", errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("transactions.date: некорректная дата") from exc
    if dates.isna().any():
        raise ValueError("transactions.date: пропуски дат")

    aggregate = transactions.groupby(["src", "dst"], as_index=False).agg(
        tx_sum=("sum_kzt", "sum"), tx_count=("sum_kzt", "size")
    )
    joined = edges.merge(aggregate, on=["src", "dst"], how="outer", indicator=True,
                         validate="one_to_one")
    if not joined["_merge"].eq("both").all():
        raise ValueError("edges и transactions: не совпадают пары src, dst")
    if not joined["n_tx"].eq(joined["tx_count"]).all():
        raise ValueError("edges и transactions: не совпадают количества n_tx")
    if not np.isclose(joined["sum_kzt"].to_numpy(dtype=float),
                      joined["tx_sum"].to_numpy(dtype=float),
                      rtol=1e-12, atol=1e-6).all():
        raise ValueError("edges и transactions: не совпадают суммы sum_kzt")
