"""Explainable inspection priority; scores are not probabilities of wrongdoing."""

from numbers import Integral

import numpy as np
import pandas as pd


CONTRIBUTION_COLUMNS = (
    "priority_volume", "priority_bridge", "priority_seed", "priority_degree"
)
TOP_COLUMNS = ["rank", "gid", "role", "priority_score", "why"]


def _validate_nodes(df: pd.DataFrame, required: set[str]) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Не хватает колонок: {', '.join(sorted(missing))}")
    if df["gid"].isna().any() or df["gid"].duplicated().any():
        raise ValueError("gid должен быть заполнен и уникален")
    if not df["is_seed"].isin([True, False, 0, 1]).all():
        raise ValueError("is_seed должен содержать только bool или 0/1")


def _nonnegative_numbers(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    try:
        values = df[columns].astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Признаки приоритета должны быть числами") from exc
    if not np.isfinite(values.to_numpy()).all() or (values < 0).any().any():
        raise ValueError("Признаки приоритета должны быть конечными и неотрицательными")
    return values


def _positive_percentile(values: pd.Series) -> pd.Series:
    """Positive-only average rank / positive count; zeros stay exactly zero."""
    result = pd.Series(0.0, index=values.index)
    positive = values > 0
    if positive.any():
        result.loc[positive] = values.loc[positive].rank(method="average", pct=True)
    return result


def compute_priority(roles_with_clusters_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with score and four weighted, additive contributions.

    V ranks outgoing amounts for seed and max(in, out) for other nodes;
    D ranks outgoing degree for seed and total degree for others. Each
    adjusted feature is ranked once over all nodes, excluding zero values.
    B ranks unweighted directed betweenness; S = min(seed reach / 3, 1).
    Priority = .35 V + .30 B + .20 S + .15 D. Role confidence is not used.
    """
    columns = ["in_kzt", "out_kzt", "in_deg", "out_deg", "betweenness", "seed_reach_count"]
    _validate_nodes(roles_with_clusters_df, {"gid", "is_seed", *columns})
    # Work on a positional index: joins are by gid, arbitrary duplicate input
    # DataFrame indices must not cause pandas alignment to duplicate rows.
    result = roles_with_clusters_df.copy(deep=True)
    values = _nonnegative_numbers(result, columns).reset_index(drop=True)
    seed = result["is_seed"].astype(bool).reset_index(drop=True)
    volume = values[["in_kzt", "out_kzt"]].max(axis=1).where(~seed, values["out_kzt"])
    degree = (values["in_deg"] + values["out_deg"]).where(~seed, values["out_deg"])
    components = {
        "priority_volume": 0.35 * _positive_percentile(volume),
        "priority_bridge": 0.30 * _positive_percentile(values["betweenness"]),
        "priority_seed": 0.20 * (values["seed_reach_count"] / 3).clip(upper=1),
        "priority_degree": 0.15 * _positive_percentile(degree),
    }
    isolated = (values["in_deg"] + values["out_deg"]) == 0
    for name, contribution in components.items():
        result[name] = contribution.mask(isolated, 0.0).to_numpy()
    result["priority_score"] = result[list(CONTRIBUTION_COLUMNS)].sum(axis=1)
    return result


def _why(row: dict) -> str:
    seed = bool(row["is_seed"])
    volume = row["out_kzt"] if seed else max(row["in_kzt"], row["out_kzt"])
    degree = row["out_deg"] if seed else row["in_deg"] + row["out_deg"]
    factors = [
        (row["priority_volume"], f"{'исходящие' if seed else 'макс. вход/выход'} {volume:,.0f} KZT"),
        (row["priority_bridge"], f"посредничество {row['betweenness']:.6g}"),
        (row["priority_seed"], f"достижим из {int(row['seed_reach_count'])} seed за 1–4 шага"),
        (row["priority_degree"], f"{'исходящих связей' if seed else 'входящих + исходящих связей'} {int(degree)}"),
    ]
    factors.sort(key=lambda item: -item[0])
    text = "Приоритет проверки: " + "; ".join(
        f"{description} (вклад {contribution:.4f})" for contribution, description in factors
    ) + "."
    if row["in_deg"] + row["out_deg"] == 0:
        text += " Связей 0: данных для структурного приоритета недостаточно."
    if seed:
        text += " Входы seed неполны; объём и степень оценены по исходящим."
    if bool(row.get("truncated_by_depth", False)) or (
        row.get("depth") == 4 and row["out_deg"] == 0
    ):
        text += " Обрыв на 4-м колене: дальнейшие переводы неизвестны."
    text += " Потоки вне выборки не видны; достижимость не доказывает маршрут денег."
    return text


def make_top_nodes(scored_df: pd.DataFrame, limit: int = 30) -> pd.DataFrame:
    """Return fixed CSV schema, ordered by score descending then gid ascending.

    Inspect the unfiltered first 30 nodes (or all if smaller). When seeds
    exceed one third, select only non-seed nodes without changing scores.
    This decision is independent of ``limit``. If fewer eligible nodes
    exist, return all available; exports validate the required dataset size.
    Selection diagnostics are available in ``result.attrs`` for run metadata.
    Input tables, their order and scores are never mutated.
    """
    if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 1:
        raise ValueError("limit должен быть положительным целым числом")
    required = {
        "gid", "is_seed", "role", "priority_score", "in_kzt", "out_kzt",
        "in_deg", "out_deg", "betweenness", "seed_reach_count", *CONTRIBUTION_COLUMNS,
    }
    _validate_nodes(scored_df, required)
    _nonnegative_numbers(scored_df, sorted(required - {"gid", "is_seed", "role"}))
    if (scored_df["priority_score"] > 1).any():
        raise ValueError("priority_score должен находиться в диапазоне 0..1")
    ordered = scored_df.sort_values(["priority_score", "gid"], ascending=[False, True])
    initial = ordered.head(30)
    n_seed = int(initial["is_seed"].astype(bool).sum())
    nonseed_only = n_seed * 3 > len(initial)
    eligible = ordered.loc[~ordered["is_seed"].astype(bool)] if nonseed_only else ordered
    chosen = eligible.head(int(limit))
    result = chosen[["gid", "role", "priority_score"]].copy().reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1, dtype=np.int64))
    result["why"] = [_why(row) for row in chosen.to_dict("records")]
    result = result[TOP_COLUMNS]
    result.attrs.update(
        selection_policy="non_seed" if nonseed_only else "all_nodes",
        initial_top_size=len(initial),
        initial_top_seed_count=n_seed,
        initial_top_seed_fraction=n_seed / len(initial) if len(initial) else 0.0,
    )
    return result
