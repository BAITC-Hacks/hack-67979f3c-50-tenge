"""Правила и эвристические оценки из Plan.md §5; это гипотезы, не обвинения."""

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype


ROLES = ("consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral")
RULE_ORDER = ("isolated", "boundary_consolidator", "boundary_unknown", "coordinator",
              "consolidator", "distributor", "terminal", "transit", "fallback")


def compute_role_thresholds(features_df):
    """Linear quantiles; None disables the rule when the sample is empty.

    Public helper for run_metadata/UI; assign_roles also saves these values
    in result.attrs['role_thresholds']. No gid participates in thresholds.
    """
    inflows = features_df.loc[
        ~features_df["is_seed"] & features_df["in_kzt"].gt(0), "in_kzt"]
    bridges = features_df.loc[features_df["betweenness"].gt(0), "betweenness"]
    return {
        "in_kzt_q75": float(inflows.quantile(0.75, interpolation="linear")) if len(inflows) else None,
        "betweenness_q95": float(bridges.quantile(0.95, interpolation="linear")) if len(bridges) else None,
    }


def _excess(values, threshold):
    if threshold is None or threshold <= 0:
        return pd.Series(0.0, index=values.index)
    return ((values / threshold - 1) / 2).clip(0, 1)


def _number(value):
    # Compact representation, never cut off the caveat to fit 200 characters.
    if 1 <= abs(value) < 1e12:
        return f"{value:,.2f}".rstrip("0").rstrip(".").replace(",", " ")
    return f"{value:.6g}"


def _evidence(row, thresholds):
    rule = row.role_rule
    incoming, outgoing = int(row.in_deg), int(row.out_deg)
    amount = _number(row.in_kzt)
    if rule == "isolated":
        return "Входящих 0, исходящих 0; наблюдаемых связей нет, данных для роли недостаточно."
    if rule == "boundary_consolidator":
        return (f"Признаки консолидации: вход от {incoming} (≥3), {amount} KZT "
                f"(Q75={_number(thresholds['in_kzt_q75'])}); depth=4, обрыв обхода; дальнейший выход неизвестен.")
    if rule == "boundary_unknown":
        return (f"Граница обхода: depth=4, вход от {incoming}, {amount} KZT; "
                "наблюдаемых исходящих 0; дальнейшие переводы неизвестны, роль не определена.")
    if rule == "coordinator":
        return (f"Признаки связующего узла: достижим от {int(row.seed_reach_count)} seed (≥2), "
                f"betweenness={_number(row.betweenness)} ≥Q95={_number(thresholds['betweenness_q95'])}; "
                f"вход/выход={incoming}/{outgoing}; не доказательство руководства.")
    if rule == "consolidator":
        return (f"Признаки консолидации: вход от {incoming} (≥3), выход к {outgoing}; "
                f"вход ≥2×max(выход,1); {amount} KZT ≥Q75={_number(thresholds['in_kzt_q75'])}.")
    if rule == "distributor":
        if row.is_seed:
            return (f"Признаки распределения: выход к {outgoing} (≥5), "
                    f"{_number(row.out_kzt)} KZT; seed: неполный вход не используется.")
        return (f"Признаки распределения: выход к {outgoing} (≥5), вход от {incoming}; "
                f"выход ≥2×max(вход,1); отправлено {_number(row.out_kzt)} KZT.")
    if rule == "terminal":
        return (f"Наблюдаемый конечный получатель: вход от {incoming}, {amount} KZT; "
                f"исходящих 0, depth={int(row.depth)}<4; удержание денег не доказано.")
    if rule == "transit":
        return (f"Признаки транзита: выход/вход={_number(row.pass_through)} в [0.7;1.3]; "
                f"вход {amount}, выход {_number(row.out_kzt)} KZT; происхождение денег не установлено.")
    if row.is_seed:
        return (f"Seed: исходящих связей {outgoing} (<5); входы неполны и не используются "
                "для роли; структурных признаков недостаточно.")
    return (f"Недостаточно признаков выбранных ролей: вход от {incoming}, выход к {outgoing}; "
            f"вход {amount}, выход {_number(row.out_kzt)} KZT.")


def assign_roles(features_df):
    """Consume features joined to B's seed metrics on unique gid, return a copy.

    All predicates and scores follow Plan.md §5. Missing seed metrics are an
    integration error, never silently replaced with zero. Quantile thresholds
    and candidate counts (before precedence) are available through df.attrs.
    """
    required = {"gid", "depth", "is_seed", "in_deg", "out_deg", "in_kzt", "out_kzt",
                "betweenness", "pass_through", "truncated_by_depth",
                "seed_reach_count", "seed_distance"}
    missing = sorted(required - set(features_df.columns))
    if missing:
        raise ValueError(f"Назначение ролей: отсутствуют колонки {missing}; нужны признаки A и seed-метрики B")
    if features_df["gid"].isna().any() or features_df["gid"].duplicated().any():
        raise ValueError("Назначение ролей: нужны уникальные непустые gid")
    for column in ("is_seed", "truncated_by_depth"):
        if not is_bool_dtype(features_df[column].dtype) or features_df[column].isna().any():
            raise ValueError(f"Назначение ролей: {column} должен быть булевым без пропусков")
    for column in ("depth", "in_deg", "out_deg", "in_kzt", "out_kzt", "betweenness", "seed_reach_count"):
        values = features_df[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Назначение ролей: {column} содержит пропуски/некорректные значения")
    frame = features_df.copy().reset_index(drop=True)
    boundary = frame["depth"].eq(4) & frame["out_deg"].eq(0)
    if not boundary.eq(frame["truncated_by_depth"]).all():
        raise ValueError("Назначение ролей: truncated_by_depth не согласован с глубиной и выходом")
    thresholds = compute_role_thresholds(frame)
    q75, q95 = thresholds["in_kzt_q75"], thresholds["betweenness_q95"]
    non_seed = ~frame["is_seed"]
    both = frame["in_deg"].gt(0) & frame["out_deg"].gt(0)
    collection = non_seed & frame["in_deg"].ge(3) & frame["in_kzt"].ge(q75 if q75 is not None else np.inf)
    candidates = {
        "isolated": frame["in_deg"].eq(0) & frame["out_deg"].eq(0),
        "boundary_consolidator": boundary & collection,
        "boundary_unknown": boundary & ~collection,
        "coordinator": non_seed & both & frame["seed_reach_count"].ge(2)
                       & frame["betweenness"].ge(q95 if q95 is not None else np.inf),
        "consolidator": ~boundary & collection & frame["in_deg"].ge(2 * frame["out_deg"].clip(lower=1)),
        "distributor": frame["out_deg"].ge(5) & (frame["is_seed"] | frame["out_deg"].ge(2 * frame["in_deg"].clip(lower=1))),
        "terminal": non_seed & frame["in_deg"].gt(0) & frame["out_deg"].eq(0) & frame["depth"].lt(4),
        "transit": non_seed & both & frame["pass_through"].between(0.7, 1.3),
        "fallback": pd.Series(True, index=frame.index),
    }
    collector_score = 0.55 + 0.15 * _excess(frame["in_deg"], 3) + 0.15 * _excess(frame["in_kzt"], q75)
    scores = {
        "isolated": 0.10,
        "boundary_consolidator": collector_score - 0.15,
        "boundary_unknown": 0.15,
        "coordinator": 0.45 + 0.15 * _excess(frame["betweenness"], q95)
                       + 0.15 * _excess(frame["seed_reach_count"], 2),
        "consolidator": collector_score,
        "distributor": 0.55 + 0.30 * _excess(frame["out_deg"], 5),
        "terminal": 0.55,
        "transit": 0.45 + 0.20 * (1 - (frame["pass_through"] - 1).abs() / 0.3).clip(0, 1),
        "fallback": 0.20,
    }
    frame["role"] = ""
    frame["role_rule"] = ""
    frame["role_score"] = 0.0
    for rule in RULE_ORDER:
        mask = candidates[rule] & frame["role_rule"].eq("")
        role = {"isolated": "peripheral", "boundary_unknown": "peripheral",
                "boundary_consolidator": "consolidator", "fallback": "peripheral"}.get(rule, rule)
        frame.loc[mask, "role"] = role
        frame.loc[mask, "role_rule"] = rule
        score = scores[rule]
        frame.loc[mask, "role_score"] = score.loc[mask] if isinstance(score, pd.Series) else score
    frame["evidence"] = pd.Series(
        [_evidence(row, thresholds) for row in frame.itertuples(index=False)],
        index=frame.index, dtype="str")
    if not frame["evidence"].str.len().between(1, 200).all():
        raise ValueError("Объяснение роли должно содержать от 1 до 200 символов")
    if not np.isfinite(frame["role_score"]).all() or not frame["role_score"].between(0, 1).all():
        raise ValueError("Оценка роли должна быть конечной и в диапазоне 0..1")
    frame.attrs["role_thresholds"] = thresholds
    frame.attrs["role_candidate_counts"] = {rule: int(mask.sum()) for rule, mask in candidates.items() if rule != "fallback"}
    return frame
