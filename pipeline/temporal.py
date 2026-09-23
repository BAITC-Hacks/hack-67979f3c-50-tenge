"""Observed daily activity and explainable patterns; never traces the same money."""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
import math
from statistics import median

import pandas as pd


COUNT_COLUMNS = [
    "temporal_active_days", "temporal_lag_1_2_day_pairs", "temporal_same_day_flow_days",
    "temporal_synchronous_days", "temporal_burst_days", "temporal_equal_amount_groups",
    "anomaly_depth_cohort_size",
]


def _validate(nodes, transactions, scored):
    requirements = (
        (nodes, {"gid", "depth", "is_seed"}, "nodes"),
        (transactions, {"src", "dst", "date", "sum_kzt"}, "transactions"),
        (scored, {"gid", "in_kzt", "out_kzt", "in_deg", "out_deg"}, "scored"),
    )
    for frame, required, name in requirements:
        if not frame.columns.is_unique or not required.issubset(frame.columns):
            raise ValueError(f"{name}: отсутствуют обязательные колонки или есть повторные имена")
        if frame[list(required)].isna().any().any():
            raise ValueError(f"{name}: пропуски в обязательных колонках")
    for frame, columns in ((nodes, ("gid", "depth")), (transactions, ("src", "dst")),
                           (scored, ("gid", "in_deg", "out_deg"))):
        for column in columns:
            values = frame[column]
            if (not pd.api.types.is_integer_dtype(values.dtype)
                    or pd.api.types.is_bool_dtype(values.dtype)):
                raise ValueError(f"{column}: требуется целочисленный тип, float запрещён")
            if len(values) and (values.min() < -(2**63) or values.max() > 2**63 - 1):
                raise ValueError(f"{column}: вне диапазона int64")
    if nodes.gid.duplicated().any() or scored.gid.duplicated().any():
        raise ValueError("gid должны быть уникальны")
    gids = set(nodes.gid)
    if gids != set(scored.gid) or not (set(transactions.src) | set(transactions.dst)) <= gids:
        raise ValueError("Таблицы должны ссылаться только на полный набор gid из nodes")
    if not nodes.depth.between(0, 4).all() or not nodes.is_seed.isin([True, False]).all():
        raise ValueError("Нужны depth в 0..4 и булев is_seed")
    for frame, columns in ((transactions, ("sum_kzt",)), (scored, ("in_kzt", "out_kzt", "in_deg", "out_deg"))):
        for column in columns:
            if (not pd.api.types.is_numeric_dtype(frame[column].dtype)
                    or pd.api.types.is_bool_dtype(frame[column].dtype)
                    or not frame[column].map(lambda value: math.isfinite(value) and value >= 0).all()):
                raise ValueError(f"{column}: нужны конечные неотрицательные числа")
    if not transactions.sum_kzt.gt(0).all():
        raise ValueError("Суммы транзакций должны быть положительными")


def _empty_day():
    return {"in_tx": 0, "out_tx": 0, "in_kzt": 0.0, "out_kzt": 0.0,
            "activity_tx": 0, "activity_kzt": 0.0, "senders": set(), "recipients": set()}


def analyze_temporal(nodes, transactions, scored):
    """Return full gid metrics and a JSON-compatible report with string IDs.

    Counts relate to days, day pairs or equal-amount groups, not attribution
    of individual incoming funds to later outgoing funds. Cohort percentiles
    use average ranks within depth and require at least 20 clients to flag.
    Activity counts a self-transfer once; its incoming/outgoing views each
    include it. No input is mutated and no model or external API is used.
    """
    _validate(nodes, transactions, scored)
    tx = transactions[["src", "dst", "date", "sum_kzt"]].copy()
    try:
        dates = pd.to_datetime(tx["date"], format="ISO8601", errors="raise")
        if dates.isna().any():
            raise ValueError("Пропущена дата")
        tx["day"] = dates.dt.date
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("transactions.date: нужны корректные даты в единой временной зоне") from exc
    gids = sorted(int(gid) for gid in nodes.gid)
    daily = {gid: defaultdict(_empty_day) for gid in gids}
    repeats = defaultdict(int)
    for row in tx.sort_values(["day", "src", "dst", "sum_kzt"]).itertuples(index=False):
        src, dst, amount = int(row.src), int(row.dst), float(row.sum_kzt)
        sending, receiving = daily[src][row.day], daily[dst][row.day]
        sending["out_tx"] += 1
        sending["out_kzt"] += amount
        sending["recipients"].add(dst)
        receiving["in_tx"] += 1
        receiving["in_kzt"] += amount
        receiving["senders"].add(src)
        for gid in {src, dst}:
            daily[gid][row.day]["activity_tx"] += 1
            daily[gid][row.day]["activity_kzt"] += amount
        repeats[(src, dst, row.day, amount)] += 1
    repeated_by_node = {gid: [] for gid in gids}
    n_repeat_groups = 0
    for (src, dst, day, amount), count in sorted(repeats.items()):
        if count < 3:
            continue
        n_repeat_groups += 1
        group = {"src": str(src), "dst": str(dst), "date": day.isoformat(),
                 "amount_kzt": amount, "n_tx": count, "sum_kzt": amount * count}
        for gid in {src, dst}:
            repeated_by_node[gid].append(group.copy())

    cohort = nodes[["gid", "depth", "is_seed"]].merge(
        scored[["gid", "in_kzt", "out_kzt", "in_deg", "out_deg"]],
        on="gid", validate="one_to_one",
    ).sort_values("gid").reset_index(drop=True)
    seed = cohort.is_seed.astype(bool)
    cohort["volume"] = cohort[["in_kzt", "out_kzt"]].max(axis=1).where(~seed, cohort.out_kzt)
    cohort["degree"] = (cohort.in_deg + cohort.out_deg).where(~seed, cohort.out_deg)
    cohort["cohort_size"] = cohort.groupby("depth").gid.transform("size")
    for field in ("volume", "degree"):
        cohort[field + "_percentile"] = cohort.groupby("depth")[field].rank(method="average", pct=True)
    # itertuples preserves long integer IDs even beside floating metrics.
    profile = {int(row.gid): row for row in cohort.itertuples(index=False)}
    report_nodes, rows = {}, []
    for gid in gids:
        days = daily[gid]
        ordered_days = sorted(days)
        median_tx = float(median([day["activity_tx"] for day in days.values()])) if days else 0.0
        median_kzt = float(median([day["activity_kzt"] for day in days.values()])) if days else 0.0
        daily_rows, pairs, synchronous, bursts, same_day = [], [], [], [], []
        for day in ordered_days:
            item = days[day]
            daily_rows.append({
                "date": day.isoformat(),
                **{key: value for key, value in item.items() if key not in {"senders", "recipients"}},
                "n_senders": len(item["senders"]), "n_recipients": len(item["recipients"]),
            })
            if item["in_tx"] and item["out_tx"]:
                same_day.append(day.isoformat())
            if item["in_tx"]:
                for lag in (1, 2):
                    later_date = day + timedelta(days=lag)
                    later = days.get(later_date)
                    if later and later["out_tx"]:
                        pairs.append({"incoming_date": day.isoformat(), "outgoing_date": later_date.isoformat(),
                                      "lag_days": lag, "in_tx": item["in_tx"], "out_tx": later["out_tx"],
                                      "in_kzt": item["in_kzt"], "out_kzt": later["out_kzt"]})
            external_senders = item["senders"] - {gid}
            if len(external_senders) >= 3:
                synchronous.append({"date": day.isoformat(), "n_senders": len(external_senders),
                                    "in_tx": item["in_tx"], "in_kzt": item["in_kzt"],
                                    "sender_gids": [str(value) for value in sorted(external_senders)]})
            count_burst = item["activity_tx"] >= 3 and item["activity_tx"] >= 3 * median_tx
            amount_burst = item["activity_tx"] >= 3 and item["activity_kzt"] >= 3 * median_kzt
            if count_burst or amount_burst:
                bursts.append({"date": day.isoformat(), "activity_tx": item["activity_tx"],
                               "activity_kzt": item["activity_kzt"], "median_active_day_tx": median_tx,
                               "median_active_day_kzt": median_kzt, "count_burst": count_burst,
                               "amount_burst": amount_burst})
        p = profile[gid]
        adequate = p.cohort_size >= 20
        high_volume = bool(adequate and p.volume > 0 and p.volume_percentile >= 0.95)
        high_degree = bool(adequate and p.degree > 0 and p.degree_percentile >= 0.95)
        status = "insufficient_cohort" if not adequate else "outlier" if high_volume or high_degree else "no_signal"
        anomaly_evidence = (
            f"Колено {p.depth}, группа {p.cohort_size}: объём {p.volume:,.0f} KZT "
            f"(P={p.volume_percentile:.3f}), связи {p.degree} (P={p.degree_percentile:.3f}). "
            + ("Группа <20: сигнал не оценивается." if not adequate else
               "Порог P≥0.95; это профиль, не доказательство нарушения.")
        )
        temporal_evidence = (
            f"Активных дней {len(days)}; пар дней вход→выход за 1–2 дня {len(pairs)}; "
            f"синхронных входов {len(synchronous)}, всплесков {len(bursts)}, повторов сумм {len(repeated_by_node[gid])}. "
            "Движение тех же денег не доказано."
        )
        anomaly = {"depth": int(p.depth), "cohort_size": int(p.cohort_size), "observed_volume_kzt": float(p.volume),
                   "observed_degree": int(p.degree), "volume_percentile": float(p.volume_percentile),
                   "degree_percentile": float(p.degree_percentile), "high_volume": high_volume,
                   "high_degree": high_degree, "seed_outgoing_only": bool(p.is_seed), "status": status}
        report_nodes[str(gid)] = {
            "daily": daily_rows, "lag_1_2_day_pairs": pairs, "same_day_flow_dates": same_day,
            "synchronous_incoming": synchronous, "bursts": bursts,
            "equal_amount_groups": repeated_by_node[gid], "depth_anomaly": anomaly,
        }
        rows.append({
            "gid": gid, "temporal_active_days": len(days), "temporal_lag_1_2_day_pairs": len(pairs),
            "temporal_same_day_flow_days": len(same_day), "temporal_synchronous_days": len(synchronous),
            "temporal_burst_days": len(bursts), "temporal_equal_amount_groups": len(repeated_by_node[gid]),
            "temporal_evidence": temporal_evidence, "anomaly_depth_cohort_size": int(p.cohort_size),
            "anomaly_volume_percentile": float(p.volume_percentile), "anomaly_degree_percentile": float(p.degree_percentile),
            "anomaly_status": status, "anomaly_evidence": anomaly_evidence,
        })
    columns = ["gid", *COUNT_COLUMNS[:6], "temporal_evidence", "anomaly_depth_cohort_size",
               "anomaly_volume_percentile", "anomaly_degree_percentile", "anomaly_status", "anomaly_evidence"]
    metrics = pd.DataFrame(rows, columns=columns).astype({
        "gid": "int64", **{column: "int64" for column in COUNT_COLUMNS},
        "anomaly_volume_percentile": "float64", "anomaly_degree_percentile": "float64",
        "temporal_evidence": "str", "anomaly_status": "str", "anomaly_evidence": "str",
    })
    report = {
        "settings": {"lag_calendar_days": [1, 2], "synchronous_min_senders": 3,
                     "burst_min_tx": 3, "burst_median_multiplier": 3,
                     "equal_amount_min_tx_per_pair_day": 3,
                     "anomaly_min_depth_cohort": 20, "anomaly_percentile_threshold": 0.95,
                     "percentile_ties": "average_rank", "seed_metrics": "outgoing_only"},
        "n_nodes": len(gids), "n_transactions": len(tx), "equal_amount_groups_total": n_repeat_groups,
        "nodes": report_nodes,
        "limitations": [
            "Пары дней отражают последовательность наблюдаемых входов и выходов, а не передачу тех же денег.",
            "Переводы в один календарный день не упорядочиваются, даже если источник содержит время.",
            "Один день может участвовать в нескольких парах; суммы пар нельзя складывать как объём транзита.",
            "Совпадение сумм и синхронность не доказывают дробление, координацию или незаконное назначение.",
            "Переводы ниже 5 000 KZT невидимы в предоставленной выборке; их наличие или отсутствие не оценивается.",
            "Данные охватывают только наблюдаемые внутрибанковские переводы; входы seed особенно неполны.",
            "На четвёртом колене исходящие обрезаны; отсутствие временного сигнала не означает отсутствие дальнейших операций.",
            "Всплески сравниваются с медианой только активных дней; редкая активность даёт мало оснований для сравнения.",
            "Профиль сравнивается только с тем же коленом, отражает охват выгрузки и не является вероятностью нарушения.",
        ],
    }
    return metrics, report
