"""Bounded route motifs and structural removal scenarios, not money tracing."""

from __future__ import annotations

import heapq
import math
import random
from collections import defaultdict

import networkx as nx
import pandas as pd


RESULT_LIMIT = 100
EVIDENCE_LIMIT = 6
RANDOM_RUNS = 20


def _keep_best(heap, key, record):
    item = (key, record)
    if len(heap) < RESULT_LIMIT:
        heapq.heappush(heap, item)
    elif key > heap[0][0]:
        heapq.heapreplace(heap, item)


def _edge_record(G, src, dst):
    attrs = G[src][dst]
    return {"src": str(src), "dst": str(dst), "sum_kzt": float(attrs["sum_kzt"]),
            "n_tx": int(attrs["n_tx"])}


def _recurring_routes(G, transactions):
    frame = transactions.copy()
    frame["day"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if frame["day"].isna().any():
        raise ValueError("В транзакциях маршрутов не должно быть пустых дат")
    daily = frame.groupby(["src", "dst", "day"], sort=True).agg(
        n_tx=("sum_kzt", "size"), sum_kzt=("sum_kzt", "sum")
    ).reset_index()
    by_pair = defaultdict(dict)
    for item in daily.itertuples(index=False):
        by_pair[(item.src, item.dst)][item.day] = (int(item.n_tx), float(item.sum_kzt))
    repeated = {pair: days for pair, days in by_pair.items()
                if sum(value[0] for value in days.values()) >= 2}
    incoming, outgoing = defaultdict(list), defaultdict(list)
    for src, dst in sorted(repeated):
        if src != dst and G.has_edge(src, dst):
            incoming[dst].append(src)
            outgoing[src].append(dst)
    eligible = considered = 0
    best = []
    for middle in sorted(set(incoming) & set(outgoing)):
        for src in incoming[middle]:
            for dst in outgoing[middle]:
                if src == dst:
                    continue
                considered += 1
                first, second = repeated[(src, middle)], repeated[(middle, dst)]
                matched_days, matched_pairs, examples = 0, 0, []
                for day, (in_count, in_amount) in sorted(first.items()):
                    matches = 0
                    for lag in (1, 2):
                        later = day + pd.Timedelta(days=lag)
                        if later not in second:
                            continue
                        matches += 1
                        matched_pairs += 1
                        out_count, out_amount = second[later]
                        if len(examples) < EVIDENCE_LIMIT:
                            examples.append({
                                "incoming_date": day.date().isoformat(),
                                "outgoing_date": later.date().isoformat(), "lag_days": lag,
                                "incoming_n_tx": in_count, "incoming_sum_kzt": in_amount,
                                "outgoing_n_tx": out_count, "outgoing_sum_kzt": out_amount,
                            })
                    if matches:
                        matched_days += 1
                if matched_days < 2:
                    continue
                eligible += 1
                first_sum = math.fsum(value[1] for value in first.values())
                second_sum = math.fsum(value[1] for value in second.values())
                record = {
                    "gids": [str(src), str(middle), str(dst)],
                    "matched_incoming_days": matched_days, "matched_day_pairs": matched_pairs,
                    "first_leg_n_tx": sum(value[0] for value in first.values()),
                    "second_leg_n_tx": sum(value[0] for value in second.values()),
                    "first_leg_sum_kzt": first_sum, "second_leg_sum_kzt": second_sum,
                    "evidence": examples, "evidence_total": matched_pairs,
                    "evidence_truncated": matched_pairs > EVIDENCE_LIMIT,
                    "why": f"На обоих участках ≥2 переводов; на {matched_days} разных входящих датах наблюдается выход через 1–2 дня. Связь тех же денег не установлена.",
                }
                _keep_best(best, (matched_days, first_sum + second_sum, -int(src), -int(middle), -int(dst)), record)
    return {"items": [record for _, record in sorted(best, key=lambda item: item[0], reverse=True)],
            "eligible_total": eligible, "candidate_total": considered, "limit": RESULT_LIMIT,
            "truncated": eligible > RESULT_LIMIT,
            "method": "Три разных клиента A→B→C; каждый участок содержит ≥2 транзакций; ≥2 разных даты входа с выходом через 1–2 дня. Сортировка: число входящих дат, сумма обоих участков, gid.",
            "caveat": "Переводы одного дня исключены: внутридневной порядок неизвестен. Один исходящий день может соответствовать нескольким входящим; даты не образуют однозначное сопоставление транзакций и не доказывают движение тех же денег."}


def _short_cycles(G):
    best = []
    eligible = 0
    by_length = {"2": 0, "3": 0}

    def add(nodes):
        nonlocal eligible
        eligible += 1
        by_length[str(len(nodes))] += 1
        edges = [_edge_record(G, nodes[index], nodes[(index + 1) % len(nodes)])
                 for index in range(len(nodes))]
        amount = math.fsum(edge["sum_kzt"] for edge in edges)
        record = {"gids": [str(gid) for gid in nodes], "length": len(nodes),
                  "sum_kzt_edges": amount, "n_tx_edges": sum(edge["n_tx"] for edge in edges),
                  "edges": edges,
                  "why": f"Направленный цикл длины {len(nodes)} в агрегированных переводах. Временной порядок и возврат тех же денег не установлены."}
        _keep_best(best, (amount, -len(nodes), tuple(-int(gid) for gid in nodes)), record)

    # A is always the smallest gid: unique rotation, opposite direction retained.
    for first in sorted(G):
        for second in sorted(G.successors(first)):
            if second <= first:
                continue
            if G.has_edge(second, first):
                add((first, second))
            for third in sorted(G.successors(second)):
                if third > first and third != second and G.has_edge(third, first):
                    add((first, second, third))
    return {"items": [record for _, record in sorted(best, key=lambda item: item[0], reverse=True)],
            "eligible_total": eligible, "by_length": by_length, "limit": RESULT_LIMIT,
            "truncated": eligible > RESULT_LIMIT,
            "method": "Только направленные циклы длины 2 и 3. Циклические сдвиги объединены; обратное направление сохранено. Сортировка по сумме наблюдаемых рёбер.",
            "caveat": "Это топологические циклы за весь период, не доказанные хронологические возвраты. Сумма рёбер не является объёмом уникальных денег. Циклы большей длины не анализируются."}


def _structural_stats(G, removed, original_largest, original_weight):
    remaining = set(G) - set(removed)
    residual = G.subgraph(remaining)
    components = list(nx.weakly_connected_components(residual))
    largest = max((len(component) for component in components), default=0)
    largest_remaining = original_largest & remaining
    fragments = list(nx.weakly_connected_components(G.subgraph(largest_remaining)))
    fragment_largest = max((len(component) for component in fragments), default=0)
    weight = math.fsum(float(data["sum_kzt"]) for _, _, data in residual.edges(data=True))
    return {
        "removed_node_count": len(removed), "remaining_node_count": len(remaining),
        "weak_component_count": len(components), "largest_component_nodes": largest,
        "largest_component_fraction_remaining": largest / len(remaining) if remaining else 0.0,
        "remaining_edge_count": residual.number_of_edges(), "retained_edge_weight_kzt": weight,
        "retained_edge_weight_fraction": weight / original_weight if original_weight else 0.0,
        "original_largest_removed_nodes": len(original_largest) - len(largest_remaining),
        "original_largest_remaining_nodes": len(largest_remaining),
        "original_largest_fragment_count": len(fragments),
        "original_largest_fragment_nodes": fragment_largest,
        "original_largest_nodes_outside_largest_fragment": len(largest_remaining) - fragment_largest,
    }


def _resilience(G, scored):
    all_nodes = sorted(G)
    original_largest = max(nx.weakly_connected_components(G), key=lambda part: (len(part), -min(part)), default=set())
    original_weight = math.fsum(float(data["sum_kzt"]) for _, _, data in G.edges(data=True))
    baseline = _structural_stats(G, set(), original_largest, original_weight)
    ordered = scored.sort_values(["priority_score", "gid"], ascending=[False, True])["gid"].tolist()
    scenarios = []
    rng = random.Random(42)
    for count in sorted({min(n, len(all_nodes)) for n in (1, 3, 5, 10, 20)} - {0}):
        removed = set(ordered[:count])
        stats = _structural_stats(G, removed, original_largest, original_weight)
        stats["removed_gids"] = [str(gid) for gid in ordered[:count]]
        random_largest = []
        for _ in range(RANDOM_RUNS):
            sample = set(rng.sample(all_nodes, count))
            residual = G.subgraph(set(all_nodes) - sample)
            random_largest.append(max((len(part) for part in nx.weakly_connected_components(residual)), default=0))
        stats["random_baseline"] = {
            "runs": RANDOM_RUNS, "largest_component_nodes_mean": math.fsum(random_largest) / RANDOM_RUNS,
            "largest_component_nodes_min": min(random_largest),
            "largest_component_nodes_max": max(random_largest),
        }
        stats["largest_component_difference_from_random_mean"] = stats["largest_component_nodes"] - stats["random_baseline"]["largest_component_nodes_mean"]
        scenarios.append(stats)
    return {"baseline": baseline, "scenarios": scenarios, "random_seed": 42,
            "method": "Удаляются top-N всех узлов по priority_score (при равенстве gid), N=1,3,5,10,20 с ограничением размером графа. Сравнение с 20 случайными наборами того же размера, без повторов внутри набора, seed=42.",
            "caveat": "Моделируется удаление вершин из наблюдаемого графа. Снижение размера из-за удаления отделено от распада исходной крупнейшей компоненты. Слабая связность не описывает направленную достижимость денег; исчезнувшие рёбра — наблюдаемый оборот, не предотвращённые переводы. Случайный baseline даёт контекст, не гарантирует отключение реальной сети."}


def analyze_routes(G, transactions, scored):
    """Return JSON-compatible motifs and resilience; never mutate inputs."""
    if not G.is_directed() or G.is_multigraph():
        raise ValueError("Для анализа маршрутов нужен nx.DiGraph")
    if not {"src", "dst", "date", "sum_kzt"}.issubset(transactions.columns):
        raise ValueError("Транзакции должны содержать src, dst, date, sum_kzt")
    if not {"gid", "priority_score"}.issubset(scored.columns):
        raise ValueError("Для сценариев удаления нужны gid и priority_score")
    if scored["gid"].isna().any() or not scored["gid"].is_unique or set(scored["gid"]) != set(G):
        raise ValueError("Таблица приоритета должна содержать каждый узел графа ровно один раз")
    if not scored["priority_score"].map(lambda value: math.isfinite(value) and 0 <= value <= 1).all():
        raise ValueError("Приоритет должен быть конечным числом от 0 до 1")
    for _, _, attrs in G.edges(data=True):
        amount, count = float(attrs["sum_kzt"]), float(attrs["n_tx"])
        if not math.isfinite(amount) or amount <= 0 or not math.isfinite(count) or count < 1 or count != int(count):
            raise ValueError("Рёбра должны содержать положительную сумму и целое положительное n_tx")
    if not transactions["sum_kzt"].map(lambda value: math.isfinite(value) and value > 0).all():
        raise ValueError("Суммы транзакций должны быть конечными положительными числами")
    if any(not G.has_edge(src, dst) for src, dst in transactions[["src", "dst"]].itertuples(index=False, name=None)):
        raise ValueError("Транзакция ссылается на отсутствующее ребро графа")
    return {"schema_version": 1, "recurring_routes": _recurring_routes(G, transactions),
            "short_cycles": _short_cycles(G), "resilience": _resilience(G, scored),
            "limitations": "Только наблюдаемые внутрибанковские переводы: глубина 4, один месяц, порог 5 000 KZT. Анализ не меняет роли или приоритет и не доказывает происхождение денег либо виновность."}
