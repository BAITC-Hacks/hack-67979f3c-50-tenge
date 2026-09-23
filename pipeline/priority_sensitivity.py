"""Finite weight sensitivity experiment, not an accuracy estimate."""

import numpy as np

from pipeline.priority import CONTRIBUTION_COLUMNS, compute_priority, make_top_nodes


def analyze_priority_sensitivity(nodes):
    """Vary each weight by +/-20%, renormalize, recompute official selection."""
    baseline = compute_priority(nodes)
    weights = np.array([.35, .30, .20, .15])
    features = baseline[list(CONTRIBUTION_COLUMNS)].to_numpy() / weights
    base_top = make_top_nodes(baseline)
    base_ids = set(map(int, base_top.gid))
    ranks = {int(g): [] for g in baseline.gid}
    hits = {int(g): 0 for g in baseline.gid}
    scenarios = []
    for index, name in enumerate(CONTRIBUTION_COLUMNS):
        for factor in (.8, 1.2):
            changed = weights.copy()
            changed[index] *= factor
            changed /= changed.sum()
            frame = baseline.copy(deep=True)
            frame[list(CONTRIBUTION_COLUMNS)] = features * changed
            frame['priority_score'] = frame[list(CONTRIBUTION_COLUMNS)].sum(axis=1).clip(0, 1)
            top = make_top_nodes(frame)
            ids = set(map(int, top.gid))
            for position, gid in enumerate(frame.sort_values(
                    ['priority_score', 'gid'], ascending=[False, True]).gid, 1):
                ranks[int(gid)].append(position)
                hits[int(gid)] += int(int(gid) in ids)
            scenarios.append({
                'factor': name, 'multiplier': factor,
                'weights': dict(zip(CONTRIBUTION_COLUMNS, map(float, changed))),
                'retained': len(ids & base_ids), 'baseline_size': len(base_ids),
                'selection_policy': top.attrs['selection_policy'],
                'entered': [str(g) for g in sorted(ids - base_ids)],
                'left': [str(g) for g in sorted(base_ids - ids)],
            })
    return {
        'method': 'Каждый из 4 весов отдельно изменён на ±20%, затем веса нормированы до суммы 1: всего 8 сценариев. Политика отбора seed пересчитывается в каждом сценарии.',
        'caveat': 'Устойчивость к выбранным весам не означает правильность роли или подозрительность. Изменение данных, всех весов одновременно и порогов ролей здесь не проверяется.',
        'baseline_selection_policy': base_top.attrs['selection_policy'],
        'scenarios': scenarios,
        'nodes': {str(g): {'top_hits': hits[g], 'scenarios': len(scenarios),
                          'rank_min': min(values), 'rank_max': max(values)}
                  for g, values in ranks.items()},
    }
