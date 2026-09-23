"""Analyst view of community sensitivity and observed boundary transfers."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def _neighbor_flows(roles, edges, cluster_id):
    """Aggregate original directed boundary edges, retaining transfer counts."""
    membership = roles.set_index("gid")["cluster_id"]
    source = edges["src"].map(membership)
    target = edges["dst"].map(membership)
    outgoing = edges.loc[source.eq(cluster_id) & target.ne(cluster_id) & target.notna()].copy()
    incoming = edges.loc[target.eq(cluster_id) & source.ne(cluster_id) & source.notna()].copy()
    outgoing["other"] = target.loc[outgoing.index]
    incoming["other"] = source.loc[incoming.index]
    out = outgoing.groupby("other").agg(out_kzt=("sum_kzt", "sum"), out_tx=("n_tx", "sum"))
    inc = incoming.groupby("other").agg(in_kzt=("sum_kzt", "sum"), in_tx=("n_tx", "sum"))
    result = inc.join(out, how="outer").fillna(0).rename_axis("cluster_id").reset_index()
    result["total_kzt"] = result["in_kzt"] + result["out_kzt"]
    return result.sort_values(["total_kzt", "cluster_id"], ascending=[False, True])


def _disputed_clients(roles, edges, members, cluster_id):
    """Find each client's strongest other group from observed incident sums."""
    membership = roles.set_index("gid")["cluster_id"]
    rows = []
    for member in members.itertuples(index=False):
        # Keep each observed edge once, including any self-transfer.
        incident = edges.loc[edges["src"].eq(member.gid) | edges["dst"].eq(member.gid)]
        counterpart = incident["dst"].where(incident["src"].eq(member.gid), incident["src"])
        groups = counterpart.map(membership)
        amounts = incident.assign(other=groups).groupby("other")["sum_kzt"].sum()
        own_amount = float(amounts.get(cluster_id, 0.0))
        alternatives = amounts.drop(index=cluster_id, errors="ignore").rename("amount").reset_index()
        alternatives = alternatives.sort_values(["amount", "other"], ascending=[False, True])
        if alternatives.empty:
            other_group, other_amount = "Нет внешних связей", 0.0
        else:
            strongest = alternatives.iloc[0]
            other_group = f"Группа {int(strongest['other'])}"
            other_amount = float(strongest["amount"])
        support = member.cluster_membership_stability
        rows.append({
            "Клиент": str(member.gid),
            "Повторяемость, %": float(support) * 100 if pd.notna(support) else None,
            "Наиболее связанная другая группа": other_group,
            "Переводы с другой группой, ₸": other_amount,
            "Переводы со своей группой, ₸": own_amount,
        })
    return pd.DataFrame(rows).sort_values(
        ["Переводы с другой группой, ₸", "Клиент"], ascending=[False, True]
    ) if rows else pd.DataFrame()


def render_cluster_quality(roles, edges, cluster_id, summary=None):
    """Render optional sensitivity diagnostics; old exports still show flows.

    ``summary`` may be a Series with stability_mean/min/status/stability_runs.
    Membership calculations belong to the pipeline; this view does not rerun
    clustering or change any analytical score.
    """
    members = roles.loc[roles["cluster_id"].eq(cluster_id)]
    st.markdown("#### Насколько устойчив состав группы")
    if members.empty:
        st.info("В этой группе нет клиентов в загруженной выборке.")
        return

    required = {"cluster_membership_stability", "cluster_membership_status"}
    if not required.issubset(members.columns):
        st.info("В этой выгрузке ещё нет оценки устойчивости. Пересчитайте пайплайн и обновите результаты. Связи с соседними группами показаны ниже.")
    else:
        status = members["cluster_membership_status"]
        n_core = int(status.eq("core").sum())
        n_disputed = int(status.eq("disputed").sum())
        n_isolated = int(status.eq("isolated").sum())
        support = pd.to_numeric(members["cluster_membership_stability"], errors="coerce")
        support = support.where(support.between(0, 1))
        mean = summary.get("stability_mean") if summary is not None else None
        minimum = summary.get("stability_min") if summary is not None else None
        mean = float(mean) if mean is not None and pd.notna(mean) else None
        minimum = float(minimum) if minimum is not None and pd.notna(minimum) else None
        metrics = st.columns(3)
        metrics[0].metric("Устойчивое ядро", n_core)
        metrics[1].metric("Спорное отнесение", n_disputed)
        metrics[2].metric("Сходство состава (Жаккар), среднее", f"{mean * 100:.1f}%" if mean is not None else "Нет данных")
        if n_isolated:
            st.caption(f"Изолированных клиентов: {n_isolated}. Отсутствие наблюдаемых связей не подтверждает принадлежность к содержательной группе.")
        if minimum is not None:
            st.caption(f"Минимальное сходство состава группы (Жаккар): {minimum * 100:.1f}%. Устойчивая группа: минимум ≥80%; изменчивая: ≥50%; иначе — неустойчивая.")
        if mean is None or minimum is None:
            st.caption("Групповые показатели сходства не переданы. Они не заменяются средними оценками отдельных клиентов.")
        if summary is not None:
            value = summary.get("stability_status")
            labels = {
                "stable": "состав устойчив в проверенных запусках",
                "mixed": "устойчивость состава неоднородна",
                "variable": "состав изменчив при смене случайного старта",
                "unstable": "границы меняются при смене случайного старта",
                "isolated": "изолированная вершина без наблюдаемых связей",
            }
            if value in labels:
                if value in {"variable", "unstable"}:
                    st.warning("Оценка группы: " + labels[value] + ". Проверьте соседние группы перед выводом об общей структуре.")
                else:
                    st.caption("Оценка группы: " + labels[value] + ".")
        st.caption("Ядро может сохраняться, даже если группа объединяется с соседней. Поэтому отсутствие спорных участников не гарантирует устойчивости границ всей группы.")
        runs = summary.get("stability_runs") if summary is not None else None
        if runs is not None and pd.notna(runs):
            run_text = f"Выполнено повторных запусков: {int(runs)}. "
        else:
            run_text = "Метод предусматривает 10 воспроизводимых повторных запусков. "
        st.caption(run_text + "Группы сопоставляются по наибольшему совпадению состава (индекс Жаккара). Повторяемость клиента — доля запусков, в которых он остался в сопоставленной группе. Это не вероятность правильности или нарушения.")
        st.caption("В устойчивое ядро входят неизолированные клиенты с повторяемостью ≥80%. Изоляты выделены отдельно; высокая повторяемость одиночной вершины не подтверждает содержательную группу.")
        disputed = members.loc[status.eq("disputed")].copy()
        disputed["cluster_membership_stability"] = support.loc[disputed.index]
        if disputed.empty:
            st.success("Спорных участников по рассчитанному критерию не выявлено.")
        else:
            st.markdown("**Участники, чью принадлежность стоит проверить**")
            st.dataframe(_disputed_clients(roles, edges, disputed, cluster_id),
                hide_index=True, use_container_width=True,
                column_config={
                    "Клиент": st.column_config.TextColumn(width="medium"),
                    "Повторяемость, %": st.column_config.NumberColumn(format="%.1f"),
                    "Переводы с другой группой, ₸": st.column_config.NumberColumn(format="localized"),
                    "Переводы со своей группой, ₸": st.column_config.NumberColumn(format="localized"),
                })
            st.caption("Сначала показаны участники с наибольшей суммой переводов с другой группой. Другая группа выбрана по сумме входящих и исходящих; при равенстве выбран меньший номер группы. Это основание для просмотра связей, а не автоматическое переназначение.")

    st.markdown("#### Переводы с соседними группами")
    flows = _neighbor_flows(roles, edges, cluster_id)
    if flows.empty:
        st.info("Переводов между этой группой и другими группами в выборке нет.")
    else:
        display = pd.DataFrame({
            "Соседняя группа": flows["cluster_id"].map(lambda value: f"Группа {int(value)}"),
            "Получено от группы, ₸": flows["in_kzt"],
            "Входящих переводов": flows["in_tx"].astype("int64"),
            "Отправлено группе, ₸": flows["out_kzt"],
            "Исходящих переводов": flows["out_tx"].astype("int64"),
            "Сумма обоих направлений, ₸": flows["total_kzt"],
        })
        st.dataframe(display, hide_index=True, use_container_width=True,
            column_config={column: st.column_config.NumberColumn(format="localized")
                           for column in display.columns if "₸" in column})
        st.caption("Группы отсортированы по сумме переводов в обоих направлениях. Число переводов взято из n_tx; каждая направленная связь учтена в своём направлении.")
    st.caption("Суммы отражают только наблюдаемую выборку и могут повторно учитывать деньги на разных этапах движения. Сильная связь и устойчивое сообщество не доказывают совместную преступную деятельность; отсутствие клиентов исходного списка не доказывает непричастность.")
