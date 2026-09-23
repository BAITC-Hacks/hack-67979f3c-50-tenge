"""Local analyst dashboard over completed CSV exports."""

from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
DATA = ROOT / "data"
COLORS = {
    "consolidator": "#d97706", "transit": "#2563eb",
    "distributor": "#7c3aed", "terminal": "#0d9488",
    "coordinator": "#dc2626", "peripheral": "#64748b",
}
ROLE_NAMES = {
    "consolidator": "Консолидация", "transit": "Транзит",
    "distributor": "Распределение", "terminal": "Конечный получатель",
    "coordinator": "Координация", "peripheral": "Периферия",
}


def _stamp(path: Path) -> int:
    return path.stat().st_mtime_ns


@st.cache_data(show_spinner=False)
def load_tables(stamps: tuple[int, ...]):
    del stamps
    return (
        pd.read_csv(OUT / "nodes_roles.csv"),
        pd.read_csv(OUT / "clusters.csv"),
        pd.read_csv(OUT / "top_nodes.csv"),
        pd.read_parquet(DATA / "edges.parquet"),
    )


def node_value(row, key, default="—"):
    value = row.get(key, default)
    return default if pd.isna(value) else value


def neighborhood(edges: pd.DataFrame, gid: int, hops: int, max_nodes: int = 200):
    """Undirected traversal for display; edge directions are retained."""
    adjacency = {}
    for edge in edges.itertuples(index=False):
        adjacency.setdefault(int(edge.src), set()).add(int(edge.dst))
        adjacency.setdefault(int(edge.dst), set()).add(int(edge.src))
    seen = {gid}
    frontier = {gid}
    for _ in range(hops):
        frontier = set().union(*(adjacency.get(n, set()) for n in frontier)) - seen
        seen.update(frontier)
    ordered = [gid] + sorted(seen - {gid})
    visible = set(ordered[:max_nodes])
    return visible, len(seen) - len(visible)


def graph_html(edges, roles, gid, hops):
    visible, hidden = neighborhood(edges, gid, hops)
    table = roles.set_index("gid")
    graph = Network(height="580px", width="100%", directed=True, cdn_resources="in_line")
    graph.barnes_hut()
    for node_id in sorted(visible):
        if node_id not in table.index:
            continue
        row = table.loc[node_id]
        role = str(node_value(row, "role", "peripheral"))
        seed = bool(node_value(row, "is_seed", False))
        boundary = bool(node_value(row, "truncated_by_depth", False))
        score = float(node_value(row, "priority_score", 0))
        suffix = " ★" if seed else (" ◇" if boundary else "")
        graph.add_node(str(node_id), label=str(node_id) + suffix,
                       color=COLORS.get(role, "#64748b"),
                       size=13 + 25 * score,
                       borderWidth=4 if node_id == gid else 1,
                       title=f"gid {node_id}<br>{ROLE_NAMES.get(role, role)}<br>"
                             f"Приоритет: {score:.3f}<br>seed: {seed}; граница: {boundary}")
    subset = edges[edges.src.isin(visible) & edges.dst.isin(visible)]
    for edge in subset.itertuples(index=False):
        title = f"{float(edge.sum_kzt):,.0f} KZT; {int(edge.n_tx)} переводов"
        graph.add_edge(str(int(edge.src)), str(int(edge.dst)), arrows="to", title=title)
    return graph.generate_html(), hidden


def main():
    st.set_page_config(page_title="Граф денег", layout="wide")
    st.title("Граф денег")
    st.caption("Гипотезы для проверки по наблюдаемым переводам, не вывод о виновности.")
    required = [OUT / name for name in ("nodes_roles.csv", "clusters.csv", "top_nodes.csv")]
    required.append(DATA / "edges.parquet")
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        st.error("Нет файлов: " + ", ".join(missing) + ". Сначала выполните python run.py --data data --out out")
        st.stop()
    roles, clusters, top, edges = load_tables(tuple(_stamp(p) for p in required))
    if not roles.gid.is_unique:
        st.error("В nodes_roles.csv повторяются gid")
        st.stop()
    for name, path in zip(("Узлы", "Кластеры", "Top-30"), required[:3]):
        st.download_button(f"Скачать {name}", path.read_bytes(), file_name=path.name, mime="text/csv")

    st.subheader("Приоритет проверки")
    role_filter = st.multiselect("Роль", sorted(roles.role.dropna().unique()))
    cluster_filter = st.multiselect("Кластер", sorted(roles.cluster_id.dropna().unique()))
    seed_filter = st.selectbox("Seed", ["Все", "Да", "Нет"])
    listing = top.merge(roles[["gid", "cluster_id", "is_seed"]], on="gid", validate="one_to_one")
    if role_filter:
        listing = listing[listing.role.isin(role_filter)]
    if cluster_filter:
        listing = listing[listing.cluster_id.isin(cluster_filter)]
    if seed_filter != "Все":
        listing = listing[listing.is_seed.eq(seed_filter == "Да")]
    st.dataframe(listing, hide_index=True, use_container_width=True)

    st.subheader("Карточка узла")
    raw_gid = st.text_input("gid", value=str(int(roles.gid.iloc[0])))
    try:
        gid = int(raw_gid.strip())
    except ValueError:
        st.warning("Введите целочисленный gid")
        st.stop()
    found = roles[roles.gid.eq(gid)]
    if found.empty:
        st.warning(f"gid {gid} отсутствует в данных. Поиск охватывает все {len(roles)} узлов, включая изоляты.")
        st.stop()
    row = found.iloc[0]
    role = str(node_value(row, "role"))
    st.markdown(f"**{ROLE_NAMES.get(role, role)}** · score роли {float(node_value(row, 'role_score', 0)):.3f} · "
                f"приоритет {float(node_value(row, 'priority_score', 0)):.3f}")
    st.write("Основание:", node_value(row, "evidence"))
    st.write("Правило и порог:", node_value(row, "role_rule"))
    st.caption("Численные пороги и формула правила приведены в docs/roles.md; фактические метрики ниже.")
    fields = ["in_deg", "out_deg", "in_tx", "out_tx", "in_kzt", "out_kzt", "depth",
              "is_seed", "truncated_by_depth", "pagerank", "betweenness", "seed_reach_count",
              "seed_distance", "pass_through", "priority_volume", "priority_bridge",
              "priority_seed", "priority_degree"]
    st.dataframe(pd.DataFrame([{"Метрика": field, "Значение": node_value(row, field)}
                               for field in fields]), hide_index=True)
    if bool(node_value(row, "truncated_by_depth", False)):
        st.warning("Граница обхода: отсутствие исходящих переводов на глубине 4 не означает удержание денег.")
    st.markdown("**Что неизвестно**")
    st.write("Переводы вне этого банка и периода, суммы ниже 5 000 KZT, остатки на счетах и личность клиента. "
             "Входящие суммы seed могут быть неполными. Score — относительная оценка, не вероятность.")
    st.markdown("**Что запросить для проверки**")
    st.write("Полную историю счёта и связанных счетов за соседние периоды. Для граничного узла — дальнейшие "
             "переводы за четвёртым коленом; для seed — входящие переводы вне текущей выборки.")

    cluster_id = node_value(row, "cluster_id")
    summary = clusters[clusters.cluster_id.eq(cluster_id)]
    st.subheader(f"Кластер {cluster_id}")
    if not summary.empty:
        st.dataframe(summary, hide_index=True, use_container_width=True)

    hops = st.radio("Окружение", [1, 2], horizontal=True)
    html, hidden = graph_html(edges, roles, gid, hops)
    st.caption(f"Стрелка показывает направление перевода; ★ — seed, ◇ — граница. "
               f"Скрыто узлов из-за лимита 200: {hidden}.")
    components.html(html, height=600, scrolling=True)


if __name__ == "__main__":
    main()
