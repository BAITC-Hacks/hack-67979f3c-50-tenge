"""Local analyst dashboard over completed CSV exports."""

import json
import colorsys
from html import escape
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
    "consolidator": "Признаки сбора средств", "transit": "Передаёт средства дальше",
    "distributor": "Распределяет средства", "terminal": "Нет видимых переводов дальше",
    "coordinator": "Связующий участник", "peripheral": "Роль не определена",
}


def money(value):
    return f"{float(value):,.0f}".replace(",", " ") + " ₸"


def client_table(frame):
    """Display-only strings preserve the full int64 identifiers in browsers."""
    return pd.DataFrame({
        "Клиент": frame.gid.astype(str),
        "Приоритет / 100": (frame.priority_score * 100).round(1),
        "Предполагаемая роль": frame.role.map(ROLE_NAMES).fillna(frame.role),
        "Группа": frame.cluster_id.astype(str),
        "В исходном списке": frame.is_seed.map({True: "Да", False: "Нет"}),
    }).reset_index(drop=True)


def cluster_color(cluster_id: int) -> str:
    """Stable categorical color for a cluster, independent of display order."""
    hue = (int(cluster_id) * 0.61803398875) % 1
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.68, 0.82)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


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
        json.loads((OUT / "run_metadata.json").read_text(encoding="utf-8")),
    )


def node_value(row, key, default="—"):
    value = row.get(key, default)
    return default if pd.isna(value) else value


def neighborhood(edges: pd.DataFrame, gid: int, hops: int, max_nodes: int = 200):
    """Show a connected prefix of BFS layers; count the complete neighborhood."""
    adjacency = {}
    for edge in edges.itertuples(index=False):
        adjacency.setdefault(int(edge.src), set()).add(int(edge.dst))
        adjacency.setdefault(int(edge.dst), set()).add(int(edge.src))
    seen = {gid}
    visible = {gid}
    frontier = [gid]
    for _ in range(hops):
        following = set()
        for parent in sorted(frontier):
            for neighbor in sorted(adjacency.get(parent, ())):
                if neighbor not in seen:
                    seen.add(neighbor)
                    following.add(neighbor)
                if parent in visible and len(visible) < max_nodes:
                    visible.add(neighbor)
        frontier = sorted(following)
    return visible, len(seen) - len(visible)


def filtered_nodes(roles, selected_roles, selected_clusters, seed_filter):
    result = roles
    if selected_roles:
        result = result[result.role.isin(selected_roles)]
    if selected_clusters:
        result = result[result.cluster_id.isin(selected_clusters)]
    if seed_filter != "Все":
        result = result[result.is_seed.eq(seed_filter == "Да")]
    return result.sort_values(["priority_score", "gid"], ascending=[False, True]).copy()


def group_members(roles, cluster_id, selected_ids=None):
    """Return all or selected members in priority order, preserving int64 gid."""
    members = roles.loc[roles.cluster_id.eq(cluster_id)]
    if selected_ids is not None:
        members = members.loc[members.gid.astype(str).isin(selected_ids)]
    return members.sort_values(["priority_score", "gid"], ascending=[False, True]).copy()


def graph_html(edges, roles, gid, hops, color_by="По ролям"):
    visible, hidden = neighborhood(edges, gid, hops)
    table = roles.set_index("gid")
    graph = Network(height="510px", width="100%", directed=True,
                    bgcolor="#111827", font_color="#f8fafc", cdn_resources="in_line")
    graph.barnes_hut()
    for node_id in sorted(visible):
        if node_id not in table.index:
            continue
        row = table.loc[node_id]
        role = str(node_value(row, "role", "peripheral"))
        seed = bool(node_value(row, "is_seed", False))
        boundary = bool(node_value(row, "truncated_by_depth", False))
        score = float(node_value(row, "priority_score", 0))
        cluster = int(row.cluster_id)
        color = cluster_color(cluster) if color_by == "По группам" else COLORS.get(role, "#64748b")
        suffix = (" ★" if seed else "") + (" ◇" if boundary else "")
        label = "…" + str(node_id)[-7:] + suffix
        if node_id == gid:
            label = "Выбранный\n" + label
        graph.add_node(str(node_id), label=label,
                       color={"background": color, "border": "#f8fafc" if node_id == gid else color},
                       size=15 + 25 * score, borderWidth=4 if node_id == gid else 1,
                       title=f"Клиент {node_id}<br>{escape(ROLE_NAMES.get(role, role))}<br>"
                             f"Приоритет: {score * 100:.1f} / 100; группа {cluster}<br>"
                             f"В исходном списке: {'да' if seed else 'нет'}; "
                             f"граница данных: {'да' if boundary else 'нет'}")
    subset = edges[edges.src.isin(visible) & edges.dst.isin(visible)]
    for edge in subset.itertuples(index=False):
        title = f"{edge.src} → {edge.dst}<br>{money(edge.sum_kzt)}; {int(edge.n_tx)} переводов"
        graph.add_edge(str(int(edge.src)), str(int(edge.dst)), arrows="to", title=title,
                       color="#94a3b8")
    return graph.generate_html(), hidden


def review_reason(row):
    incoming, outgoing = int(row.in_deg), int(row.out_deg)
    if incoming + outgoing == 0:
        return "В выборке нет переводов этого клиента. Для оценки его связей данных недостаточно."
    text = (f"В выборке получил {money(row.in_kzt)} от {incoming} отправителей; "
            f"перевёл {money(row.out_kzt)} {outgoing} получателям.")
    reached = int(node_value(row, "seed_reach_count", 0))
    if reached:
        text += f" До него ведут направленные связи от {reached} клиентов исходного списка за 1–4 шага."
    return text


def audit_panel(row, metadata):
    with st.expander("Проверить расчёт: правило, пороги и вклад каждого признака"):
        st.write("Обоснование роли:", node_value(row, "evidence"))
        rule = str(node_value(row, "role_rule"))
        thresholds = metadata.get("role_thresholds", {})
        q75, q95 = thresholds.get("in_kzt_q75"), thresholds.get("betweenness_q95")
        rules = {
            "isolated": "Входящих и исходящих связей нет.",
            "boundary_consolidator": f"Не из исходного списка; глубина 4, выход 0; отправителей ≥3; вход ≥Q75 ({q75} KZT). Отсутствие выхода не означает удержание.",
            "boundary_unknown": "Глубина 4, исходящих связей 0; правило сбора средств на границе не выполнено.",
            "coordinator": f"Не из исходного списка; есть вход и выход; достижим от ≥2 исходных клиентов; посредничество ≥Q95 ({q95}).",
            "consolidator": f"Не из исходного списка; отправителей ≥3 и ≥2×max(получателей,1); вход ≥Q75 ({q75} KZT).",
            "distributor": "Получателей ≥5; для клиентов вне исходного списка также получателей ≥2×max(отправителей,1).",
            "terminal": "Не из исходного списка; входящая степень >0, исходящая =0, глубина <4.",
            "transit": "Не из исходного списка; есть вход и выход; отношение выхода к входу 0,7–1,3.",
            "fallback": "Предыдущие правила не выполнены; наблюдаемых признаков недостаточно.",
        }
        st.write(f"Правило: {rule}. {rules.get(rule, 'Правило отсутствует в описании.')}")
        st.caption("Правила проверяются по порядку: изолят → граница → связующий участник → сбор → распределение → конечный получатель → транзит → недостаточно признаков.")
        st.write(f"Сила признаков роли: {float(row.role_score) * 100:.1f} / 100. Это эвристическая оценка, не измеренная вероятность.")
        contribution_names = {
            "priority_volume": "Объём переводов (максимум 35)",
            "priority_bridge": "Посредничество в сети (максимум 30)",
            "priority_seed": "Связи с исходным списком (максимум 20)",
            "priority_degree": "Количество связей (максимум 15)",
        }
        st.dataframe(pd.DataFrame({"Фактор приоритета": list(contribution_names.values()),
                     "Вклад, баллы": [round(float(node_value(row, key, 0)) * 100, 3) for key in contribution_names]}),
                     hide_index=True, use_container_width=True)
        st.caption("Вклады складываются. У исходных клиентов объём и число связей оцениваются по исходящим; у остальных — с учётом входящих. Приоритет не умножается на оценку роли.")
        names = {"in_deg": "Разных отправителей", "out_deg": "Разных получателей",
                 "in_tx": "Входящих переводов", "out_tx": "Исходящих переводов",
                 "in_kzt": "Входящая сумма, KZT", "out_kzt": "Исходящая сумма, KZT",
                 "depth": "Глубина обхода", "is_seed": "В исходном списке",
                 "truncated_by_depth": "Обрыв на границе", "pagerank": "PageRank",
                 "betweenness": "Посредничество", "seed_reach_count": "Достижим от исходных клиентов",
                 "seed_distance": "Минимальное число шагов от исходного списка",
                 "pass_through": "Отношение исходящей суммы к входящей"}
        st.dataframe(pd.DataFrame([{"Показатель": label, "Поле": key,
                                    "Значение": str(node_value(row, key))} for key, label in names.items()]),
                     hide_index=True, use_container_width=True)


def main():
    st.set_page_config(page_title="Граф денег · очередь проверки", layout="wide")
    st.title("Граф денег")
    st.caption("Очередь проверки клиентов и наблюдаемые связи. Роли — гипотезы, приоритет — баллы для выбора следующей проверки, не вероятность виновности.")
    required = [OUT / name for name in ("nodes_roles.csv", "clusters.csv", "top_nodes.csv")]
    required.extend([DATA / "edges.parquet", OUT / "run_metadata.json"])
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        st.error("Нет файлов: " + ", ".join(missing) + ". Сначала выполните python run.py --data data --out out")
        st.stop()
    roles, clusters, top, edges, metadata = load_tables(tuple(_stamp(p) for p in required))
    if not roles.gid.is_unique:
        st.error("В nodes_roles.csv повторяются gid")
        st.stop()
    counters = st.columns(4)
    for col, title, value in zip(counters,
            ["Клиентов в выборке", "В исходном списке", "Групп клиентов", "Сумма переводов"],
            [f"{len(roles):,}".replace(",", " "), int(roles.is_seed.sum()), len(clusters), money(edges.sum_kzt.sum())]):
        col.metric(title, value)
    st.caption("Сумма переводов учитывает каждый перевод: одни и те же деньги могли пройти через несколько клиентов.")
    with st.sidebar:
        if st.toggle("❔ Как пользоваться", key="show_help", help="Краткая инструкция и значение основных терминов"):
            st.markdown("""
**С чего начать**

1. Посмотрите «Кого проверить первым» и выберите клиента из очереди.
2. Если известен номер клиента, вставьте его в поиск: он работает по всей выборке.
3. Откройте «Связи на схеме», чтобы увидеть направление переводов.
4. Во вкладке «Группа клиентов» покажите всю группу или выберите одного либо нескольких участников.

**Что означают слова**

- **Группа** — клиенты, объединённые наблюдаемыми связями; это гипотеза, не доказательство совместной деятельности.
- **Приоритет** — баллы для очереди проверки, не вероятность нарушения.
- **★** — клиент из исходного списка; **◇** — граница данных на четвёртом шаге.

Данные не охватывают другие банки, соседние периоды и переводы ниже 5 000 ₸.
""")
        st.divider()
        st.header("Область проверки")
        st.caption("Фильтры применяются ко всем клиентам. Исходный список — отправные точки расследования, обозначены ★.")
        role_filter = st.multiselect("Предполагаемая роль", sorted(roles.role.unique()),
                                     format_func=lambda value: ROLE_NAMES.get(value, value), key="role_filter")
        cluster_filter = st.multiselect("Группа клиентов", sorted(roles.cluster_id.unique()), key="cluster_filter")
        seed_filter = st.selectbox("В исходном списке", ["Все", "Да", "Нет"], key="seed_filter")
        st.divider()
        st.subheader("Выгрузки для проверки")
        for name, path in zip(("Все клиенты · CSV", "Группы · CSV", "Приоритетный список · CSV"), required[:3]):
            st.download_button(name, path.read_bytes(), file_name=path.name, mime="text/csv", use_container_width=True)
        st.caption(f"В официальном списке {len(top)} клиентов. Очередь на экране пересчитывается по выбранным фильтрам, баллы не меняются.")

    listing = filtered_nodes(roles, role_filter, cluster_filter, seed_filter)
    queue, detail = st.columns([1, 1.6], gap="large")
    with queue:
        st.subheader("Кого проверить первым")
        st.caption(f"Найдено {len(listing)} из {len(roles)} клиентов. В таблице первые 30 по приоритету.")
        options = listing.gid.astype(str).tolist()
        if options:
            lookup = listing.set_index("gid")
            selected = st.selectbox("Открыть клиента из очереди", options,
                format_func=lambda value: f"{value} · {float(lookup.loc[int(value), 'priority_score']) * 100:.1f} балла",
                key="client_select")
            st.dataframe(client_table(listing.head(30)), hide_index=True, use_container_width=True, height=320)
        else:
            selected = None
            st.info("По выбранным фильтрам клиентов нет. Измените фильтры или найдите номер вручную.")
        raw_gid = st.text_input("Поиск по полному номеру клиента (gid)", key="client_search",
                               placeholder="Вставьте полный номер", help="Поиск по всей выборке, независимо от фильтров. Очистите поле, чтобы вернуться к очереди.")
    with detail:
        chosen = raw_gid.strip() or selected
        if chosen is None:
            st.info("Выберите клиента для просмотра.")
            return
        try:
            gid = int(chosen)
        except ValueError:
            st.warning("Введите целочисленный номер клиента без пробелов и других символов.")
            return
        found = roles[roles.gid.eq(gid)]
        if found.empty:
            st.warning(f"Клиент {gid} отсутствует в данных. Поиск охватывает все {len(roles)} клиентов, включая изоляты.")
            return
        row = found.iloc[0]
        st.subheader(f"Клиент {gid}")
        st.write(f"**{ROLE_NAMES.get(row.role, row.role)}** · приоритет **{float(row.priority_score) * 100:.1f} / 100** · группа {int(row.cluster_id)}")
        st.caption("★ В исходном списке расследования" if bool(row.is_seed) else "Клиент вне исходного списка")
        st.markdown("**Почему обратить внимание**")
        st.write(review_reason(row))
        factors = {
            "priority_volume": "объём переводов",
            "priority_bridge": "положение между другими клиентами в сети",
            "priority_seed": "связи с исходными клиентами дела",
            "priority_degree": "число отправителей и получателей",
        }
        strongest = sorted(factors, key=lambda key: float(node_value(row, key, 0)), reverse=True)
        strongest = [key for key in strongest[:2] if float(node_value(row, key, 0)) > 0]
        if strongest:
            st.write("Больше всего на приоритет повлияли: " + "; ".join(
                f"{factors[key]} — {float(row[key]) * 100:.1f} балла" for key in strongest) + ".")
        flow = st.columns(4)
        for col, title, value in zip(flow, ["Получил в выборке", "Отправил в выборке", "Отправителей", "Получателей"],
                                   [money(row.in_kzt), money(row.out_kzt), int(row.in_deg), int(row.out_deg)]):
            col.metric(title, value)
        if bool(node_value(row, "truncated_by_depth", False)):
            st.warning("Граница данных: на четвёртом шаге обход остановлен. Отсутствие исходящих не означает, что деньги остались у клиента.")
            next_request = "Запросить дальнейшие исходящие переводы за четвёртым шагом и полную историю счёта."
        elif bool(row.is_seed):
            next_request = "Запросить входящие переводы за пределами выборки: вход этого клиента известен не полностью."
        else:
            next_request = "Сверить полную историю счёта и переводы за соседние периоды, чтобы проверить предполагаемую роль."
        st.write("**Следующий шаг:** " + next_request)
        st.caption("Не видны другие банки и периоды, переводы ниже 5 000 ₸, остатки. У любого клиента могут отсутствовать входы извне; совпадение сумм и достижимость не доказывают маршрут тех же денег.")

    network_tab, flows_tab, group_tab = st.tabs(["Связи на схеме", "Отправители и получатели", "Группа клиента"])
    with network_tab:
        controls = st.columns(2)
        color_by = controls[0].radio("Цвет узлов", ["По ролям", "По группам"], horizontal=True, key="graph_color")
        hops = controls[1].radio("Шагов от клиента", [1, 2], horizontal=True, key="graph_hops")
        visible, _ = neighborhood(edges, gid, hops)
        if color_by == "По ролям":
            legend = [(color, ROLE_NAMES[role]) for role, color in COLORS.items()]
        else:
            ids = sorted(roles.loc[roles.gid.isin(visible), "cluster_id"].unique())
            legend = [(cluster_color(int(group)), f"Группа {int(group)}") for group in ids]
        st.markdown(" · ".join(f'<span style="color:{color}">●</span> {escape(label)}' for color, label in legend), unsafe_allow_html=True)
        st.caption("Стрелки показывают направление перевода. ★ — исходный список, ◇ — граница данных. Подписи сокращены: полный номер и сумма доступны при наведении. Выбранный клиент выделен белой рамкой.")
        html, hidden = graph_html(edges, roles, gid, hops, color_by)
        components.html(html, height=530, scrolling=True)
        if hidden:
            st.info(f"Показано до 200 связанных узлов; скрыто {hidden}. Расчёты используют всю сеть.")
        if color_by == "По группам":
            st.caption("Цвет каждой группы постоянен; точный номер группы указан при наведении.")
    with flows_tab:
        for title, column, counterpart in [("От кого получил", "dst", "src"), ("Кому отправил", "src", "dst")]:
            transfers = edges.loc[edges[column].eq(gid)].sort_values("sum_kzt", ascending=False)
            st.markdown(f"**{title} · {len(transfers)} клиентов**")
            if transfers.empty:
                st.caption("В наблюдаемой выборке переводов нет.")
            else:
                peers = transfers[counterpart].map(roles.set_index("gid").role)
                st.dataframe(pd.DataFrame({"Клиент": transfers[counterpart].astype(str),
                    "Сумма за период, ₸": transfers.sum_kzt, "Число переводов": transfers.n_tx,
                    "Предполагаемая роль": peers.map(ROLE_NAMES)}), hide_index=True, use_container_width=True)
        st.caption("Одна строка объединяет все переводы между парой клиентов за период. Это не список отдельных операций.")
    with group_tab:
        summary = clusters.loc[clusters.cluster_id.eq(row.cluster_id)]
        if not summary.empty:
            group = summary.iloc[0]
            st.subheader(f"Группа {int(row.cluster_id)}")
            group_metrics = st.columns(3)
            group_metrics[0].metric("Клиентов в группе", int(group.n_nodes))
            group_metrics[1].metric("Из исходного списка", int(group.n_seed))
            group_metrics[2].metric("Переводы внутри группы", money(group.sum_kzt_internal))
            st.write("**Гипотеза по данным:**", group.hypothesis)
            all_members = group_members(roles, row.cluster_id)
            mode = st.radio("Кого показать в группе", ["Всю группу", "Выбрать клиентов"],
                            horizontal=True, key="group_mode",
                            help="Вся группа откроется одной кнопкой. В ручном выборе можно указать одного или нескольких клиентов по номеру.")
            if mode == "Всю группу":
                members = all_members
            else:
                chosen_ids = st.multiselect("Клиенты для просмотра", all_members.gid.astype(str).tolist(),
                    key=f"group_clients_{int(row.cluster_id)}", placeholder="Введите номер и выберите одного или нескольких",
                    help="Поиск принимает полный номер клиента. Ненужный выбор можно удалить крестиком.")
                members = group_members(roles, row.cluster_id, chosen_ids)
            st.caption(f"Выбрано {len(members)} из {len(all_members)} клиентов группы. "
                       "Таблица отсортирована по приоритету проверки.")
            if members.empty:
                st.info("Пока никто не выбран. Введите номер клиента или переключитесь на «Всю группу».")
            else:
                st.dataframe(client_table(members), hide_index=True, use_container_width=True)
                st.download_button("Скачать выбранных клиентов · CSV",
                    members.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"group_{int(row.cluster_id)}_selected.csv", mime="text/csv",
                    key="group_download")
            st.caption("Группа объединяет клиентов по связности переводов, не доказывает общую деятельность. Встречные переводы складываются только при поиске групп; на схеме их направление сохранено.")
    audit_panel(row, metadata)


if __name__ == "__main__":
    main()
