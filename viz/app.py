"""Local analyst dashboard over completed CSV exports."""

import json
import colorsys
import sys
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from viz.ai_assistant import render_assistant
from viz.cluster_quality import render_cluster_quality
from viz.bonus_panels import render_bonus_panels
from viz.graph_agent import render_graph_agent
from pipeline.roles import role_candidates

OUT = ROOT / "out"
DATA = ROOT / "data"
COLORS = {
    "consolidator": "#fbbf24", "transit": "#60a5fa",
    "distributor": "#a78bfa", "terminal": "#2dd4bf",
    "coordinator": "#fb7185", "peripheral": "#94a3b8",
}
ROLE_NAMES = {
    "consolidator": "Признаки сбора средств", "transit": "Передаёт средства дальше",
    "distributor": "Распределяет средства", "terminal": "Нет видимых переводов дальше",
    "coordinator": "Связующий участник", "peripheral": "Роль не определена",
}


def apply_design():
    """Local visual styles: stable Streamlit containers and our own classes."""
    st.markdown("""<style>
    [data-testid="stAppViewContainer"] {background:#0b1120;color:#e5edf8;}
    [data-testid="stHeader"] {background:rgba(11,17,32,.94);}
    [data-testid="stSidebar"] {background:#10192a;border-right:1px solid #243149;}
    [data-testid="stMainBlockContainer"] {max-width:1540px;padding-top:2rem;padding-bottom:2rem;}
    [data-testid="stMetric"] {background:#141e30;border:1px solid #26344d;border-radius:14px;padding:16px;}
    [data-testid="stMetricValue"] {font-variant-numeric:tabular-nums;font-size:1.45rem;}
    [data-testid="stDataFrame"] {border:1px solid #26344d;border-radius:12px;overflow:hidden;}
    [data-testid="stExpander"] {border-color:#26344d;border-radius:12px;}
    [data-testid="stTabs"] [role="tablist"] {gap:1rem;border-bottom:1px solid #26344d;}
    [data-testid="stTabs"] [role="tab"] {padding:12px 5px;font-weight:550;}
    [data-testid="stTabs"] [aria-selected="true"] {color:#93c5fd;}
    .money-header {display:flex;align-items:center;gap:14px;margin:0 0 8px;}
    .money-mark {display:grid;place-items:center;width:46px;height:46px;border-radius:14px;background:#1e3453;color:#93c5fd;font-size:26px;flex-shrink:0;}
    .money-title {font-size:1.85rem;line-height:1.15;letter-spacing:-.04em;font-weight:750;color:#f1f5f9;}
    .money-eyebrow {color:#94a3b8;font-size:.76rem;letter-spacing:.09em;text-transform:uppercase;margin-bottom:5px;}
    .money-subtitle {color:#a9b8cc;font-size:.92rem;line-height:1.55;margin:0 0 18px;max-width:940px;}
    .money-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:8px 0 4px;}
    .money-grid.flow {grid-template-columns:repeat(2,minmax(0,1fr));margin:16px 0;}
    .money-stat {background:linear-gradient(135deg,#172339,#121c2d);border:1px solid #293750;border-radius:14px;padding:15px 18px;min-width:0;}
    .money-stat-label {color:#b0bfd2;font-size:.8rem;margin-bottom:7px;}
    .money-stat-value {color:#f1f5f9;font-size:clamp(1.08rem,1.65vw,1.65rem);line-height:1.3;font-weight:650;font-variant-numeric:tabular-nums;overflow-wrap:anywhere;}
    .money-client {display:flex;justify-content:space-between;align-items:flex-start;gap:14px;margin:8px 0 12px;}
    .money-id {font-size:clamp(1.05rem,1.7vw,1.4rem);font-weight:650;font-variant-numeric:tabular-nums;color:#f1f5f9;overflow-wrap:anywhere;}
    .money-score {text-align:right;min-width:90px;color:#93c5fd;font-size:1.8rem;font-weight:750;line-height:1.2;}
    .money-score small {font-size:.78rem;color:#a9b8cc;font-weight:400;display:block;}
    .money-badges {display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;}
    .money-badge {font-size:.76rem;background:#19273d;border:1px solid #30415e;border-radius:24px;padding:5px 10px;color:#c9d8eb;}
    .money-next {border-left:3px solid #60a5fa;background:#14233a;border-radius:0 10px 10px 0;padding:13px 16px;font-size:.91rem;line-height:1.6;margin:12px 0;}
    .money-next strong {color:#bfdbfe;display:block;font-size:.78rem;margin-bottom:4px;}
    .money-legend {display:flex;flex-wrap:wrap;gap:8px 16px;margin:12px 0;font-size:.8rem;color:#c5d1e3;}
    .money-legend-item {display:inline-flex;align-items:center;gap:7px;}
    .money-dot {width:9px;height:9px;border-radius:50%;display:inline-block;flex-shrink:0;}
    @media(max-width:760px){.money-grid{grid-template-columns:repeat(2,minmax(0,1fr));}.money-title{font-size:1.5rem;}.money-stat{padding:12px;}}
    </style>""", unsafe_allow_html=True)


def stat_cards(items, *, flow=False):
    cards = "".join(
        '<div class="money-stat"><div class="money-stat-label">' + escape(str(label))
        + '</div><div class="money-stat-value">' + escape(str(value)) + '</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="money-grid{" flow" if flow else ""}">{cards}</div>', unsafe_allow_html=True)


def reset_filters():
    st.session_state["role_filter"] = []
    st.session_state["cluster_filter"] = []
    st.session_state["seed_filter"] = "Все"


def clear_search():
    st.session_state["client_search"] = ""


def open_collectors():
    reset_filters()
    clear_search()
    st.session_state["role_filter"] = ["consolidator"]
    st.session_state.pop("client_select", None)


def secondary_signals(row, metadata):
    thresholds = metadata.get("role_thresholds", {})
    if not {"in_kzt_q75", "betweenness_q95"}.issubset(thresholds):
        return []
    candidates = role_candidates(pd.DataFrame([row.to_dict()]), thresholds)
    return [name for name in ("coordinator", "consolidator", "distributor", "terminal", "transit")
            if name != row.role and bool(candidates[name].iloc[0])]


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
                    bgcolor="#0f192b", font_color="#e5edf8", cdn_resources="in_line")
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
    diagnostic = metadata.get("priority_sensitivity")
    if diagnostic:
        with st.expander("Насколько приоритет зависит от выбранных весов"):
            st.write(diagnostic["method"])
            detail = diagnostic["nodes"].get(str(int(row.gid)))
            if detail:
                st.write(f"Входит в официальный top-30 в {detail['top_hits']} из {detail['scenarios']} сценариев. "
                         f"Место в общей очереди всех клиентов: {detail['rank_min']}–{detail['rank_max']}.")
            labels = {"priority_volume": "Объём", "priority_bridge": "Посредничество",
                      "priority_seed": "Связи с исходным списком", "priority_degree": "Количество связей"}
            st.dataframe(pd.DataFrame([{
                "Изменённый вес": labels[s["factor"]], "Множитель": s["multiplier"],
                "Сохранилось из исходного top": s["retained"], "Размер исходного top": s["baseline_size"],
                "Отбор": "Только вне исходного списка" if s["selection_policy"] == "non_seed" else "Все клиенты",
            } for s in diagnostic["scenarios"]]), hide_index=True)
            st.caption(diagnostic["caveat"])
    with st.expander("Правило, пороги и вклад каждого признака", expanded=True):
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
    apply_design()
    st.markdown('<div class="money-header"><div class="money-mark">↗</div><div>'
                '<div class="money-eyebrow">50 Tenge · анализ переводов</div>'
                '<div class="money-title">Граф денег</div></div></div>'
                '<p class="money-subtitle">Выберите клиента, разберите его связи и определите следующий шаг проверки. '
                'Роли — гипотезы; приоритет помогает упорядочить работу и не означает вероятность нарушения.</p>',
                unsafe_allow_html=True)
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
    stat_cards([
        ("Клиентов в выборке", f"{len(roles):,}".replace(",", " ")),
        ("В исходном списке", int(roles.is_seed.sum())),
        ("Групп клиентов", len(clusters)),
        ("Сумма переводов", money(edges.sum_kzt.sum())),
    ])
    st.caption("Сумма переводов учитывает каждый перевод: одни и те же деньги могли пройти через несколько клиентов.")
    collectors = roles.loc[roles.role.eq("consolidator")]
    with st.container(border=True):
        st.markdown(f"**Точки сбора средств · {len(collectors)} клиентов**")
        st.caption("Отдельная очередь помогает увидеть сборщиков, даже если они не вошли в общий top-30. "
                   "Внутри очереди действует основной приоритет. Сбор поступлений не доказывает накопление денег на счёте.")
        st.button("Открыть очередь сборщиков", key="open_collectors", on_click=open_collectors,
                  disabled=collectors.empty)
    with st.sidebar:
        if st.toggle("❔ Как пользоваться", key="show_help", help="Краткая инструкция и значение основных терминов"):
            st.markdown("""
**С чего начать**

1. Посмотрите «Очередь проверки» и выберите клиента из списка.
2. Если известен номер клиента, вставьте его в поиск: он работает по всей выборке.
3. Откройте «Схема связей», чтобы увидеть направление переводов.
4. Во вкладке «Группа клиента» покажите всю группу или выберите одного либо нескольких участников.
5. «Помощник» отвечает на вопросы по данным, а «Расчёт и основания» показывает правила и числа.

**Что означают слова**

- **Группа** — клиенты, объединённые наблюдаемыми связями; это гипотеза, не доказательство совместной деятельности.
- **Приоритет** — баллы для очереди проверки, не вероятность нарушения.
- **★** — клиент из исходного списка; **◇** — граница данных на четвёртом шаге.

Данные не охватывают другие банки, соседние периоды и переводы ниже 5 000 ₸.
""")
        st.divider()
        st.subheader("Фильтры очереди")
        st.caption("Применяются ко всей выборке. Поиск по номеру работает независимо от фильтров.")
        role_filter = st.multiselect("Предполагаемая роль", sorted(roles.role.unique()),
                                     format_func=lambda value: ROLE_NAMES.get(value, value), key="role_filter",
                                     placeholder="Все роли")
        cluster_filter = st.multiselect("Группа клиентов", sorted(roles.cluster_id.unique()), key="cluster_filter",
                                       placeholder="Все группы", format_func=lambda value: f"Группа {value}")
        seed_filter = st.selectbox("В исходном списке", ["Все", "Да", "Нет"], key="seed_filter",
                                  help="Исходный список — известные отправные точки расследования, отмечены ★.")
        if role_filter or cluster_filter or seed_filter != "Все":
            st.button("Сбросить фильтры", on_click=reset_filters, key="reset_filters", use_container_width=True)
        st.divider()
        with st.expander("Скачать результаты · CSV"):
            for name, path in zip(("Все клиенты", "Все группы", "Приоритетный список"), required[:3]):
                st.download_button(name, path.read_bytes(), file_name=path.name, mime="text/csv", use_container_width=True)
            st.caption(f"Приоритетный список содержит {len(top)} клиентов. Выгрузки полные и не зависят от фильтров экрана.")

    listing = filtered_nodes(roles, role_filter, cluster_filter, seed_filter)
    queue, detail = st.columns([1, 1.6], gap="large")
    with queue:
        st.subheader("Очередь проверки")
        raw_gid = st.text_input("Поиск по полному номеру клиента", key="client_search",
                               placeholder="Введите или вставьте полный номер",
                               help="Поиск охватывает всех клиентов, включая изоляты. Очистите поле, чтобы вернуться к очереди.")
        if raw_gid.strip():
            st.button("Вернуться к очереди", on_click=clear_search, key="clear_search", use_container_width=True)
            st.caption("Открыт результат точного поиска. Фильтры очереди на него не влияют.")
        st.caption(f"В очереди {len(listing)} из {len(roles)} клиентов · сначала наибольший приоритет")
        options = listing.gid.astype(str).tolist()
        if options:
            lookup = listing.set_index("gid")
            selected = st.selectbox("Клиент для проверки", options,
                format_func=lambda value: f"{value} · {float(lookup.loc[int(value), 'priority_score']) * 100:.1f} балла",
                key="client_select", disabled=bool(raw_gid.strip()))
            st.dataframe(client_table(listing.head(30)), hide_index=True, use_container_width=True, height=290,
                column_config={"Клиент": st.column_config.TextColumn("Клиент", width="medium"),
                               "Приоритет / 100": st.column_config.ProgressColumn("Баллы", min_value=0, max_value=100, format="%.1f"),
                               "В исходном списке": None})
            st.caption("В таблице первые 30. Остальные доступны в списке выбора выше.")
        else:
            selected = None
            st.info("По выбранным фильтрам клиентов нет. Измените фильтры или найдите номер вручную.")
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
        st.markdown('<div class="money-client"><div><div class="money-eyebrow">Карточка клиента</div>'
                    f'<div class="money-id">{gid}</div></div><div class="money-score">'
                    f'{float(row.priority_score) * 100:.1f}<small>приоритет / 100</small></div></div>', unsafe_allow_html=True)
        tags = [ROLE_NAMES.get(row.role, row.role), f"Группа {int(row.cluster_id)}",
                "★ Исходный список" if bool(row.is_seed) else "Вне исходного списка"]
        if bool(node_value(row, "truncated_by_depth", False)):
            tags.append("◇ Граница данных")
        st.markdown('<div class="money-badges">' + ''.join(
            f'<span class="money-badge">{escape(str(tag))}</span>' for tag in tags) + '</div>', unsafe_allow_html=True)
        st.markdown("**Почему обратить внимание**")
        st.write(review_reason(row))
        additional = secondary_signals(row, metadata)
        if additional:
            st.markdown("**Дополнительные признаки по правилам**")
            st.write(" · ".join(ROLE_NAMES[name] for name in additional))
            st.caption("Одновременно выполнены несколько правил. Основная роль выбрана по документированному порядку; "
                       "дополнительные признаки не являются независимым подтверждением подозрительности.")
        if row.role == "consolidator" or "consolidator" in additional:
            st.caption("Наблюдается сбор переводов от нескольких отправителей. Остаток на счёте и удержание денег по этим данным не установлены.")
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
        stat_cards([("Получил в выборке", money(row.in_kzt)), ("Отправил в выборке", money(row.out_kzt)),
                    ("Разных отправителей", int(row.in_deg)), ("Разных получателей", int(row.out_deg))], flow=True)
        if bool(node_value(row, "truncated_by_depth", False)):
            st.warning("Граница данных: на четвёртом шаге обход остановлен. Отсутствие исходящих не означает, что деньги остались у клиента.")
            next_request = "Запросить дальнейшие исходящие переводы за четвёртым шагом и полную историю счёта."
        elif bool(row.is_seed):
            next_request = "Запросить входящие переводы за пределами выборки: вход этого клиента известен не полностью."
        else:
            next_request = "Сверить полную историю счёта и переводы за соседние периоды, чтобы проверить предполагаемую роль."
        st.markdown('<div class="money-next"><strong>СЛЕДУЮЩИЙ ШАГ ПРОВЕРКИ</strong>'
                    + escape(next_request) + '</div>', unsafe_allow_html=True)
        st.caption("Не видны другие банки и периоды, переводы ниже 5 000 ₸, остатки. У любого клиента могут отсутствовать входы извне; совпадение сумм и достижимость не доказывают маршрут тех же денег.")

    network_tab, flows_tab, group_tab, bonus_tab, assistant_tab, audit_tab = st.tabs(
        ["Схема связей", "Переводы", "Группа клиента", "Дополнительный анализ", "Помощник", "Расчёт и основания"])
    with network_tab:
        controls = st.columns(2)
        color_by = controls[0].radio("Раскраска схемы", ["По ролям", "По группам"], horizontal=True, key="graph_color")
        hops = controls[1].radio("Показать связи", [1, 2], horizontal=True, key="graph_hops",
                                format_func=lambda value: "Прямые" if value == 1 else "До двух шагов")
        visible, _ = neighborhood(edges, gid, hops)
        if color_by == "По ролям":
            legend = [(color, ROLE_NAMES[role]) for role, color in COLORS.items()]
        else:
            ids = sorted(roles.loc[roles.gid.isin(visible), "cluster_id"].unique())
            legend = [(cluster_color(int(group)), f"Группа {int(group)}") for group in ids]
        st.markdown('<div class="money-legend">' + ''.join(
            f'<span class="money-legend-item"><span class="money-dot" style="background:{color}"></span>{escape(label)}</span>'
            for color, label in legend) + '</div>', unsafe_allow_html=True)
        st.caption("Стрелка — направление денег · ★ исходный список · ◇ граница данных · белая рамка — выбранный клиент")
        if int(row.in_deg) + int(row.out_deg) == 0:
            st.info("У клиента нет наблюдаемых переводов. На схеме показан только он; это не означает отсутствие связей вне выборки.")
        html, hidden = graph_html(edges, roles, gid, hops, color_by)
        components.html(html, height=530, scrolling=True)
        st.caption("Перетаскивайте узлы и меняйте масштаб. Наведите на узел для полного номера, на стрелку — для суммы переводов.")
        if hidden:
            st.info(f"Показано до 200 связанных узлов; скрыто {hidden}. Расчёты используют всю сеть.")
        if color_by == "По группам":
            st.caption("Цвет каждой группы постоянен; точный номер группы указан при наведении.")
    with flows_tab:
        st.caption("Все наблюдаемые переводы выбранного клиента за период, начиная с наибольшей суммы.")
        for title, column, counterpart in [("От кого получил", "dst", "src"), ("Кому отправил", "src", "dst")]:
            transfers = edges.loc[edges[column].eq(gid)].sort_values("sum_kzt", ascending=False)
            st.markdown(f"**{title} · {len(transfers)} клиентов**")
            if transfers.empty:
                st.caption("В наблюдаемой выборке переводов нет.")
            else:
                peers = transfers[counterpart].map(roles.set_index("gid").role)
                st.dataframe(pd.DataFrame({"Клиент": transfers[counterpart].astype(str),
                    "Сумма за период, ₸": transfers.sum_kzt, "Число переводов": transfers.n_tx,
                    "Предполагаемая роль": peers.map(ROLE_NAMES)}), hide_index=True, use_container_width=True,
                    column_config={"Сумма за период, ₸": st.column_config.NumberColumn(format="localized"),
                                   "Клиент": st.column_config.TextColumn(width="medium")})
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
            render_cluster_quality(roles, edges, int(row.cluster_id), summary=group)
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
    with assistant_tab:
        explain_tab, graph_query_tab = st.tabs(["Объяснить клиента", "Задать вопрос по сети"])
        with explain_tab:
            render_assistant(roles, edges, gid)
        with graph_query_tab:
            render_graph_agent(roles, edges, gid)
    with bonus_tab:
        render_bonus_panels(OUT, gid)
    with audit_tab:
        st.caption("Проверьте гипотезу по исходным числам. Оценки рассчитаны правилами; ниже сохранены условия, пороги и составляющие приоритета.")
        audit_panel(row, metadata)


if __name__ == "__main__":
    main()
