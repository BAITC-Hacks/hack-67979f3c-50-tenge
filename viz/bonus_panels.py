"""Read-only views of deterministic bonus reports produced by run.py."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def _read_report(path, stamp):
    del stamp
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load(out, name):
    path = Path(out) / name
    if not path.exists():
        return None
    return _read_report(str(path), path.stat().st_mtime_ns)


def _table(rows, labels=None):
    if not rows:
        st.caption("По этому критерию совпадений в выборке не найдено.")
        return
    frame = pd.DataFrame(rows)
    for column in frame:
        if column in {"gid", "src", "dst", "a", "b", "c"} or column.endswith("_gid"):
            frame[column] = frame[column].map(str)
        elif frame[column].map(lambda value: isinstance(value, (list, dict))).any():
            frame[column] = frame[column].map(
                lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value)
    st.dataframe(frame.rename(columns=labels or {}), hide_index=True, use_container_width=True)


DAY_LABELS = {"date": "Дата", "in_tx": "Входящих переводов", "out_tx": "Исходящих переводов",
    "in_kzt": "Получено, ₸", "out_kzt": "Отправлено, ₸", "activity_tx": "Всего операций",
    "activity_kzt": "Сумма операций, ₸", "n_senders": "Отправителей", "n_recipients": "Получателей",
    "incoming_date": "День поступления", "outgoing_date": "День отправления", "lag_days": "Разница, дней",
    "sender_gids": "Клиенты-отправители", "median_active_day_tx": "Медиана операций активного дня",
    "median_active_day_kzt": "Медиана суммы активного дня, ₸", "count_burst": "Всплеск количества",
    "amount_burst": "Всплеск суммы", "src": "Отправитель", "dst": "Получатель",
    "amount_kzt": "Повторяющаяся сумма, ₸", "n_tx": "Число переводов", "sum_kzt": "Общая сумма, ₸"}


def _render_temporal(report, gid):
    details = report.get("nodes", {}).get(gid)
    st.subheader(f"Активность клиента {gid}")
    if details is None:
        st.info("Клиент отсутствует в отчёте. Пересоздайте выгрузки.")
        return
    daily = details.get("daily", [])
    if daily:
        chart = pd.DataFrame(daily)[["date", "in_kzt", "out_kzt"]].copy()
        chart["date"] = pd.to_datetime(chart["date"])
        chart = chart.set_index("date").asfreq("D", fill_value=0)
        st.line_chart(chart.rename(columns={"in_kzt": "Получено, ₸", "out_kzt": "Отправлено, ₸"}))
        st.caption("Нули между активными днями означают отсутствие операций в этой выборке. Другие банки и суммы ниже порога не видны.")
    else:
        st.info("В предоставленных транзакциях операций этого клиента нет.")
    with st.expander("Дневные показатели"):
        _table(daily, DAY_LABELS)
    st.markdown("**Поступления и отправления через один–два дня**")
    st.caption("Это совпадения активности по календарным дням. Одна сумма может встречаться в нескольких строках; складывать их как объём транзита нельзя.")
    _table(details.get("lag_1_2_day_pairs", []), DAY_LABELS)
    same_day = details.get("same_day_flow_dates", [])
    if same_day:
        st.caption(f"Вход и выход в один день: {len(same_day)} дней. Порядок таких операций не установлен.")
    with st.expander("Синхронные поступления · от трёх отправителей за день"):
        _table(details.get("synchronous_incoming", []), DAY_LABELS)
        st.caption("Отправители считаются без самого клиента. Сумма и число входящих операций относятся ко всему дню, включая возможные самопереводы.")
    with st.expander("Всплески · не менее трёх операций и трёх медиан активного дня"):
        _table(details.get("bursts", []), DAY_LABELS)
    with st.expander("Повторяющиеся суммы · не менее трёх переводов в паре за день"):
        _table(details.get("equal_amount_groups", []), DAY_LABELS)
        st.caption("Совпадение сумм не доказывает дробление. Переводы ниже 5 000 ₸ не представлены.")
    anomaly = details.get("depth_anomaly", {})
    st.markdown("**Чем профиль отличается от клиентов того же колена**")
    if anomaly.get("status") == "insufficient_cohort":
        st.info("Недостаточно клиентов для сравнения: требуется минимум 20 в одном колене.")
    elif anomaly:
        labels = []
        if anomaly.get("high_volume"):
            labels.append("объём переводов")
        if anomaly.get("high_degree"):
            labels.append("число связей")
        if labels:
            st.warning("Процентиль не ниже 95 по признакам: " + ", ".join(labels) + ". Это повод изучить профиль, не вывод о нарушении.")
        else:
            st.caption("Порог 95-го процентиля по положительному объёму или числу связей не достигнут.")
        st.write(f"Колено: {anomaly.get('depth')}; клиентов для сравнения: {anomaly.get('cohort_size')}.")
        st.write(f"Процентиль объёма: {100 * anomaly.get('volume_percentile', 0):.1f}; связей: {100 * anomaly.get('degree_percentile', 0):.1f}. При равенствах используется средний ранг.")
        st.caption("У исходных клиентов сравниваются исходящие показатели. У остальных — наблюдаемые входы и выходы. Полнота выборки влияет на результат.")
    st.caption("Признаки служат гипотезами для проверки. Они не устанавливают происхождение или движение тех же денег.")


def _render_routes(report, gid):
    scope = st.radio("Какие маршруты показать", ["С выбранным клиентом", "По всей сети"], horizontal=True, key="bonus_route_scope")
    local = scope == "С выбранным клиентом"
    for key, title in (("recurring_routes", "Повторяющиеся маршруты из двух переводов"),
                       ("short_cycles", "Возвратные связи · циклы длины два–три")):
        block = report.get(key, {})
        items = block.get("items", [])
        chosen = [item for item in items if not local or gid in item.get("gids", [])]
        st.subheader(title)
        st.caption(f"Найдено во всей сети: {block.get('eligible_total', 0)}. Сохранено в отчёте: {len(items)}. В текущем просмотре: {len(chosen)}.")
        if block.get("truncated"):
            st.info("В отчёт вошли только первые маршруты по документированному рейтингу. Отсутствие клиента здесь не означает отсутствия маршрутов.")
        st.caption(block.get("caveat", ""))
        if not chosen:
            st.caption("В сохранённом наборе совпадений нет. Можно переключиться на всю сеть.")
            continue
        options = list(range(len(chosen)))
        index = st.selectbox("Маршрут для разбора" if key == "recurring_routes" else "Цикл для разбора",
            options, format_func=lambda i: " → ".join(chosen[i]["gids"] + ([chosen[i]["gids"][0]] if key == "short_cycles" else [])),
            key=f"bonus_{key}_select")
        item = chosen[index]
        st.write(item.get("why", ""))
        if key == "recurring_routes":
            st.caption(f"Дней с подходящим поступлением: {item['matched_incoming_days']}; пар дней: {item['matched_day_pairs']}. Переводов по первому ребру: {item['first_leg_n_tx']}, по второму: {item['second_leg_n_tx']}.")
            _table(item.get("evidence", []), {**DAY_LABELS, "incoming_n_tx": "Переводов на первом ребре",
                "incoming_sum_kzt": "Сумма первого ребра за день, ₸", "outgoing_n_tx": "Переводов на втором ребре",
                "outgoing_sum_kzt": "Сумма второго ребра за день, ₸"})
            if item.get("evidence_truncated"):
                st.caption(f"Показана часть примеров из {item['evidence_total']} пар дней.")
        else:
            _table(item.get("edges", []), DAY_LABELS)
        st.caption(block.get("method", ""))


def _render_resilience(report):
    block = report.get("resilience", {})
    baseline = block.get("baseline", {})
    scenarios = block.get("scenarios", [])
    st.subheader("Что изменится при исключении приоритетных клиентов из графа")
    st.caption("Математический сценарий для наблюдаемой сети. Он не является рекомендацией блокировки и не прогнозирует действия участников.")
    st.write(f"До исключения: {baseline.get('remaining_node_count', '—')} клиентов; крупнейшая связная часть — {baseline.get('largest_component_nodes', '—')}; компонент — {baseline.get('weak_component_count', '—')}.")
    rows = [{"Исключено клиентов": s["removed_node_count"], "Осталось клиентов": s["remaining_node_count"],
        "Крупнейшая часть после": s["largest_component_nodes"],
        "При случайном исключении, среднее": round(s["random_baseline"]["largest_component_nodes_mean"], 1),
        "Случайное: минимум": s["random_baseline"]["largest_component_nodes_min"],
        "Случайное: максимум": s["random_baseline"]["largest_component_nodes_max"],
        "Фрагментов исходной крупнейшей части": s["original_largest_fragment_count"],
        "Сохранилось суммы на рёбрах, %": round(s["retained_edge_weight_fraction"] * 100, 2)} for s in scenarios]
    _table(rows)
    if scenarios:
        chosen = st.selectbox("Подробности сценария", list(range(len(scenarios))),
            format_func=lambda i: f"Исключение {scenarios[i]['removed_node_count']} клиентов", key="bonus_removal")
        s = scenarios[chosen]
        st.write("Исключённые клиенты: " + ", ".join(s["removed_gids"]))
        st.write(f"Из исходной крупнейшей части удалено {s['original_largest_removed_nodes']} клиентов. Среди оставшихся {s['original_largest_nodes_outside_largest_fragment']} оказались вне её крупнейшего остаточного фрагмента.")
        st.caption(f"Сравнение с {s['random_baseline']['runs']} случайными наборами того же размера. Это ограниченная выборка сценариев, не доверительный интервал.")
    st.caption(block.get("caveat", ""))


def render_bonus_panels(out, gid):
    try:
        temporal = _load(out, "temporal_patterns.json")
        network = _load(out, "network_patterns.json")
    except (OSError, ValueError):
        st.error("Не удалось прочитать дополнительный анализ. Пересоздайте результаты пайплайна.")
        return
    if temporal is None or network is None:
        st.info("Для дополнительного анализа выполните python run.py --data data --out out и обновите страницу.")
        return
    times, routes, resilience = st.tabs(["Активность клиента", "Маршруты и циклы", "Уязвимость сети"])
    with times:
        _render_temporal(temporal, str(gid))
    with routes:
        _render_routes(network, str(gid))
    with resilience:
        _render_resilience(network)
    st.download_button("Скачать результаты дополнительного анализа · JSON",
        json.dumps({"temporal": temporal, "network": network}, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="additional_analysis.json", mime="application/json", key="bonus_download")
