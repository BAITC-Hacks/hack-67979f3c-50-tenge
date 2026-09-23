"""Readable direct flows; complete amounts and exact client IDs."""

import streamlit as st


def direct_flows(edges, gid, limit=5):
    incoming = edges.loc[edges.dst.eq(gid) & edges.src.ne(gid)].sort_values(
        ["sum_kzt", "src"], ascending=[False, True])
    outgoing = edges.loc[edges.src.eq(gid) & edges.dst.ne(gid)].sort_values(
        ["sum_kzt", "dst"], ascending=[False, True])
    return [(frame if limit is None else frame.head(limit), len(frame))
            for frame in (incoming, outgoing)]


def _amount(value):
    return f"{float(value):,.2f}".replace(",", " ") + " ₸"


def _open_client(gid):
    st.session_state["client_search"] = str(gid)


def render_flow_overview(edges, roles, row, role_names):
    gid = int(row.gid)
    st.subheader("От кого получил → выбранный клиент → кому отправил")
    st.caption("Начните с крупнейших переводов. Каждая карточка — сумма всех переводов между парой клиентов за период. "
               "Левая и правая стороны не устанавливают путь одних и тех же денег.")
    show_all = st.toggle("Показать все прямые связи", key=f"flow_all_{gid}")
    flows = direct_flows(edges, gid, None if show_all else 5)
    shown = sum(len(frame) for frame, _ in flows)
    total = sum(count for _, count in flows)
    st.caption(f"Показано {shown} из {total} направленных связей с другими клиентами; скрыто {total - shown}. "
               + ("По пять крупнейших с каждой стороны." if not show_all else "Показаны все прямые связи."))
    lookup = roles.set_index("gid")
    left, center, right = st.columns([1.2, 1, 1.2], gap="medium")
    with center:
        st.markdown("### Выбранный клиент")
        with st.container(border=True):
            st.code(str(gid), language=None)
            st.write(role_names.get(row.role, row.role))
            st.metric("Всего получил в выборке", _amount(row.in_kzt))
            st.metric("Всего отправил в выборке", _amount(row.out_kzt))
            if bool(row.truncated_by_depth):
                st.warning("Граница обхода: дальнейшие переводы неизвестны.")
            st.caption("Суммы не являются остатком на счёте.")
    for panel, (frame, count), side, peer_column, heading in (
        (left, flows[0], "in", "src", "Поступления →"),
        (right, flows[1], "out", "dst", "→ Отправления"),
    ):
        with panel:
            st.markdown(f"### {heading}")
            st.caption(f"Контрагентов: {count}")
            if frame.empty:
                st.info("В выборке нет переводов с другими клиентами в этом направлении.")
            for edge in frame.to_dict("records"):
                peer = int(edge[peer_column])
                with st.container(border=True):
                    st.code(str(peer), language=None)
                    st.markdown(f"**{_amount(edge['sum_kzt'])}** · переводов: {int(edge['n_tx'])}")
                    if peer in lookup.index:
                        st.caption(role_names.get(lookup.loc[peer, "role"], lookup.loc[peer, "role"]))
                    st.button("Открыть клиента", key=f"flow_{gid}_{side}_{peer}",
                              on_click=_open_client, args=(peer,))
    self_edges = edges.loc[edges.src.eq(gid) & edges.dst.eq(gid)]
    if not self_edges.empty:
        st.info(f"Отдельно: переводы самому себе — {_amount(self_edges.sum_kzt.sum())}, "
                f"операций: {int(self_edges.n_tx.sum())}. Они входят в общие суммы клиента, "
                "но не показаны как связи с другими клиентами.")
