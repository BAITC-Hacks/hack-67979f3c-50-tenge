# Статус потока A

Реализовано и проверено локально 23 сентября 2026. **Модули A готовы к
подключению B/C; полный командный пайплайн пока не проверен.**

## Готово

- `pipeline/io.py`: `load_data(data_dir) → (nodes, edges, transactions)`,
  `validate_inputs(nodes, edges, transactions) → None`.
- `pipeline/graph.py`: `build_graph(nodes, edges) → nx.DiGraph`, включая изоляты.
- `pipeline/features.py`: `compute_features(G, nodes) → DataFrame`.
- `pipeline/roles.py`: `assign_roles(features_df) → DataFrame`, правила и
  формулы точно по Plan.md §5; helper `compute_role_thresholds(features_df)`.
- `tests/test_io.py`, `tests/test_roles.py`; объяснения для команды в `docs/roles.md`.

Plan.md и файлы других потоков не менялись. SHA-256 плана до и после работы:
`d10b435cbea04f96ac28702dcf92e643a5a36534b6f1720a2059f298db412a16`.

## Реально выполненные проверки

В окружении исследования вне репозитория:

```bash
/tmp/50tenge-plan-env/bin/python -m pytest tests/test_io.py tests/test_roles.py -q -s
```

Результат: **63 passed in 6.03s**, без предупреждений. В него входят два
расчёта A на настоящих parquet с точным сравнением результата при
перемешанных входных строках; они заняли суммарно около 4.79 секунды.
Отдельный замер загрузки, проверки, сборки графа и признаков — около 2.19 секунды.
Это замеры текущего окружения, не обещание времени на любом ноутбуке.

Окружение: Python 3.12.3; pandas 3.0.6; NumPy 2.5.3; NetworkX 3.7;
PyArrow 25.0.1; SciPy 1.18.1; pytest 9.1.1. Зависимости уже перечислены
в общем requirements.txt; менять или фиксировать версии должен C.
Путь `/tmp/50tenge-plan-env` не требуется для сдачи: после установки зависимостей
те же тесты запускаются обычным `python -m pytest ...`.

Подтверждено: 2 248 узлов, 3 119 рёбер, 4 840 транзакций, 35 компонент,
19 изолятов, согласованность сумм и количеств. Пороги: Q75=165000 KZT,
Q95 betweenness=0.0013361804007553844. Все 444 узла обрыва не являются terminal:
1 boundary_consolidator и 443 boundary_unknown.

## Передача B/C

1. B возвращает `gid, seed_reach_count, seed_distance` из `compute_seed_reach`.
   C объединяет с признаками A по gid через `how='left', validate='one_to_one'`
   **до** вызова `assign_roles`. Пропуски reach-count вызывают ошибку.
2. Обязательные признаки A сохранены. Дополнительно возвращаются
   необязательные `in_avg_kzt`, `out_avg_kzt`: средняя сумма перевода, NaN
   при нулевом количестве. Их можно показывать/экспортировать по усмотрению B/C;
   обязательный контракт не расширяется.
3. `roles.attrs['role_thresholds']` содержит точные вычисленные пороги;
   C сохраняет их в метаданных запуска. `compute_role_thresholds` также
   доступен напрямую. `attrs` не является дополнительной колонкой CSV.
4. Список `role_rule`: isolated, boundary_consolidator, boundary_unknown,
   coordinator, consolidator, distributor, terminal, transit, fallback.
   C использует его и `truncated_by_depth` для карточки ограничений.
5. `role_candidate_counts` в attrs — диагностика пересечений до выбора
   основной роли; `role_rule` — фактически сработавшее правило.

## Что остаётся после интеграции

- В рабочей копии во время проверки ещё нет seeds/clusters/priority/outputs
  потока B и run.py потока C. Seed-колонки в интеграционном тесте A создаются
  независимым обратным BFS **только в тесте**. Результаты ролей сверить после
  подключения настоящего B, в том числе для координаторов.
- Три CSV, полный запуск менее 5 минут и интерфейс пока не проверены;
  тестовый результат A не выдаётся за готовую сдачу.
- После появления B прочитать формулу приоритета и передать замечания
  владельцу, не менять его файлы самостоятельно.
- C добавляет в README краткие правила и ссылку на `docs/roles.md`.
