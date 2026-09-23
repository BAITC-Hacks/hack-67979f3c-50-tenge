# hack-67979f3c-50-tenge
Hackathon team repository for 50 Tenge

## Запуск через Docker Compose

Нужен Docker с плагином Compose и запущенным движком Linux-контейнеров.
Из корня репозитория, где в `data/` лежат `nodes.parquet`,
`edges.parquet` и `transactions.parquet`, выполните:

```bash
docker compose run --build --rm pipeline
```

Контейнер выполнит `starter/starter.py` и завершится. Результаты сохранятся
на компьютере в `out/nodes_roles.csv`, `out/clusters.csv` и `out/top_nodes.csv`.
Повторный запуск перезаписывает эти файлы. Входные данные подключены только
для чтения и не входят в образ.

Сейчас запускается стартовый код организаторов: роли ещё не заполнены,
а `clusters.csv` и `top_nodes.csv` содержат только заголовки.
