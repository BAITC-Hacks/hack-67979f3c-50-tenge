"""Explicit opt-in smoke check: two questions, real API credits, never prints keys."""
import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true', help='Authorize two real questions using the configured API key')
    args = parser.parse_args()
    if not args.live:
        parser.error('Pass --live to authorize paid API calls')
    import pandas as pd
    from viz.ai_transport import read_config, request_explanation, error_message
    from viz.ai_assistant import _context, _facts, _limitations, INSTRUCTIONS
    from viz.graph_agent import GraphQueries, _run_agent
    config = read_config()
    key = config.get('OPENAI_API_KEY')
    if not key:
        raise SystemExit('OPENAI_API_KEY is not configured')
    model = config.get('OPENAI_MODEL') or 'gpt-4.1-mini'
    roles = pd.read_csv(ROOT / 'out/nodes_roles.csv')
    edges = pd.read_parquet(ROOT / 'data/edges.parquet')
    gid = int(roles.sort_values(['priority_score','gid'],ascending=[False,True]).gid.iloc[0])
    context = _context(roles, edges, gid)
    context['facts'] = _facts(context)
    context['limitations'] = _limitations(context)
    question = 'Почему стоит проверить этого клиента и какие сведения запросить?'
    try:
        explanation = request_explanation(key, model, question, context, INSTRUCTIONS)
        print(json.dumps({'model': model, 'explanation': explanation}, ensure_ascii=False), flush=True)
        # Select a real node with at least two incoming senders, without hardcoded IDs.
        counts = edges.groupby('dst').src.nunique()
        target = int(counts[counts >= 2].sort_values(ascending=False).index[0])
        sources = sorted(map(int, edges.loc[edges.dst.eq(target), 'src'].unique()))[:2]
        q = 'Найди общих прямых получателей этих клиентов: ' + ', '.join(map(str, sources))
        result = _run_agent(key, model, q, gid, GraphQueries(roles, edges))
        print(json.dumps({'question': q, 'agent': result}, ensure_ascii=False), flush=True)
        assert any(t['tool'] == 'common_recipients' for t in result['trace']), 'Expected graph tool was not called'
    except HTTPError as exc:
        raise SystemExit(error_message(exc.code)) from None
    except Exception as exc:
        # Never emit provider bodies or request headers.
        raise SystemExit('Live check failed: ' + type(exc).__name__) from None


if __name__ == '__main__':
    main()
