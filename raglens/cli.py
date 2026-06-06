"""RAGLens command-line interface (FR1.8).

Commands:
  raglens run <script.py>        Execute a Python pipeline script (with raglens importable).
  raglens explain <traces.jsonl> Show counterfactual attribution stored in traces.
  raglens insights <traces.jsonl> Print auto-generated observability insights.
  raglens doc <traces.jsonl>     Render a RAG datasheet (Markdown/HTML).
  raglens demo                   Run the bundled example end-to-end.
  raglens version                Print version.

Note on `explain`: counterfactual attribution is *causal* — it re-runs your
generator with chunks removed — so it must be computed in-process while the
pipeline is live (call ``cap.attribute()`` in your script). The CLI `explain`
command displays attributions already stored in a trace file; it does not (and
cannot) re-generate from a static JSONL alone.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from typing import List, Optional

from . import __version__
from .datasheet import write_datasheet
from .insights import aggregate_metrics, generate_fleet_insights, generate_insights
from .writer import read_traces


def _cmd_run(args: argparse.Namespace) -> int:
    sys.argv = [args.script] + (args.script_args or [])
    runpy.run_path(args.script, run_name="__main__")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    traces = read_traces(args.traces)
    if args.query_id:
        traces = [t for t in traces if t.query_id.startswith(args.query_id)]
    if not traces:
        print("No matching traces.", file=sys.stderr)
        return 1
    any_attr = False
    for t in traces:
        print(f"\n=== Query {t.query_id[:8]} ===")
        print(f"Q: {t.query}")
        if t.answer:
            ans = t.answer if len(t.answer) <= 300 else t.answer[:300] + "…"
            print(f"A: {ans}")
        if not t.attributions:
            print("  (no attributions — compute with cap.attribute() in your script)")
            continue
        any_attr = True
        by_id = {c.id: c for c in t.final_chunks()}
        print(f"  {'chunk':<22}{'importance':<12}preview")
        for a in sorted(t.attributions, key=lambda x: x.score, reverse=True):
            c = by_id.get(a.chunk_id)
            preview = (c.text[:50].replace("\n", " ") if c else "")
            bar = "█" * int(round(a.score * 10))
            print(f"  {a.chunk_id:<22}{a.score:<6.3f} {bar:<10} {preview}")
    return 0 if any_attr else 0


def _cmd_insights(args: argparse.Namespace) -> int:
    traces = read_traces(args.traces)
    if not traces:
        print("No traces.", file=sys.stderr)
        return 1

    if len(traces) > 1:
        agg = aggregate_metrics(traces)
        lat = agg["latency_ms"]
        print(f"Fleet: {agg['n_traces']} traces · {len(agg['config_fingerprints'])} config(s) · "
              f"latency p50/p95 {lat['p50']:.0f}/{lat['p95']:.0f} ms · cost {agg['cost_usd_total']:.4f} USD")
        for ins in generate_fleet_insights(traces):
            print(f"  {ins.line()}")
        print()

    for t in traces:
        print(f"=== Query {t.query_id[:8]} ===  {t.query}")
        found = generate_insights(t)
        if not found:
            print("  (no notable findings — run attribution for grounding/efficiency insights)")
        for ins in found:
            print(f"  {ins.line()}")
        print()
    return 0


def _cmd_doc(args: argparse.Namespace) -> int:
    traces = read_traces(args.traces)
    out = args.output or "datasheet.md"
    write_datasheet(out, traces, title=args.title)
    print(f"Wrote datasheet for {len(traces)} trace(s) → {out}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    # Import lazily so `raglens version` etc. don't need the example present.
    from importlib import import_module

    try:
        demo = import_module("examples.simple_pipeline")
    except Exception:
        import os

        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = os.path.join(here, "examples", "simple_pipeline.py")
        if not os.path.exists(script):
            print("Bundled example not found.", file=sys.stderr)
            return 1
        runpy.run_path(script, run_name="__main__")
        return 0
    demo.main()
    return 0


def _cmd_version(args: argparse.Namespace) -> int:
    print(f"raglens {__version__}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="raglens", description="Explainability & auto-docs for your RAG pipeline.")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="Run a Python pipeline script with raglens importable.")
    pr.add_argument("script", help="Path to a .py script that uses raglens.")
    pr.add_argument("script_args", nargs=argparse.REMAINDER, help="Args passed through to the script.")
    pr.set_defaults(func=_cmd_run)

    pe = sub.add_parser("explain", help="Show counterfactual attribution from a trace file.")
    pe.add_argument("traces", help="Path to a .jsonl trace file.")
    pe.add_argument("--query-id", help="Only show traces whose id starts with this prefix.")
    pe.set_defaults(func=_cmd_explain)

    pi = sub.add_parser("insights", help="Print auto-generated insights from traces.")
    pi.add_argument("traces", help="Path to a .jsonl trace file.")
    pi.set_defaults(func=_cmd_insights)

    pd = sub.add_parser("doc", help="Render a RAG datasheet from traces.")
    pd.add_argument("traces", help="Path to a .jsonl trace file.")
    pd.add_argument("-o", "--output", help="Output path (.md or .html). Default: datasheet.md")
    pd.add_argument("--title", default="RAG Datasheet", help="Datasheet title.")
    pd.set_defaults(func=_cmd_doc)

    pdemo = sub.add_parser("demo", help="Run the bundled end-to-end example.")
    pdemo.set_defaults(func=_cmd_demo)

    pv = sub.add_parser("version", help="Print version.")
    pv.set_defaults(func=_cmd_version)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
