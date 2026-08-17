"""Run the pipeline end to end: collect, build panels, estimate, make figures."""
import pathlib
import subprocess
import sys

CODE = pathlib.Path(__file__).resolve().parent / "code"

COLLECT = ["fetch_live.py", "fetch_model_pages.py", "fetch_endpoints.py",
           "fetch_hf.py", "fetch_price_history.py", "fetch_usage_history.py"]
BUILD = ["build_models.py", "build_speed.py", "build_supply.py", "build_tasks.py",
         "build_panel.py", "long_panel.py"]
ANALYSE = ["results.py", "robustness.py"]
FIGURES = ["fig_market.py", "fig_prices.py", "fig_entry.py", "fig_counterfactual.py"]


def run(stage, scripts):
    print(f"\n=== {stage} ===", flush=True)
    for s in scripts:
        print(f"--- {s}", flush=True)
        subprocess.run([sys.executable, str(CODE / s)], check=True, cwd=CODE)


if __name__ == "__main__":
    stages = sys.argv[1:] or ["build", "analyse", "figures"]
    if "collect" in stages:
        run("collect", COLLECT)   # hours: the archive endpoints are rate limited
    if "build" in stages:
        run("build", BUILD)
    if "analyse" in stages:
        run("analyse", ANALYSE)
    if "figures" in stages:
        run("figures", FIGURES)
