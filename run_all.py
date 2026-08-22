"""Run the pipeline end to end: collect, build panels, estimate, make figures."""
import subprocess
import sys

COLLECT = ["fetch_live", "fetch_model_pages", "fetch_endpoints", "fetch_hf",
           "fetch_price_history", "fetch_usage_history"]
BUILD = ["build_models", "build_speed", "build_supply", "build_tasks",
         "build_panel", "long_panel"]
ANALYSE = ["results", "robustness"]
FIGURES = ["fig_market", "fig_structure", "fig_supply", "fig_prices",
           "fig_elasticity", "fig_entry", "fig_counterfactual", "frontier_sim"]

PACKAGE = {"collect": COLLECT, "build": BUILD, "analysis": ANALYSE, "plots": FIGURES}


def run(stage, modules):
    print(f"\n=== {stage} ===", flush=True)
    for m in modules:
        print(f"--- {m}", flush=True)
        subprocess.run([sys.executable, "-m", f"src.{stage}.{m}"], check=True)


if __name__ == "__main__":
    stages = sys.argv[1:] or ["build", "analysis", "plots"]
    for stage in ("collect", "build", "analysis", "plots"):
        if stage in stages:
            run(stage, PACKAGE[stage])   # collect takes hours: the archive is rate limited
