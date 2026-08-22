# Open-weight competition in the market for inference

Data collection and analysis for a study of how open-weight models compete with
proprietary ones on a large inference router. The pipeline builds a model-level panel
of prices, capability scores, weight-release status and revealed usage, estimates the
own-price elasticity of demand from posted-price revisions, tests whether model
releases displace incumbents, and computes what removing open-weight models from the
choice set would cost users.

## Layout

```
code/     collection, panel construction, estimation, figures
data/     raw/ (downloaded, gitignored) and derived/ (analysis panels)
figures/  figures used in the paper
paper/    LaTeX source
results.json, robustness.json   every number quoted in the paper
```

## Running it

```
pip install -r requirements.txt
python run_all.py                 # build panels, estimate, draw figures
python run_all.py collect         # re-download everything first
```

`collect` takes several hours because the Internet Archive endpoints are rate limited;
the other stages take a few minutes. The derived panels are committed, so the analysis
runs without re-collecting.

## Data

Everything comes from public, unauthenticated endpoints:

- OpenRouter catalogue, leaderboard, benchmark and task endpoints, and the per-model
  pages, which embed a trailing-month daily usage series in their render payload.
- Hugging Face model metadata, for parameter counts.
- Internet Archive captures of the catalogue and leaderboard, which extend the panel
  back to 2023.

`code/flight.py` parses the streamed render payload that the leaderboard and model
pages embed; `code/wayback.py` is the rate-limited archive client.

## Pipeline

| Stage | Scripts |
|---|---|
| Collect | `fetch_live`, `fetch_model_pages`, `fetch_endpoints`, `fetch_hf`, `fetch_price_history`, `fetch_usage_history` |
| Build | `build_models`, `build_speed`, `build_supply`, `build_tasks`, `build_panel`, `long_panel` |
| Analyse | `results`, `robustness` |
| Figures | `fig_market`, `fig_structure`, `fig_supply`, `fig_prices`, `fig_entry`, `fig_counterfactual`, `frontier_sim` |

Estimation lives in `capability.py` (capability vectors, similarity, imputation),
`demand.py` and `estimate.py` (share regressions and instrument diagnostics),
`event_study.py` (release event studies), `supply.py` and `counterfactual.py`
(nested-logit inversion and the counterfactual choice set), and
`task_counterfactual.py` (the same exercise inside each task market).

## Paper

```
cd paper && tectonic -X compile main.tex --outdir build
```
