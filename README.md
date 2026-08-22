# Open-weight competition in the market for inference

Data collection and analysis for a study of how open-weight models compete with
proprietary ones on a large inference router. The pipeline builds a model-level panel
of prices, capability scores, weight-release status and revealed usage, estimates the
own-price elasticity of demand from posted-price revisions, tests whether model
releases displace incumbents, and computes what removing open-weight models from the
choice set would cost users.

## Layout

```
collect/    downloads: live endpoints, model pages, Hugging Face, archive captures
build/      panel construction from the raw downloads
analysis/   capability vectors, demand estimation, event studies, counterfactuals
data/           raw/ (downloaded, gitignored) and derived/ (analysis panels)
results.json, robustness.json   the estimates and the robustness checks
```

## Running it

```
pip install -r requirements.txt
python run_all.py                 # build the panels and estimate
python run_all.py collect         # re-download everything first
```

Modules run as package entry points, so invoke them from the repository root:

```
python -m analysis.results
python -m analysis.robustness
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

`collect/flight.py` parses the streamed render payload that the leaderboard and
model pages embed; `collect/wayback.py` is the rate-limited archive client.

## Pipeline

| Stage | Modules |
|---|---|
| `collect` | `fetch_live`, `fetch_model_pages`, `fetch_endpoints`, `fetch_hf`, `fetch_price_history`, `fetch_usage_history` |
| `build` | `build_models`, `build_speed`, `build_supply`, `build_tasks`, `build_panel`, `long_panel` |
| `analysis` | `results`, `robustness` |

Estimation lives in `capability.py` (capability vectors, similarity, imputation),
`demand.py` and `estimate.py` (share regressions and instrument diagnostics),
`event_study.py` (release event studies), `price_dynamics.py` (price survival and the
response of volume to a revision), `supply.py` and `counterfactual.py` (nested-logit
inversion and the counterfactual choice set), `task_counterfactual.py` (the same exercise
inside each task market), and `frontier.py` (the hypothetical entrant).

`results.json` holds every estimate the analysis produces; `robustness.json` holds the
alternative specifications.
