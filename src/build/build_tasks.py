"""Task-level market shares (30-day window) from the OpenRouter task-spend panel."""
import json, pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW, DER = ROOT / "data" / "raw", ROOT / "data" / "derived"


def main():
    d = json.loads((RAW / "rankings_task_spend.json").read_text())["data"]
    rows = []
    for measure in ("spend", "tokens"):
        blk = d[measure]
        for t in blk["tasks"]:
            for m in t["models"]:
                rows.append(dict(measure=measure, task=t["tag"],
                                 macro=t["macroCategory"],
                                 task_share_of_total=t["spendShareOfTotal"],
                                 permaslug=m["model"], share=m["share"],
                                 delta_pp=m["deltaPp"]))
    df = pd.DataFrame(rows)
    xs = pd.read_csv(DER / "models_cross_section.csv")
    df = df.merge(xs[["permaslug", "name", "author", "open_weight", "p_in", "p_out",
                      "p_blend", "q_intelligence", "q_coding", "q_agentic"]],
                  on="permaslug", how="left")
    df.to_csv(DER / "task_markets.csv", index=False)

    sp = df[df.measure == "spend"]
    inside = sp.groupby("task")["share"].sum()
    print(f"tasks: {sp.task.nunique()}  models: {sp.permaslug.nunique()}")
    print(f"inside share per task: mean {inside.mean():.3f}  min {inside.min():.3f}  max {inside.max():.3f}")
    print(f"unmatched models: {sp[sp.name.isna()].permaslug.unique()}")
    open_share = (sp.assign(w=sp.share * sp.task_share_of_total)
                    .groupby("open_weight")["w"].sum())
    print("open-weight share of covered spend:\n", (open_share / open_share.sum()).round(3))


if __name__ == "__main__":
    main()
