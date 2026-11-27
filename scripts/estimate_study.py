"""Build a credential-free execution envelope for the proposed held-out matrix."""

import argparse
from collections import Counter
from pathlib import Path

from elevate_rag.config import Settings
from elevate_rag.data import dataset_manifest, write_json
from elevate_rag.datasets import MAIN
from elevate_rag.experiments import design
from elevate_rag.providers import Providers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/phase2/study-estimate.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/reference.toml"))
    args = parser.parse_args()
    settings = Settings.read(args.config)
    heldout = {name: dataset_manifest(Path("data") / name)["questions"]["heldout"] for name in MAIN}
    blocks = {}
    for number in range(1, 7):
        block = f"E{number}"
        totals = Counter()
        for case in design(block):
            if case.get("level") == "A" or case.get("folds") or case["variant"] == "offline-oracle":
                totals["separate_measurement_arms"] += 1
                continue
            limit = case.get("questions", heldout[case["dataset"]])
            repetitions = case.get("repetitions", 1)
            if case["factor"] == "external-framework":
                totals["external_arms"] += 1
                totals["external_questions"] += limit
                continue
            totals["internal_arms"] += 1
            totals["evaluated_questions"] += limit * repetitions
            totals["warmup_questions"] += case.get("warmup", 0) * repetitions
            if case["variant"] == "R4":
                totals["maximum_extra_fallback_calls"] += limit * repetitions
            if case["variant"] == "R3":
                totals["router_calls"] += limit * repetitions
        blocks[block] = dict(totals)
    answers = sum(row.get("evaluated_questions", 0) for row in blocks.values())
    calls = answers + sum(
        row.get("warmup_questions", 0) + row.get("maximum_extra_fallback_calls", 0)
        for row in blocks.values()
    )
    judge_tokens = (1200, 200)
    generator_tokens = (2800, 200)
    judged_usd = Providers.price(settings.model("judge"), *judge_tokens)
    write_json(
        args.out,
        {
            "classification": "planning estimates; no model requests made",
            "dataset_heldout_counts": heldout,
            "blocks": blocks,
            "assumptions": {
                "mean_generation_input_tokens": generator_tokens[0],
                "mean_generation_output_tokens": generator_tokens[1],
                "mean_judge_input_tokens": judge_tokens[0],
                "mean_judge_output_tokens": judge_tokens[1],
                "seconds_per_generation_scenario": [2, 6],
                "use_peak_configured_prices": True,
            },
            "internal_scenario": {
                "maximum_generation_calls_including_warmup_and_one_fallback": calls,
                "mean_scenario_generation_input_tokens": calls * generator_tokens[0],
                "mean_scenario_generation_output_tokens": calls * generator_tokens[1],
                "judge_cost_if_all_internal_rows_receive_one_grade_usd": answers * judged_usd,
                "judge_input_tokens": answers * judge_tokens[0],
                "judge_output_tokens": answers * judge_tokens[1],
                "serial_generation_only_hours": [calls * seconds / 3600 for seconds in (2, 6)],
            },
            "exclusions": [
                "indexing, warmup judge traffic, optional RAGAS and pairwise judging, model refresh, transport retries, external frameworks, paid E5/E6 tier rates, concurrency or discount effects"
            ],
        },
    )


if __name__ == "__main__":
    main()
