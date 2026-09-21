import argparse
from pathlib import Path
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=Path, required=True, help="EXPERIMENT_ID to generate report from")
    args = parser.parse_args()

    try:
        with open(f"outputs/{args.experiment_id}/res.json", "r") as f:
            res_json = json.load(f)
        with open(f"outputs/{args.experiment_id}/stats.jsonl", "r") as f:
            lines = f.read().strip().split("\n")
            stats_jsonl = [json.loads(line) for line in lines]

        output_path = f"outputs/{args.experiment_id}/{args.experiment_id}-report.md"
        with open(output_path, "w") as f:
            ## quality
            f.write(f"# Run-`{args.experiment_id}` Report\n")
            f.write(f"## Quality\n")
            f.write(f"| num_items | exact_match | acc | f1 |\n")
            f.write(f"|-:| -:| -:| -:|\n")
            f.write(
                f"| {res_json['num_items']} | {res_json['exact_match']:.4f}% | {res_json['acc']:.4f}% | {res_json['f1']:.4f}% |\n\n---\n\n")

            ## runtime
            f.write("## Runtime\n")
            f.write(
                f"| Total | KG Building | KG Retrieval | Entropy Filter | Prediction + Eval | Paths | Filtered Paths |\n")
            f.write(f"| -: |-:| -:| -:| -:| -:| -:|\n")
            runtime_kg_building, runtime_kg_retrieval, runtime_entropy_filter, runtime_pred_eval, paths, filtered_paths = 0.0, 0.0, 0.0, 0.0, 0, 0
            for stat in stats_jsonl:
                runtime_kg_building += stat["runtime_kg_building"]
                runtime_kg_retrieval += stat["runtime_kg_retrieval"]
                runtime_entropy_filter += stat["runtime_entropy_filter"]
                runtime_pred_eval += stat["runtime_pred_eval"]
                paths += len(stat["elements_list"])
                filtered_paths += len(stat["filtered_elements_list"])
            total_time = runtime_kg_building + runtime_kg_retrieval + runtime_entropy_filter + runtime_pred_eval
            f.write(
                f"| {total_time:.2f} s | {runtime_kg_building:.2f} s | {runtime_kg_retrieval:.2f} s | {runtime_entropy_filter:.2f} s | {runtime_pred_eval:.2f} s | {paths} | {filtered_paths} |\n\n---\n\n")

            f.write("## Runtime-average\n")
            f.write(
                f"| KG Building - per item | KG Retrieval - per item | Entropy Filter - per item | Prediction + Eval - per item | Paths - per item | Filtered Paths - per item |\n")
            f.write(f"|-:| -:| -:| -:| -:| -:|\n")
            f.write(
                f"| {(runtime_kg_building / res_json['num_items']):.2f} s | {(runtime_kg_retrieval / res_json['num_items']):.2f} s | {(runtime_entropy_filter / res_json['num_items']):.2f} s | {(runtime_pred_eval / res_json['num_items']):.2f} s | {(paths / res_json['num_items']):.2f} | {(filtered_paths / res_json['num_items']):.2f} |\n\n---\n\n")


    except Exception:
        raise Exception(f"EXPERIMENT_ID={args.experiment_id} is not a valid experiment ID")


if __name__ == "__main__":
    main()
