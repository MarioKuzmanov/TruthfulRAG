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

        pipeline_mode = stats_jsonl[0].get("pipeline_mode")
        if pipeline_mode is None:
            pipeline_mode = (
                "direct-paths"
                if "runtime_path_extraction" in stats_jsonl[0]
                else "kg"
            )

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
            runtime_kg_building = 0.0
            runtime_kg_retrieval = 0.0
            runtime_path_extraction = 0.0
            runtime_entropy_filter = 0.0
            runtime_pred_eval = 0.0
            paths = 0
            filtered_paths = 0
            extraction_errors = 0
            for stat in stats_jsonl:
                runtime_kg_building += stat.get("runtime_kg_building", 0.0)
                runtime_kg_retrieval += stat.get("runtime_kg_retrieval", 0.0)
                runtime_path_extraction += stat.get("runtime_path_extraction", 0.0)
                runtime_entropy_filter += stat["runtime_entropy_filter"]
                runtime_pred_eval += stat["runtime_pred_eval"]
                paths += len(stat["elements_list"])
                filtered_paths += len(stat["filtered_elements_list"])
                if stat.get("extraction_error") is not None:
                    extraction_errors += 1

            total_time = (
                runtime_kg_building
                + runtime_kg_retrieval
                + runtime_path_extraction
                + runtime_entropy_filter
                + runtime_pred_eval
            )
            if pipeline_mode == "direct-paths":
                f.write(
                    "| Total | Direct Path Extraction | Entropy Filter | "
                    "Prediction + Eval | Paths | Filtered Paths | Extraction Errors |\n"
                )
                f.write("| -: | -: | -: | -: | -: | -: | -: |\n")
                f.write(
                    f"| {total_time:.2f} s | {runtime_path_extraction:.2f} s | "
                    f"{runtime_entropy_filter:.2f} s | {runtime_pred_eval:.2f} s | "
                    f"{paths} | {filtered_paths} | {extraction_errors} |\n\n---\n\n"
                )
            else:
                f.write(
                    "| Total | KG Building | KG Retrieval | Entropy Filter | "
                    "Prediction + Eval | Paths | Filtered Paths |\n"
                )
                f.write("| -: | -: | -: | -: | -: | -: | -: |\n")
                f.write(
                    f"| {total_time:.2f} s | {runtime_kg_building:.2f} s | "
                    f"{runtime_kg_retrieval:.2f} s | {runtime_entropy_filter:.2f} s | "
                    f"{runtime_pred_eval:.2f} s | {paths} | {filtered_paths} |\n\n---\n\n"
                )

            f.write("## Runtime-average\n")
            if pipeline_mode == "direct-paths":
                f.write(
                    "| Direct Path Extraction - per item | Entropy Filter - per item | "
                    "Prediction + Eval - per item | Paths - per item | "
                    "Filtered Paths - per item |\n"
                )
                f.write("| -: | -: | -: | -: | -: |\n")
                f.write(
                    f"| {(runtime_path_extraction / res_json['num_items']):.2f} s | "
                    f"{(runtime_entropy_filter / res_json['num_items']):.2f} s | "
                    f"{(runtime_pred_eval / res_json['num_items']):.2f} s | "
                    f"{(paths / res_json['num_items']):.2f} | "
                    f"{(filtered_paths / res_json['num_items']):.2f} |\n\n---\n\n"
                )
            else:
                f.write(
                    "| KG Building - per item | KG Retrieval - per item | "
                    "Entropy Filter - per item | Prediction + Eval - per item | "
                    "Paths - per item | Filtered Paths - per item |\n"
                )
                f.write("| -: | -: | -: | -: | -: | -: |\n")
                f.write(
                    f"| {(runtime_kg_building / res_json['num_items']):.2f} s | "
                    f"{(runtime_kg_retrieval / res_json['num_items']):.2f} s | "
                    f"{(runtime_entropy_filter / res_json['num_items']):.2f} s | "
                    f"{(runtime_pred_eval / res_json['num_items']):.2f} s | "
                    f"{(paths / res_json['num_items']):.2f} | "
                    f"{(filtered_paths / res_json['num_items']):.2f} |\n\n---\n\n"
                )


    except Exception:
        raise Exception(f"EXPERIMENT_ID={args.experiment_id} is not a valid experiment ID")


if __name__ == "__main__":
    main()
