import argparse
from pathlib import Path
import json


## another eval variant specifically tailored to the baseline experiments
## w/o RAG and RAG
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=Path, required=True, help="EXPERIMENT_ID to generate report from")
    args = parser.parse_args()

    try:
        with open(f"outputs/{args.experiment_id}/res.json", "r") as f:
            res_json = json.load(f)

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
                f"|Prediction + Eval |\n")
            f.write(f"| -: |\n")

            f.write(
                f"| {res_json['runtime_pred_eval']:.2f} s |\n\n---\n\n")

            f.write("## Runtime-average\n")
            f.write(
                f"| Prediction + Eval - per item |\n")
            f.write(f"|-:|\n")
            f.write(
                f"| {(res_json['runtime_pred_eval'] / res_json['num_items']):.2f} s |\n\n---\n\n")

    except Exception:
        raise Exception(f"EXPERIMENT_ID={args.experiment_id} is not a valid experiment ID")


if __name__ == "__main__":
    main()
