import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

def aggregate(final_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Processes a list of evaluation results to generate aggregated statistics.

    This function first groups results into "multimodal" or "text". Within
    each group, it subdivides by "source". A sub-breakdown by "eval_method"
    is only provided if more than one evaluation method exists for that source.
    The keys for modality, source, and eval_method are sorted alphabetically
    to ensure a consistent output order.

    - Errors are categorized as "format_error" or "judge_error".
    - "mean_metrics": A direct average of all numerical metric values.
    - "strict_mean_metrics": An average where records with errors have their
      metric values treated as 0.0. This key is only included in the output
      for a scope (source or method) if there were failures within that scope.

    Args:
        final_results: A list of dictionaries, where each dictionary is a full
                       per-sample evaluation record.

    Returns:
        A dictionary containing the aggregated statistics, structured by
        modality and then source. An overall summary is included at the top level.
    """
    # === Step 1: Initial Grouping and Global Counts ===

    global_format_errors = 0
    global_judge_errors = 0
    grouped_by_modality_and_source = defaultdict(lambda: defaultdict(list))

    for record in final_results:
        eval_details = record.get("eval_details", {})
        if "format_error" in eval_details:
            global_format_errors += 1
        if "judge_error" in eval_details:
            global_judge_errors += 1

        modality = "multimodal" if record.get("images_path") or record.get("ct_path") else "text"
        source = record.get("source", "unknown_source")
        grouped_by_modality_and_source[modality][source].append(record)

    # === Step 2: Main Aggregation Loop ===

    aggregated_stats = {}

    for modality in sorted(grouped_by_modality_and_source.keys()):
        sources = grouped_by_modality_and_source[modality]
        aggregated_stats[modality] = {}

        for source in sorted(sources.keys()):
            records = sources[source]
            source_format_errors = 0
            source_judge_errors = 0
            source_mean_accumulator = defaultdict(lambda: {"sum": 0.0, "count": 0})
            source_strict_mean_accumulator = defaultdict(lambda: {"sum": 0.0, "count": 0})
            
            method_stats = defaultdict(lambda: {
                "total": 0, "format_errors": 0, "judge_errors": 0,
                "mean_metrics_accumulator": defaultdict(lambda: {"sum": 0.0, "count": 0}),
                "strict_mean_metrics_accumulator": defaultdict(lambda: {"sum": 0.0, "count": 0})
            })

            for record in records:
                method = record.get("eval_method", "unknown_method")
                eval_details = record.get("eval_details", {})
                has_format_error = "format_error" in eval_details
                has_judge_error = "judge_error" in eval_details
                is_failed = has_format_error or has_judge_error

                method_data = method_stats[method]
                method_data["total"] += 1
                if has_format_error:
                    method_data["format_errors"] += 1
                    source_format_errors += 1
                if has_judge_error:
                    method_data["judge_errors"] += 1
                    source_judge_errors += 1

                metrics = eval_details.get("evaluation_metrics", {})
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            strict_value = 0.0 if is_failed else value
                            
                            source_mean_accumulator[metric_name]["sum"] += value
                            source_mean_accumulator[metric_name]["count"] += 1
                            method_data["mean_metrics_accumulator"][metric_name]["sum"] += value
                            method_data["mean_metrics_accumulator"][metric_name]["count"] += 1
                            
                            # Accumulate strict values only if there's a chance they'll be used
                            if is_failed:
                                source_strict_mean_accumulator[metric_name]["sum"] += strict_value
                                method_data["strict_mean_metrics_accumulator"][metric_name]["sum"] += strict_value
                            else: # if not failed, strict_value == value
                                source_strict_mean_accumulator[metric_name]["sum"] += value
                                method_data["strict_mean_metrics_accumulator"][metric_name]["sum"] += value
                            
                            source_strict_mean_accumulator[metric_name]["count"] += 1
                            method_data["strict_mean_metrics_accumulator"][metric_name]["count"] += 1

            # --- Final Assembly for the Source ---

            total_source_records = len(records)
            total_source_errors = source_format_errors + source_judge_errors
            source_mean_metrics = {
                name: data["sum"] / data["count"]
                for name, data in source_mean_accumulator.items() if data["count"] > 0
            }
            
            source_output = {
                "total_records": total_source_records,
                "successful_evaluations": total_source_records - total_source_errors,
                "format_errors": source_format_errors,
                "judge_errors": source_judge_errors,
                "mean_metrics": source_mean_metrics,
            }

            # UPDATED: Conditionally add 'strict_mean_metrics' at the source level
            if total_source_errors > 0:
                source_strict_mean_metrics = {
                    name: data["sum"] / data["count"]
                    for name, data in source_strict_mean_accumulator.items() if data["count"] > 0
                }
                source_output["strict_mean_metrics"] = source_strict_mean_metrics

            if len(method_stats) > 1:
                by_eval_method = {}
                for method in sorted(method_stats.keys()):
                    stats = method_stats[method]
                    total_errors = stats["format_errors"] + stats["judge_errors"]
                    method_mean_metrics = {
                        name: data["sum"] / data["count"]
                        for name, data in stats["mean_metrics_accumulator"].items() if data["count"] > 0
                    }
                    
                    method_output = {
                        "total_records": stats["total"],
                        "successful_evaluations": stats["total"] - total_errors,
                        "format_errors": stats["format_errors"],
                        "judge_errors": stats["judge_errors"],
                        "mean_metrics": method_mean_metrics,
                    }

                    # UPDATED: Conditionally add 'strict_mean_metrics' at the method level
                    if total_errors > 0:
                        method_strict_mean_metrics = {
                            name: data["sum"] / data["count"]
                            for name, data in stats["strict_mean_metrics_accumulator"].items() if data["count"] > 0
                        }
                        method_output["strict_mean_metrics"] = method_strict_mean_metrics
                    
                    by_eval_method[method] = method_output
                source_output["by_eval_method"] = by_eval_method

            aggregated_stats[modality][source] = source_output

    # === Step 3: Construct the final output with summary ===

    final_output = {
        "summary": {
            "total_records": len(final_results),
            "total_format_errors": global_format_errors,
            "total_judge_errors": global_judge_errors,
        },
        **aggregated_stats
    }

    return final_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation results from a per_sample_eval.jsonl file."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input per_sample_eval.jsonl file."
    )
    args = parser.parse_args()
    input_path = Path(args.input_file)
    if not input_path.is_file():
        print(f"Error: Input file not found at {input_path}")
        exit(1)
    print(f"Reading data from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        all_results_data = [json.loads(line) for line in f]
    print(f"Loaded {len(all_results_data)} records.")
    print("\nRunning aggregation...")
    aggregated_statistics = aggregate(all_results_data)
    print("\n--- Aggregation Summary ---")
    print(json.dumps(aggregated_statistics, indent=4))
    output_path = input_path.parent / "aggregated_summary.json"
    print(f"\nSaving aggregated summary to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(aggregated_statistics, f_out, indent=4, ensure_ascii=False)
    print("Aggregation complete.")

