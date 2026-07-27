import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

def _calculate_stats_for_group(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Helper function to calculate statistics for a given list of records.

    This calculates total records, failures, mean metrics, and strict mean metrics.
    - "failures": Count of records with "format_error" or "judge_error".
    - "mean_metrics": Average of only the valid, numerical metric values found.
    - "strict_mean_metrics": Average where failures count as 0. The denominator is
      the total number of records in the group. This key is only included if
      there are failures.

    Args:
        records: A list of per-sample evaluation record dictionaries.

    Returns:
        A dictionary containing the calculated statistics for the group.
    """
    if not records:
        return {"total_records": 0}

    total_records = len(records)
    failures = 0
    mean_accumulator = defaultdict(lambda: {"sum": 0.0, "count": 0})
    strict_mean_sum = defaultdict(float)

    for record in records:
        eval_details = record.get("eval_details", {})
        is_failure = eval_details.get("format_error") or eval_details.get("judge_error")

        if is_failure:
            failures += 1

        metrics = eval_details.get("evaluation_metrics", {})
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    # Accumulate for regular mean (only includes valid values)
                    mean_accumulator[metric_name]["sum"] += value
                    mean_accumulator[metric_name]["count"] += 1
                    # Accumulate for strict mean (non-failures only)
                    if not is_failure:
                        strict_mean_sum[metric_name] += value

    # Calculate final metrics
    mean_metrics = {
        name: data["sum"] / data["count"]
        for name, data in mean_accumulator.items() if data["count"] > 0
    }
    
    output = {
        "total_records": total_records,
        "mean_metrics": mean_metrics,
    }

    if failures > 0:
        strict_mean_metrics = {
            name: s_sum / total_records
            for name, s_sum in strict_mean_sum.items()
        }
        output["strict_mean_metrics"] = strict_mean_metrics

    return output

def aggregate(final_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Processes evaluation results to generate nested, aggregated statistics.

    This function's primary breakdown is by modality. For each modality, it
    provides a summary of all records for that modality, as well as a further
    breakdown by the evaluation method used. Keys are sorted alphabetically for
    consistent output.

    The top-level "summary" provides global statistics across all records.

    Within each level (global, per-modality), the statistics include:
    - "failures": Count of records with "format_error" or "judge_error".
    - "mean_metrics": A direct average of all valid numerical metric values.
    - "strict_mean_metrics": An average where records with errors have their
      metric values treated as 0.0. This key is only included if there were
      failures within that scope.

    Args:
        final_results: A list of dictionaries, where each dictionary is a full
                       per-sample evaluation record.

    Returns:
        A dictionary with a global "summary" and a nested "by_modality"
        breakdown, which itself contains summaries and "by_eval_method" stats.
    """
    # === Step 1: Group records by modality first ===
    by_modality_groups = defaultdict(list)
    for record in final_results:
        modality = "multimodal" if record.get("images_path") else "text"
        by_modality_groups[modality].append(record)

    # === Step 2: Process each modality group to create nested stats ===
    modality_stats = {}
    for modality, records_in_modality in sorted(by_modality_groups.items()):
        # Calculate the summary for the entire modality
        modality_summary = _calculate_stats_for_group(records_in_modality)

        # Now, sub-group the records within this modality by eval_method
        by_eval_method_subgroups = defaultdict(list)
        for record in records_in_modality:
            eval_method = record.get("eval_method", "unknown_method")
            by_eval_method_subgroups[eval_method].append(record)
        
        # Calculate stats for each eval_method subgroup
        eval_method_breakdown = {}
        for method, records_in_method in sorted(by_eval_method_subgroups.items()):
            eval_method_breakdown[method] = _calculate_stats_for_group(records_in_method)

        # Assemble the final structure for this modality
        modality_stats[modality] = {
            "summary": modality_summary,
            "by_eval_method": eval_method_breakdown,
        }

    # === Step 3: Calculate global summary stats (unchanged) ===
    global_stats = _calculate_stats_for_group(final_results)
    summary = {
        "total_records": global_stats.get("total_records", 0),
        "global_mean_metrics": global_stats.get("mean_metrics", {}),
    }
    if "strict_mean_metrics" in global_stats:
        summary["global_strict_mean_metrics"] = global_stats["strict_mean_metrics"]

    # === Step 4: Construct the final nested output ===
    final_output = {
        "summary": summary,
        "by_modality": modality_stats,
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

