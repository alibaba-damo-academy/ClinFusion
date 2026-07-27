import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

def _calculate_staging_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculates a specific set of staging evaluation metrics for a given list of records.

    This helper function sums up the values from the 'correctness' dictionary
    and then computes various accuracy metrics based on those sums.

    Args:
        records: A list of evaluation records.

    Returns:
        A dictionary containing the calculated metrics.
    """
    if not records:
        return {}

    sum_correctness = defaultdict(int)
    total_records = len(records)

    for record in records:
        # Safely navigate to the 'correctness' dictionary
        correctness_data = record.get("eval_details", {}).get("evaluation_metrics", {}).get("correctness")
        if isinstance(correctness_data, dict):
            for key, val in correctness_data.items():
                if isinstance(val, (int, float)):
                    sum_correctness[key] += val
    
    # Small epsilon to avoid division by zero
    epsilon = 1e-8

    # Assemble final metrics dictionary using the exact keys from the user's reference
    final_metrics = {
        "overall_bclc_accuracy": sum_correctness["bclc"] / total_records,
        "overall_cnlc_accuracy": sum_correctness["cnlc"] / total_records,
        "overall_tnm_accuracy": sum_correctness["tnm"] / total_records,
        
        "overall_bclc_pos_accuracy": sum_correctness["bclc_pos"] / (sum_correctness["high_confident_num"] + epsilon),
        "overall_cnlc_pos_accuracy": sum_correctness["cnlc_pos"] / (sum_correctness["high_confident_num"] + epsilon),
        "overall_tnm_pos_accuracy": sum_correctness["tnm_pos"] / (sum_correctness["high_confident_num"] + epsilon),
        
        "overall_bclc_neg_accuracy": (sum_correctness["bclc"] - sum_correctness["bclc_pos"]) / (total_records - sum_correctness["high_confident_num"] + epsilon),
        "overall_cnlc_neg_accuracy": (sum_correctness["cnlc"] - sum_correctness["cnlc_pos"]) / (total_records - sum_correctness["high_confident_num"] + epsilon),
        "overall_tnm_neg_accuracy": (sum_correctness["tnm"] - sum_correctness["tnm_pos"]) / (total_records - sum_correctness["high_confident_num"] + epsilon),

        "overall_tumor_number_accuracy": sum_correctness["tumor_number"] / total_records,
        "overall_tumor_size_accuracy": sum_correctness["tumor_size"] / total_records,
        "overall_vessel_invasion_accuracy": sum_correctness["vessel_invasion"] / total_records,
        
        "overall_vessel_invasion_accuracy_pos": (sum_correctness["vessel_invasion_pos"] / total_records) / (1 - (sum_correctness["vessel_invasion"] / total_records) + epsilon),
        "overall_vessel_invasion_accuracy_neg": (sum_correctness["vessel_invasion_neg"] / total_records) / (1 - (sum_correctness["vessel_invasion"] / total_records) + epsilon),
        
        "overall_N_stage_accuracy": sum_correctness["N_stage"] / total_records,
        "overall_N_stage_accuracy_pos": (sum_correctness["N_stage_pos"] / total_records) / (1 - (sum_correctness["N_stage"] / total_records) + epsilon),
        "overall_N_stage_accuracy_neg": (sum_correctness["N_stage_neg"] / total_records) / (1 - (sum_correctness["N_stage"] / total_records) + epsilon),
        
        "overall_M_stage_accuracy": sum_correctness["M_stage"] / total_records,
        "overall_M_stage_accuracy_pos": (sum_correctness["M_stage_pos"] / total_records) / (1 - (sum_correctness["M_stage"] / total_records) + epsilon),
        "overall_M_stage_accuracy_neg": (sum_correctness["M_stage_neg"] / total_records) / (1 - (sum_correctness["M_stage"] / total_records) + epsilon),
        
        "overall_bclc_soft_accuracy": sum_correctness.get("bclc_soft", 0) / total_records,
        "overall_cnlc_soft_accuracy": sum_correctness.get("cnlc_soft", 0) / total_records,
        "overall_tnm_soft_accuracy": sum_correctness.get("tnm_soft", 0) / total_records,
    }

    # Clean up any potential NaN/inf values from division by zero
    for key, value in final_metrics.items():
        if not isinstance(value, (int, float)) or not -float('inf') < value < float('inf'):
            final_metrics[key] = 0.0

    return final_metrics


def aggregate(final_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Processes a list of evaluation results to generate a single, overall
    summary of statistics based on staging correctness metrics.

    This function calculates a detailed set of accuracy metrics across all
    provided records, derived from the 'correctness' field in the
    evaluation details. There is no grouping by source or modality.

    Args:
        final_results: A list of dictionaries, where each dictionary is a full
                       per-sample evaluation record.

    Returns:
        A dictionary containing a single 'summary' key with the aggregated
        statistics for all records.
    """
    # === Step 1: Handle Empty Input ===
    if not final_results:
        return {"summary": {"total_records": 0, "metrics": {}}}

    # === Step 2: Calculate Global Summary Metrics ===
    # All records are processed together to get a single summary.
    summary_metrics = _calculate_staging_metrics(final_results)

    # === Step 3: Construct Final Output ===
    final_output = {
        "summary": {
            "total_records": len(final_results),
            "metrics": summary_metrics
        }
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
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            all_results_data = [json.loads(line) for line in f]
        print(f"Loaded {len(all_results_data)} records.")
    except json.JSONDecodeError as e:
        print(f"Error reading JSONL file: {e}")
        print("Please ensure each line in the input file is a valid JSON object.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)

    print("\nRunning aggregation...")
    aggregated_statistics = aggregate(all_results_data)
    
    print("\n--- Aggregation Summary ---")
    print(json.dumps(aggregated_statistics, indent=4))
    
    output_path = input_path.parent / "aggregated_summary.json"
    print(f"\nSaving aggregated summary to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(aggregated_statistics, f_out, indent=4, ensure_ascii=False)
        
    print("Aggregation complete.")
