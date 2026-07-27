import os
import sys
import argparse
import json
import math
import random
from pathlib import Path
import importlib.util

import numpy as np
import ray
import yaml
from tqdm import tqdm

# --- Add Project Root to Python Path ---
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from eval.InferenceEngine.rollout import RolloutWorker, kill_actors_and_wait, load_encrypted_jsonl
from concurrent.futures import ThreadPoolExecutor, as_completed



def main(args):
    # 1. LOAD CONFIG and CONNECT TO RAY
    # =================================================================
    print("--- 1. Initializing and Loading Config ---")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '6379')
    ray_address = f"{master_addr}:{master_port}"
    
    print(f"Connecting to Ray cluster at: {ray_address}")
    if 'None' in ray_address:
        ray.init(ignore_reinit_error=True)
    else:
        ray.init(address=ray_address, ignore_reinit_error=True)  
    print("Successfully connected to Ray.")
    
    path_cfg = config['path_config']
    model_cfg = config['model_config']
    cluster_cfg = config['cluster_config']
    gen_cfg = config['generation_config']

    print("\n--- 2. Preparing Data and Evaluator ---")
    eval_data_path = path_cfg['eval_data_path']

    if eval_data_path.endswith('.enc'):
        # NEW: Logic for encrypted files
        print(f"INFO: Encrypted data file detected. Using decryption loader for '{eval_data_path}'.")
        all_data = load_encrypted_jsonl(
            encrypted_path=eval_data_path,
            key_path=path_cfg.get('data_secret_key_path')
        )
    else:
        # ORIGINAL: Unchanged logic for standard JSONL files
        print(f"INFO: Unencrypted data file detected. Using standard loader for '{eval_data_path}'.")
        with open(eval_data_path, "r", encoding="utf-8") as f:
            all_data = [json.loads(line) for line in f]
    
    if not all_data:
        print("Input evaluation data file is empty. Exiting.")
        return
    
    initial_data_count = len(all_data)
    print(f"Loaded {initial_data_count} total data points from {path_cfg['eval_data_path']}")

    # --- START: NEW FILTERING LOGIC ---
    filter_cfg = config.get('filter_config')
    if filter_cfg:
        print("\n--- Applying Dataset Filters ---")

        # Define which keys should use "intersection" logic (for comma-separated string values)
        # Add any other future comma-separated fields to this set.
        INTERSECTION_KEYS = {'organs_involved'}

        # 1. Parse the filter rules from the config
        # Converts "mcq,report_generation" into {'mcq', 'report_generation'} for fast lookups
        parsed_filters = {
            key: set(val.strip() for val in value.split(','))
            for key, value in filter_cfg.items()
        }
        for key, values in parsed_filters.items():
            if key in INTERSECTION_KEYS:
                print(f"  - Rule: Keep if '{key}' contains any of {list(values)}")
            else:
                print(f"  - Rule: Keep if '{key}' is one of {list(values)}")

        # 2. Define a helper function for flexible filtering
        def check_item(item):
            """
            Checks if a single data item passes all defined filters.
            """
            for key, required_values in parsed_filters.items():
                item_value = item.get(key)
                if item_value is None:
                    return False  # Filter fails if the key is missing in the data

                # Logic for comma-separated fields like 'organs_involved'
                if key in INTERSECTION_KEYS:
                    # Convert the item's comma-separated string into a set of values
                    item_values_set = set(val.strip() for val in item_value.split(','))
                    # Check for intersection. If there's no overlap, the filter fails.
                    if not item_values_set.intersection(required_values):
                        return False
                # Original logic for exact-match fields like 'source' or 'language'
                else:
                    if item_value not in required_values:
                        return False
            
            return True # Item passes all filters

        # 3. Apply the filters using the helper function
        original_data = all_data
        all_data = [item for item in original_data if check_item(item)]
        
        print(f"\nFiltering complete. Kept {len(all_data)} out of {initial_data_count} records.")

        # 4. Print breakdown by 'source' (This part remains the same)
        if all_data:
            from collections import Counter
            source_counts = Counter(item['source'] for item in all_data)
            print("Filtered data breakdown by 'source':")
            for source, count in sorted(source_counts.items()):
                print(f"  - {source}: {count} records")
        else:
            print("Warning: All data was filtered out. No records remain for evaluation.")
            # We will exit here as there's nothing to process
            return

    else:
        print("No 'filter_config' found in config. Using all loaded data.")

    random.seed(42)
    random.shuffle(all_data)
    print("Data successfully shuffled with random seed 42.")

    evaluator_path_str = path_cfg.get('evaluator_path')
    print(f"Initializing Evaluator from {evaluator_path_str}")

    if not evaluator_path_str:
        print("  [!] Error: 'evaluator_path' not found in config['path_config'].")
        print("      Please specify the path to your evaluator.py file.")
        sys.exit(1) # Exit because the evaluator is essential for the script to run.

    evaluator_path = Path(evaluator_path_str)
    if not evaluator_path.is_file():
        print(f"  [!] Error: Custom evaluator file not found at: {evaluator_path}")
        sys.exit(1)

    try:
        # Dynamically import the evaluator module from the specified path
        spec = importlib.util.spec_from_file_location("custom_evaluator", evaluator_path)
        evaluator_module = importlib.util.module_from_spec(spec)
        # Add the module to sys.modules to handle potential relative imports within the evaluator file itself
        sys.modules["custom_evaluator"] = evaluator_module
        spec.loader.exec_module(evaluator_module)

        # Get the 'Evaluator' class from the module
        EvaluatorClass = getattr(evaluator_module, 'Evaluator', None)

        if EvaluatorClass and callable(EvaluatorClass):
            # Instantiate the custom Evaluator class
            evaluator = EvaluatorClass(config=config.get('evaluator_config',{}))
            print(f"Successfully loaded and initialized custom Evaluator from: {evaluator_path}")
        else:
            print(f"  [!] Error: 'Evaluator' class not found or not callable in {evaluator_path}")
            sys.exit(1)

    except Exception as e:
        print(f"  [!] Error during custom evaluator loading: {e}")
        print("      Please check your evaluator script for errors.")
        sys.exit(1)
    
    # Phase 1: Prepare prompts for the model-under-test
    print("Preparing prompts for all data points...")

    def prepare_prompts(item):
        return evaluator.get_prompt(item, item['eval_method'])

    data_for_inference = []
    with ThreadPoolExecutor(max_workers=192) as executor:
        futures = [executor.submit(prepare_prompts, item) for item in all_data]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Prompts (Parallel)"):
            result = future.result()
            data_for_inference.append(result)
    

    # 3. RUN DISTRIBUTED INFERENCE
    # =================================================================
    print("\n--- 3. Running Inference with Model-Under-Test ---")
    
    # Calculate optimal number of workers
    is_api_model = model_cfg.get('model_type') == 'api'
    if is_api_model:
        num_workers = 1
        print("API model detected. Using 1 worker for API calls.")
    else:
        max_available_workers = cluster_cfg['total_gpus'] // cluster_cfg['gpus_per_worker']
        max_useful_workers = max(1, len(data_for_inference) // gen_cfg['batch_size'])
        num_workers = min(max_available_workers, max_useful_workers)
        print(f"Local model detected. Launching {num_workers} worker(s).")
        
    data_chunks = np.array_split(data_for_inference, num_workers)
    data_chunks = [chunk.tolist() for chunk in data_chunks]
    
    # Launch RolloutWorker actors
    gpus_per_actor = 0 if is_api_model else cluster_cfg['gpus_per_worker']
    workers = [
        RolloutWorker.options(num_gpus=gpus_per_actor).remote(
            worker_id=i, config=config
        ) for i in range(num_workers)
    ]
    print(f"Launched {len(workers)} RolloutWorker actor(s).")

    # Process data chunks in parallel
    tasks = [
        worker.process_batch.remote(data_chunks[i])
        for i, worker in enumerate(workers)
    ]
    
    # Gather results (this list will contain the model_generation field)
    results_from_workers = ray.get(tasks)
    data_with_generations = [item for sublist in results_from_workers for item in sublist]
    print(f"Inference complete. Received {len(data_with_generations)} generated results.")
    
    # Clean up inference workers to free GPU memory
    kill_actors_and_wait(workers)
    print("Inference workers shut down and resources released.")

    # 4. PERFORM EVALUATION
    # =================================================================
    print("\n--- 4. Performing Final Evaluation with Judger LLM ---")
    def eval_wrapper(item):
        return evaluator.eval(item, item['eval_method'])

    final_results = []
    with ThreadPoolExecutor(max_workers=192) as executor:
        futures = [executor.submit(eval_wrapper, item) for item in data_with_generations]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating Generations (Parallel)"):
            result = future.result()
            final_results.append(result)

    print(f"Evaluation complete. Processed {len(final_results)} items.")

    # 5. AGGREGATE AND SAVE RESULTS
    # =================================================================
    print("\n--- 5. Aggregating and Saving Final Results ---")
    # The output path is now a directory
    output_dir = Path(path_cfg['eval_output_path'])
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregator_path_str = path_cfg.get('aggregator_path')
    if aggregator_path_str:
        print("\n--- 5a. Running Custom Aggregation ---")
        aggregator_path = Path(aggregator_path_str)
        print(f"Running custom aggregation from {aggregator_path}")
        if not aggregator_path.is_file():
            print(f"  [!] Warning: Custom aggregator file not found at: {aggregator_path}")
            print("      Skipping custom aggregation.")
        else:
            try:
                # Dynamically import the aggregator module from the specified path
                spec = importlib.util.spec_from_file_location("custom_aggregator", aggregator_path)
                aggregator_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(aggregator_module)

                # Get the 'aggregate' function from the module
                aggregator_func = getattr(aggregator_module, 'aggregate', None)

                if callable(aggregator_func):
                    # Run the aggregation function
                    aggregated_statistics = aggregator_func(final_results)
                    
                    # Pretty-print the results
                    print("Custom Aggregation Results:")
                    print(json.dumps(aggregated_statistics, indent=4))
                    
                    # Also save the aggregated results to a file
                    agg_output_path = output_dir / "aggregated_summary.json"
                    print(f"Saving aggregated summary to: {agg_output_path}")
                    with open(agg_output_path, "w", encoding="utf-8") as f_agg:
                        json.dump(aggregated_statistics, f_agg, indent=4, ensure_ascii=False)
                else:
                    print(f"  [!] Warning: 'aggregate' function not found or not callable in {aggregator_path}")
                    print("      Skipping custom aggregation.")

            except Exception as e:
                print(f"  [!] Error during custom aggregation: {e}")
                print("      Please check your aggregator script for errors. Skipping custom aggregation.")

    # Define paths for the individual files
    results_output_path = output_dir / "per_sample_eval.jsonl"
    failed_output_path = output_dir / "failed_data.jsonl"
    config_output_path = output_dir / "used_config.yaml"

    # Separate successful and failed records
    failed_records = []
    for record in final_results:
        # Check if the "evaluation_error" key exists in the eval_details dictionary.
        # Using .get() is a safe way to avoid a KeyError if 'eval_details' itself is missing.
        eval_details = record["eval_details"]
        if ("format_error" in eval_details) or ("judge_error" in eval_details):
            failed_records.append(record)

    # 1. Save all per-sample evaluation results to a JSONL file
    print(f"Saving all {len(final_results)} per-sample results to: {results_output_path}")
    with open(results_output_path, "w", encoding="utf-8") as f_out:
        for record in final_results:
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 2. Save only the failed cases to a separate JSONL file, if any exist
    if failed_records:
        print(f"Found {len(failed_records)} failed cases. Saving them to: {failed_output_path}")
        with open(failed_output_path, "w", encoding="utf-8") as f_fail:
            for record in failed_records:
                f_fail.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        print("No failed evaluation cases found.")

    # 3. Save the configuration used for this run to a YAML file
    print(f"Saving the run configuration to: {config_output_path}")
    with open(config_output_path, "w", encoding="utf-8") as f_cfg:
        # The 'config' variable holds the loaded configuration from the beginning
        yaml.dump(config, f_cfg, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\n--- Evaluation Complete ---")
    print(f"All outputs have been saved in the directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a full evaluation pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to the main config.yaml file.")
    parsed_args = parser.parse_args()
    main(parsed_args)
