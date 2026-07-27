import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
import random
import time

import numpy as np
import ray
import torch
import yaml
from tqdm import tqdm

from .models.get_wrapper import get_wrapper

import copy
from torch.utils.data import Dataset, DataLoader
from .models.utils import read_image, read_nifti, image_resize, nifti_resize
import PIL

from cryptography.fernet import Fernet

def load_encrypted_jsonl(encrypted_path: str, key_path: str) -> list:
    """
    Decrypts a file's content in memory, parses it as JSONL, and returns a list of dictionaries.
    """
    print(f"INFO: Reading secret key from: {key_path}")
    try:
        with open(key_path, "rb") as f:
            key = f.read()
        fernet = Fernet(key)

        print(f"INFO: Reading encrypted data from: {encrypted_path}")
        with open(encrypted_path, "rb") as f_in:
            encrypted_data = f_in.read()

        print("INFO: Decrypting data in memory...")
        decrypted_bytes = fernet.decrypt(encrypted_data)

        print("INFO: Decoding and parsing JSONL data...")
        # Decode the bytes to a string, then split into lines
        decrypted_text = decrypted_bytes.decode('utf-8')
        lines = decrypted_text.splitlines()

        # Parse each line as a JSON object
        # The `if line` check is to avoid errors from potential empty lines
        data = [json.loads(line) for line in lines if line]
        
        print("INFO: Decryption and data loading successful.")
        return data

    except FileNotFoundError as e:
        print(f"FATAL: Decryption failed. A required file was not found: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"FATAL: Failed to parse decrypted data as JSON. The data may be corrupt or not in JSONL format. Error: {e}")
        sys.exit(1)
    except Exception as e:
        # This will catch decryption errors like InvalidToken from Fernet
        print(f"FATAL: Decryption or data loading failed. Error: {e}")
        sys.exit(1)

class RolloutDataset(Dataset):

    def __init__(self, data_chunk: list, min_image_size: int, max_image_size: int, max_image_num: int):
        self.data = data_chunk
        self.min_image_size = min_image_size
        self.max_image_size = max_image_size
        self.max_image_num = max_image_num

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the original data item, which includes image paths
        original_item = self.data[idx]
        
        # Create a deep copy to modify for model input, loading the actual images
        if "eval_details" in original_item:
            original_input = original_item["eval_details"]
        else:
            original_input = original_item

        model_input = copy.deepcopy(original_input)
        
        messages = model_input["messages"]
        image_paths = messages.get("image", [])
        nifti_paths = messages.get("nifti", [])

        # subsample image paths to max_image_num
        if len(image_paths) > self.max_image_num:
            image_paths = image_paths[:self.max_image_num]
        
        if image_paths:
            loaded_images = []
            for img_path in image_paths:
                try:
                    image = read_image(img_path)
                    image = image_resize(image, self.min_image_size, self.max_image_size)
                    loaded_images.append(image)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
            
            # Replace the list of paths with the list of loaded PIL.Image objects
            messages["image"] = loaded_images
        else:
            if "image" in messages:
                del messages["image"] 
        
        if nifti_paths:
            loaded_niftis = []
            for nifti_path in nifti_paths:
                try:
                    nifti = read_nifti(nifti_path)
                    nifti = nifti_resize(nifti, self.min_image_size, self.max_image_size)
                    loaded_niftis.append(nifti)
                except Exception as e:
                    raise Exception(f"Error loading nifti {nifti_path}: {e}")
            
            # Replace the list of paths with the list of loaded numpy arrays
            messages["nifti"] = loaded_niftis
            
        # Return both the original item (for logging/saving) and the processed model input
        return original_item, model_input

def rollout_collate_fn(batch):
    original_batch = [item[0] for item in batch]
    model_input_batch = [item[1] for item in batch]
    return original_batch, model_input_batch


def kill_actors_and_wait(actors: list, timeout_seconds: int = 60):
    """
    Gracefully shuts down Ray actors and waits for their GPU resources to be released.
    """
    if not actors:
        return

    print(f"\nAttempting to gracefully shut down {len(actors)} actor(s)...")
    
    # Trigger the shutdown method on all actors without waiting for each one individually
    shutdown_tasks = [actor.shutdown.remote() for actor in actors]
    
    try:
        # Wait for all the shutdown tasks to complete
        ray.get(shutdown_tasks, timeout=timeout_seconds)
        print("All actors confirmed graceful shutdown completion.")
    except ray.exceptions.RayTaskError as e:
        print(f"Warning: An error occurred during graceful shutdown of an actor: {e}")
        print("Proceeding with resource check anyway.")
    except Exception as e:
        print(f"Warning: An unexpected error occurred while waiting for shutdown: {e}")

# --- Ray Actor for Distributed Inference ---
@ray.remote(max_restarts=-1) # Restart actor on failure
class RolloutWorker:
    def __init__(self, worker_id: int, config: dict):
        self.worker_id = worker_id
        self.config = config

        # Set seed for reproducibility
        seed = config['generation_config'].get('seed', 42)
        worker_seed = seed + self.worker_id
        torch.manual_seed(worker_seed)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        print(f"[Worker {self.worker_id}] Seed set to {worker_seed}")

        model_type = config['model_config'].get('model_type')
        print(f"[Worker {self.worker_id}] Seed set to {worker_seed}. Model type: '{model_type}'.")

        # Initialize the model using the factory
        print(f"[Worker {self.worker_id}] Initializing model wrapper...")
        self.model = get_wrapper(
            model_type=config['model_config']['model_type'],
            model_path=config['model_config'].get('model_path'),
            model_config={**config['model_config'], **config['cluster_config']},
            generation_config=config['generation_config']
        )
        print(f"[Worker {self.worker_id}] Model ready.")

    def process_batch(self, data_chunk: list) -> list:
        num_items = len(data_chunk)
        if num_items == 0:
            return []

        dataset = RolloutDataset(
            data_chunk,
            min_image_size=self.model.min_image_size,
            max_image_size=self.model.max_image_size,
            max_image_num=self.model.max_image_num
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['generation_config']['batch_size'],
            shuffle=False,
            num_workers=4, 
            collate_fn=rollout_collate_fn,
            pin_memory=True 
        )

        worker_results = []
        
        for original_batch_data, model_input_batch in tqdm(dataloader, desc=f"Worker {self.worker_id}"):
            generated_texts = self.model.generate(model_input_batch)
            
            for original_item, generated_text in zip(original_batch_data, generated_texts):
                result_record = original_item.copy()
                
                # This logic remains the same as it operates on the dictionary structure
                if "eval_details" in result_record:
                    result_record['eval_details']['model_generation'] = generated_text
                else:
                    result_record['model_generation'] = generated_text
                worker_results.append(result_record)
        
        print(f"[Worker {self.worker_id}] Finished processing. Returning {len(worker_results)} items.")
        return worker_results
    
    
    def shutdown(self):
        """
        Allows the actor to perform a graceful shutdown, cleaning up its resources.
        """
        print(f"[Worker {self.worker_id}] Shutting down gracefully.")
        # Deleting the model attribute helps ensure vLLM's internal __del__
        # methods are called, which in turn shut down its worker processes.
        if hasattr(self, 'model'):
            del self.model
        # The actor process will terminate cleanly after this method returns.


def main(args):
    # 1. LOAD CONFIG and CONNECT TO RAY
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '6379')
    ray_address = f"{master_addr}:{master_port}"
    
    print(f"Connecting to Ray cluster at: {ray_address}")
    ray.init(address=ray_address, ignore_reinit_error=True)
    print("Successfully connected to Ray.")

    # 2. PREPARE DATA AND DIRECTORIES
    path_cfg = config['path_config']
    output_dir = Path(path_cfg['final_output_path'])
    output_dir.mkdir(parents=True, exist_ok=True)
    final_output_path = output_dir / "generation_output.jsonl"

    config_save_path = output_dir / "used_config.yaml"
    
    # Write the 'config' dictionary to the new yaml file
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
    print(f"Configuration for this run saved to: {config_save_path}")
    
    input_data_path = path_cfg['input_data_path']
    all_data = [] # Initialize empty list for data

    if input_data_path.endswith(".enc"):
        print("\n--- Encrypted Input File Detected ---")
        key_path = path_cfg.get('data_secret_key_path')
        if not key_path:
            print("FATAL: Input file is encrypted (.enc), but 'data_secret_key_path' is not specified in the config.")
            sys.exit(1)
        
        # Directly get the list of dicts from the new in-memory function
        all_data = load_encrypted_jsonl(input_data_path, key_path)
        print("-------------------------------------\n")
    else:
        print(f"\nINFO: Loading unencrypted data from: {input_data_path}")
        try:
            with open(input_data_path, "r", encoding="utf-8") as f:
                all_data = [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            print(f"FATAL: Failed to load or parse data from '{input_data_path}'. Error: {e}")
            sys.exit(1)
    
    print(f"Successfully loaded {len(all_data)} records for processing.")
    
    # 3. CALCULATE OPTIMAL WORKERS & DISTRIBUTE DATA
    if not all_data:
        print("Input data file is empty. Exiting.")
        return

    model_cfg = config['model_config']
    cluster_cfg = config['cluster_config']
    gen_cfg = config['generation_config']

    is_api_model = model_cfg.get('model_type') == 'api'

    if is_api_model:
        # --- API MODEL LOGIC: SINGLE, GPU-LESS WORKER ---
        print("\n" + "="*50)
        print("--- API Model Detected: Switching to Single-Worker Mode ---")
        print("INFO: GPU settings will be ignored. A single CPU-based worker will be launched.")
        print(f"INFO: Concurrent API requests will be managed by the worker. Batch size = {gen_cfg['batch_size']}.")
        print("="*50 + "\n")
        num_workers = 1
    else:
        # --- LOCAL MODEL LOGIC: DYNAMIC GPU WORKERS ---
        if not all_data:
            print("Input data file is empty. Exiting.")
            return

        cluster_cfg = config['cluster_config']
        max_available_workers = cluster_cfg['total_gpus'] // cluster_cfg['gpus_per_worker']
        max_useful_workers = max(1, len(all_data) // gen_cfg['batch_size'])
        num_workers = min(max_available_workers, max_useful_workers)
        
        print(f"\nLocal model detected. Data-aware worker calculation:")
        print(f"  - Data size: {len(all_data)}, Batch size: {gen_cfg['batch_size']}")
        print(f"  - Max available workers (from GPUs): {max_available_workers}")
        print(f"  - Max useful workers (from data): {max_useful_workers}")
        print(f"  - DECISION: Launching {num_workers} worker(s).")

    data_chunks = np.array_split(all_data, num_workers)
    data_chunks = [chunk.tolist() for chunk in data_chunks]

    # 4. LAUNCH WORKERS AND RUN INFERENCE
    if is_api_model:
        gpus_per_actor = 0
    else:
        gpus_per_actor = config['cluster_config']['gpus_per_worker']

    workers = [
        RolloutWorker.options(
            num_cpus=1,  # Request at least 1 CPU for the worker process
            num_gpus=gpus_per_actor
        ).remote(
            worker_id=i,
            config=config # Pass the full config object
        ) for i in range(num_workers)
    ]

    print(f"\nLaunched {len(workers)} worker actor(s), each requesting {gpus_per_actor} GPU(s).")

    tasks = [
        worker.process_batch.remote(data_chunks[i])
        for i, worker in enumerate(workers)
    ]
    
    # The 'results_from_workers' will be a list of lists, e.g., [[res1, res2], [res3, res4]]
    results_from_workers = ray.get(tasks)
    print("\n--- All Workers Finished: Received results in memory ---")

    kill_actors_and_wait(workers)
    print("All RolloutWorker actors terminated and resources released.")

    # 5. FLATTEN AND WRITE RESULTS
    print("Aggregating results and writing to final output file...")
    # Flatten the list of lists into a single list of results
    all_results_flat = [item for sublist in results_from_workers for item in sublist]

    with open(final_output_path, "w", encoding="utf-8") as f_out:
        for record in tqdm(all_results_flat, desc="Writing final results"):
            f_out.write(json.dumps(record) + "\n")
    
    print(f"\n--- Inference Complete ---")
    print(f"Final results saved to: {final_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Inference using vLLM and Ray")
    parser.add_argument("--config", type=str, required=True, help="Path to the main config.yaml file.")
    parsed_args = parser.parse_args()
    main(parsed_args)