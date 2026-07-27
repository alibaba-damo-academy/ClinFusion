#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- ARGUMENT HANDLING & CONFIGURATION ---

# 1. Check if an argument was provided. If not, print a usage message and exit.
if [ "$#" -eq 0 ]; then
    echo "Error: No config file specified." >&2
    echo "Usage: $0 <path_to_config.yaml>" >&2
    exit 1
fi

# 2. Assign the first command-line argument to the CONFIG_PATH variable.
CONFIG_PATH="$1"

# 3. Check if the specified file exists and is a regular file.
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at '$CONFIG_PATH'" >&2
    exit 1
fi

API_KEY_FILE="eval/api_keys/api_key.ak"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $(pwd)/$CONFIG_PATH" >&2
    exit 1
fi

# Dynamically get the python executable from the config file.
PYTHON_EXEC=$(grep 'python_executable:' "$CONFIG_PATH" | awk '{print $2}' | tr -d '"')
if [ -z "$PYTHON_EXEC" ]; then
    echo "Error: 'python_executable:' not found or empty in $CONFIG_PATH" >&2
    exit 1
fi

PYTHON_DIR=$(dirname "$PYTHON_EXEC")
RAY_EXEC="$PYTHON_DIR/ray"
POST_PROC_SCRIPT=$(grep 'post_processing:' "$CONFIG_PATH" | awk '{print $2}' | tr -d '"')

# --- Set Defaults for Single-Node Compatibility ---
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-"6379"}


export OPENAI_API_KEY=$(cat "$API_KEY_FILE")


echo "--- Distributed Inference Environment ---"
echo "WORLD_SIZE:    $WORLD_SIZE"
echo "RANK:          $RANK"
echo "MASTER_ADDR:   $MASTER_ADDR"
echo "MASTER_PORT:   $MASTER_PORT"
echo "----------------------------------------"

# --- Force cleanup of any old Ray instances ---
echo "[Rank $RANK] Forcing cleanup of any old Ray processes..."
$RAY_EXEC stop --force || echo "No running Ray instance to stop. Continuing."

cleanup() {
    echo -e "\n\n--- INTERRUPT RECEIVED ---"
    if [[ $RANK -eq 0 ]]; then
        echo "[Rank 0] Cleaning up and shutting down Ray cluster..."
        # Give it 30 seconds to shut down gracefully.
        timeout 30s $RAY_EXEC stop --force || echo "Ray stop command timed out or failed."
    fi
    exit 1
}
trap cleanup SIGINT SIGTERM

# --- MAIN LOGIC ---
if [[ $RANK -eq 0 ]]; then
    # --- MASTER NODE (RANK 0) ---
    echo "[Rank 0] Starting Ray head node..."
    $RAY_EXEC start  --disable-usage-stats --head --port="$MASTER_PORT" --include-dashboard=false --resources='{"node_id_0": 1}'

    # If in a multi-node setup, wait for workers to join.
    if [[ $WORLD_SIZE -gt 1 ]]; then
        echo "[Rank 0] Waiting for $WORLD_SIZE nodes to join the cluster..."
        while true; do
            node_count=$($RAY_EXEC status | grep 'node' | wc -l)
            if [[ $node_count -ge $WORLD_SIZE ]]; then
                echo "[Rank 0] All $WORLD_SIZE nodes have joined the Ray cluster."
                break
            else
                echo "[Rank 0] $node_count/$WORLD_SIZE nodes have joined. Waiting..."
                sleep 10
            fi
        done
    fi

    echo "--- FINAL CLUSTER STATUS ---"
    $RAY_EXEC status
    echo "--------------------------"

    # Run the main inference script.
    echo "[Rank 0] Starting main inference script..."
    $PYTHON_EXEC -m eval.InferenceEngine.rollout --config "$CONFIG_PATH"

    if [ -n "$POST_PROC_SCRIPT" ]; then
        echo "[Rank 0] Found post-processing script: $POST_PROC_SCRIPT"
        if [ -f "$POST_PROC_SCRIPT" ]; then
            echo "[Rank 0] Running post-processing script..."
            $PYTHON_EXEC "$POST_PROC_SCRIPT" --config "$CONFIG_PATH"
            echo "[Rank 0] Post-processing script finished."
        else
            echo "Error: Post-processing script not found at '$POST_PROC_SCRIPT'" >&2
            # We shut down ray and then exit with an error.
            $RAY_EXEC stop --force
            exit 1
        fi
    else
        echo "[Rank 0] No post-processing script specified. Skipping."
    fi

    # After the script finishes, shut down the Ray cluster.
    echo "[Rank 0] Inference finished. Shutting down Ray cluster."
    $RAY_EXEC stop --force

else
    # --- WORKER NODE (RANK > 0) ---
    echo "[Rank $RANK] Starting Ray worker node, connecting to $MASTER_ADDR:$MASTER_PORT..."
    $RAY_EXEC start --address="$MASTER_ADDR:$MASTER_PORT" --resources='{"node_id_'$RANK'": 1}' --block
fi

echo "--- Launch script on Rank $RANK finished successfully. ---"