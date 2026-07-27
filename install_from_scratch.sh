#!/bin/bash
set -e

# ============================================================
# ClinFusion Environment Installer
# Usage:
#   source install_env.sh        (install + activate)
#   bash install_env.sh          (install only, print activation hint)
# ============================================================

ENV_DIR="/tmp/hangjie.yhj/envs/qwen3-vl"
REQ_FILE="requirements.txt"

MAX_RETRIES=3
RETRY_DELAY=5

retry() {
    local desc="$1"
    shift
    local attempt=1
    local delay=$RETRY_DELAY
    while [ $attempt -le $MAX_RETRIES ]; do
        if "$@"; then
            return 0
        fi
        if [ $attempt -eq $MAX_RETRIES ]; then
            echo "FAILED: $desc after $MAX_RETRIES attempts."
            return 1
        fi
        echo "WARN: $desc failed (attempt $attempt/$MAX_RETRIES), retrying in ${delay}s..."
        sleep $delay
        delay=$((delay * 2))
        attempt=$((attempt + 1))
    done
}

# Step 1: Ensure uv is available
if ! command -v uv &>/dev/null; then
    echo "[1/3] Installing uv package manager..."
    retry "pip install uv" pip install uv
else
    echo "[1/3] uv already installed, skipping."
fi

# Step 2: Create virtual environment (skip if already exists)
if [ -f "$ENV_DIR/bin/python" ]; then
    echo "[2/3] Environment already exists at $ENV_DIR, skipping creation."
else
    echo "[2/3] Creating virtual environment at $ENV_DIR ..."
    uv venv "$ENV_DIR" --python 3.11
fi

# Step 3: Validate local whl paths in requirements, then install
echo "[3/3] Installing dependencies from $REQ_FILE ..."
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    # Check lines that look like absolute file paths
    if [[ "$line" == /* && ! -f "$line" ]]; then
        echo "ERROR: Local whl not found: $line"
        echo "       Please check the file path in $REQ_FILE"
        exit 1
    fi
done < "$REQ_FILE"

retry "uv pip install" uv pip install -r "$REQ_FILE" --python "$ENV_DIR/bin/python"

echo ""
echo "============================================================"
echo "Environment installed at: $ENV_DIR"
echo ""

# Activate if sourced, otherwise print hint
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    source "$ENV_DIR/bin/activate"
    echo "Environment ACTIVATED. Python: $(which python)"
else
    echo "To activate, run:"
    echo "  source $ENV_DIR/bin/activate"
fi
echo "============================================================"


### Version 1:
# #!/bin/bash                                                                                                              
# set -e                                                                                                                   
                                                                                                                        
# # ============================================================                                                           
# # ClinFusion Environment Installer                                                                                       
# # Usage:                                                                                                                 
# #   source install_env.sh        (install + activate)                                                                    
# #   bash install_env.sh          (install only, print activation hint)                                                   
# # ============================================================                                                           
                                                                                                                        
# ENV_DIR="/tmp/hangjie.yhj/envs/qwen3-vl"
# # ENV_DIR="/mnt/develop_hz/hangjie.yhj/envs/qwen3-vl"
# # SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# REQ_FILE="/mnt/faster_nas/hangjie.yhj/code_medical/ClinFusion-Public/requirements_clean.txt"                                                                            
                                                                                                                        
# # Step 1: Ensure uv is available                                                                                         
# if ! command -v uv &>/dev/null; then                                                                                     
#     echo "[1/3] Installing uv package manager..."                                                                        
#     pip install uv                                                                                                       
# else                                                                                                                     
#     echo "[1/3] uv already installed, skipping."                                                                         
# fi                                                                                                                       
                                                                                                                        
# # Step 2: Create virtual environment (skip if already exists)                                                            
# if [ -f "$ENV_DIR/bin/python" ]; then                                                                                    
#     echo "[2/3] Environment already exists at $ENV_DIR, skipping creation."                                              
# else                                                                                                                     
#     echo "[2/3] Creating virtual environment at $ENV_DIR ..."                                                            
#     uv venv "$ENV_DIR" --python 3.11                                                                                     
# fi                                                                                                                       
                                                                                                                        
# # Step 3: Validate local whl paths in requirements, then install                                                         
# echo "[3/3] Installing dependencies from $REQ_FILE ..."                                                                  
# while IFS= read -r line || [ -n "$line" ]; do                                                                            
#     # Skip empty lines and comments                                                                                      
#     [[ -z "$line" || "$line" =~ ^# ]] && continue                                                                        
#     # Check lines that look like absolute file paths                                                                     
#     if [[ "$line" == /* && ! -f "$line" ]]; then                                                                         
#         echo "ERROR: Local whl not found: $line"                                                                         
#         echo "       Please check the file path in $REQ_FILE"                                                            
#         exit 1                                                                                                           
#     fi                                                                                                                   
# done < "$REQ_FILE"                                                                                                       
                                                                                                                        
# uv pip install -r "$REQ_FILE" --python "$ENV_DIR/bin/python"                                                             
                                                                                                                        
# echo ""                                                                                                                  
# echo "============================================================"                                                      
# echo "Environment installed at: $ENV_DIR"                                                                                
# echo ""                                                                                                                  
                                                                                                                        
# # Activate if sourced, otherwise print hint                                                                              
# if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then                                                                             
#     source "$ENV_DIR/bin/activate"                                                                                       
#     echo "Environment ACTIVATED. Python: $(which python)"                                                                
# else                                                                                                                     
#     echo "To activate, run:"                                                                                             
#     echo "  source $ENV_DIR/bin/activate"                                                                                
# fi                                                                                                                       
# echo "============================================================"    