import json
import os
# Import the shared constants
from constants import (
    VALID_ORGANS, CLASSIFICATION_SYSTEM
)

def prepare_prompts_for_tagging(input_file, output_file):
    print(f"Starting to process file: {input_file}")
    
    # Format the valid organs list for injection into the prompt
    valid_organs_str = ", ".join(VALID_ORGANS)
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for i, line in enumerate(f_in):
            sample = json.loads(line.strip())
            
            content_parts = []
            if sample.get('question'):
                content_parts.append(f"Question: {sample['question']}")
            else:
                content_parts.append("Task: Generate a radiological report based on the provided image.")

            if sample.get('options'):
                options_str = "\n".join([f"  {k}: {v}" for k, v in sample['options'].items()])
                content_parts.append(f"Options:\n{options_str}")
            
            if sample.get('context'):
                content_parts.append(f"Context: {sample['context']}")

            if sample.get('ground_truth'):
                content_parts.append(f"Ground Truth Answer/Report: {sample['ground_truth']}")

            if sample.get('images_path') and sample['images_path']:
                content_parts.append("Image: Please see the provided image(s).")

            sample_content_str = "\n\n".join(content_parts)

            full_prompt = f"""You are an expert medical data analyst. Your task is to analyze the following medical sample and generate four specific tags based on the provided information (THE DATA SAMPLE TO ANALYZE) and classification system (CLASSIFICATION SYSTEM REFERENCE).

**--- INSTRUCTIONS & DEFINITIONS ---**

1.  **`organs_involved`**: Analyze the sample and identify relevant anatomical structures. Your output for this key is strictly controlled by the rules below.

    **--- CRITICAL RULES FOR ORGAN TAGGING ---**
    *   **RULE 1: STRICT SELECTION.** Your output MUST be a comma-separated list of items selected ONLY from the `VALID_ORGANS` list provided below. Do not include spaces around the commas.
    *   **RULE 2: GENERALIZE, DON'T SUBSTITUTE.** If a specific anatomical part (e.g., "femur," "temporal lobe") is mentioned, you must check if its broader anatomical system (e.g., bone, brain) is on the VALID_ORGANS list. If the system is on the list, use it. If neither the specific part nor its system is on the list, you MUST NOT include it. Do not substitute one organ for another (e.g., do not map "kidney" to "spleen").
    *   **RULE 3: N/A USAGE.** If, after applying Rule 2, NO organs from the `VALID_ORGANS` list are relevant to the sample, your output for this key MUST be exactly "N/A".


    **--- EXAMPLES OF CORRECT LOGIC ---**
    *   If the sample is about a "liver tumor and pancreatitis," your output should be: `liver,pancreas`
    *   If the sample is about a "fracture of the femur," your output MUST be: `bone` (because `bone` is on the list, but `femur` is not).
    *   If the sample is about an "MRI of the brain showing a tumor in the temporal lobe," your output for organs MUST be: `brain` (because `brain` is on the list, but `temporal_lobe` is not).
    *   If the sample is about an "aneurysm of the aorta," your output should be: `aorta` (because `aorta` is a specific and valid tag on the list).
    *   If the sample is a general knowledge question about physiology with no specific organ focus, your output MUST be `N/A`.

    **VALID_ORGANS**: `{valid_organs_str}`

2.  **`task`**: Classify the sample by selecting the single most appropriate task from the CLASSIFICATION SYSTEM REFERENCE below. Your output for `task` must be one of {{organ_understanding, lesion_understanding, modality_recognition, change_comparison, examination_planning, etiological_diagnosis, disease_staging, drug_usage, protocol_design, treatment_response, outcome_prediction, chronic_disease_management, dietary_recommendation, basic_science_knowledge, clinical_knowledge}}.

3.  **`sample_quality`**: Assess the sample for **objective data flaws**, not medical difficulty. Use one of {{good, mild, bad}}.
    *   **`bad`**: The sample is **unusable** due to critical flaws. Assign `bad` if you see any of these:
        *   **Text/Image Contradiction:** Text mentions something not in the image (e.g., a "red box").
        *   **Unreadable Image:** The image is black, corrupted, or completely blurry.
    *   **`mild`**: The sample is **usable** but has minor issues like slight image blur or non-critical typos.
    *   **`good`**: The sample has no obvious flaws listed above.

4.  **`hardness`**: Evaluate the cognitive difficulty of the task. Use one of {{easy, mild, hard}}.
    *   `easy`: Simple knowledge retrieval or direct observation.
    *   `mild`: Requires minor reasoning, multi-choice selection with clear distractors, or straightforward application of knowledge.
    *   `hard`: Requires substantial reasoning, synthesizing information from multiple sources, differentiating between subtle options, or complex report generation.

**--- Specific Guidance for Report Generation Tasks ---**

If the sample is a 'report generation' task, you MUST base your `scenario`, `task`, and `hardness` classifications on the `Ground Truth Answer/Report`.
*   **`task` for Reports:** The `task` depends on the report's main focus (e.g., `lesion_understanding` for abnormalities, `organ_understanding` for normal descriptions).
*   **`hardness` for Reports:** `easy` for normal reports; `mild` for common abnormalities; `hard` for subtle patterns or complex findings.

**--- CLASSIFICATION SYSTEM REFERENCE ---**
{CLASSIFICATION_SYSTEM}

**--- THE DATA SAMPLE TO ANALYZE ---**
{sample_content_str}

**--- REQUIRED OUTPUT FORMAT ---**
Your response MUST be a single, valid JSON object with the four requested keys. Do not add any other text.

Example:
{{
  "organs_involved": "lung,heart,pleura",
  "task": "lesion_understanding",
  "sample_quality": "good",
  "hardness": "mild"
}}
"""
     
            messages = {"prompt": full_prompt}
            if "images_path" in sample:
                messages["images"] = sample['images_path']
            output_record = {
                "source": sample.get('source'),
                "index": sample.get('index'),
                "messages": messages
            }
            
            f_out.write(json.dumps(output_record) + '\n')

    print(f"\nProcessing complete. Output saved to: {output_file}")


if __name__ == "__main__":
    INPUT_JSONL_PATH = "/mnt/eff_nas/tangzhiwei.tzw/MedEvalKit-V3/Evaluation/datasets/general_eval_data.jsonl"
    OUTPUT_JSONL_PATH = "prompts_for_tagging.jsonl"

    if os.path.exists(INPUT_JSONL_PATH):
        prepare_prompts_for_tagging(INPUT_JSONL_PATH, OUTPUT_JSONL_PATH)
    else:
        print(f"Error: Input file not found at '{INPUT_JSONL_PATH}'.")

