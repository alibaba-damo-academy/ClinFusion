import json
import os
from collections import Counter

# This script assumes 'constants.py' is in the same directory.
try:
    from constants import (
        VALID_ORGANS_SET, VALID_TASKS_SET,
        VALID_QUALITIES_SET, VALID_HARDNESS_LEVELS_SET,
        ORGAN_SYNONYM_MAP 
    )
except ImportError:
    print("FATAL ERROR: 'constants.py' not found or is missing definitions. Please check it.")
    exit()


def print_summary_report(success_count, failure_summary, success_summary, mapping_summary):
    """Prints a detailed analysis report after processing."""
    
    parsing_errors = failure_summary.get('parsing_errors', 0)
    validation_errors = failure_summary.get('validation_errors', 0)
    total_records = success_count + parsing_errors + validation_errors
    
    print("\n" + "="*60)
    print("  VALIDATION & ANALYSIS SUMMARY REPORT")
    print("="*60)
    print(f"Total Records Processed: {total_records}")
    print(f"  - Successful (Saved) Records:    {success_count}")
    print(f"  - Failed (Skipped) Records:      {total_records - success_count}")

    # --- 1. Failure and Cleanup Analysis ---
    print("\n" + "-"*20 + " Failure & Cleanup Analysis " + "-"*20)
    
    if parsing_errors > 0:
        print(f"Parsing/Format Errors (skipped records): {parsing_errors}")
    if validation_errors > 0:
        print(f"Validation Errors (skipped records):     {validation_errors}")
    
    # --- New: Report on Applied Mappings ---
    total_mappings = sum(mapping_summary.values())
    if total_mappings > 0:
        print(f"\nApplied {total_mappings} organ mappings:")
        for item, count in mapping_summary.most_common(10):
             # The item is a tuple like ('colon_rectum', 'colon,rectum')
            original, mapped_to = item
            print(f"  - Mapped '{original}' -> '{mapped_to}' : {count} time(s)")
        if len(mapping_summary) > 10:
            print(f"  - ... and {len(mapping_summary) - 10} more mappings.")

    unmapped_organs_count = sum(failure_summary['organs_involved'].values())
    if unmapped_organs_count > 0:
        print(f"\nDiscovered and REMOVED unmappable organs (Total: {unmapped_organs_count}):")
        for item, count in failure_summary['organs_involved'].most_common():
            print(f"  - '{item}': found and removed {count} time(s)")

    # Check for fields that caused records to be skipped
    for field_name in ['task', 'sample_quality', 'hardness']:
        if failure_summary[field_name]:
            print(f"\nDiscovered invalid '{field_name}' values (caused record skip):")
            for item, count in failure_summary[field_name].most_common():
                print(f"  - '{item}': found {count} time(s)")

    # --- 2. Success Analysis (Distribution) ---
    print("\n" + "-"*24 + " Success Analysis " + "-"*25)
    if success_count == 0:
        print("No successful records to analyze.")
    else:
        for tag, counter in success_summary.items():
            denominator = sum(counter.values()) if sum(counter.values()) > 0 else 1
            print(f"\nDistribution for '{tag}' (Total Mentions: {denominator}):")
            
            limit = 15 if tag == 'organs_involved' else None
            percentage_label = "% of total organs" if tag == 'organs_involved' else "% of successful records"
            
            if tag != 'organs_involved':
                denominator = success_count

            for item, count in counter.most_common(limit):
                percentage = (count / denominator) * 100
                print(f"  - {item:<25}: {count:<5} ({percentage:.1f}% {percentage_label})")
            if limit and len(counter) > limit:
                print(f"  - ... and {len(counter) - limit} more.")
    
    print("\n" + "="*60)


def process_and_validate_tags(input_file, output_file):
    """
    Reads tags, intelligently maps organs using ORGAN_SYNONYM_MAP, validates, 
    and writes clean data with a detailed summary.
    """
    print(f"Starting to process and validate file: {input_file}")
    
    success_count = 0
    failure_summary = {
        'parsing_errors': 0, 'validation_errors': 0,
        'organs_involved': Counter(), 'task': Counter(),
        'sample_quality': Counter(), 'hardness': Counter(),
    }
    success_summary = {
        'organs_involved': Counter(), 'task': Counter(),
        'sample_quality': Counter(), 'hardness': Counter(),
    }
    mapping_summary = Counter() # <-- To track applied mappings

    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for i, line in enumerate(f_in, 1):
                is_valid_record = True
                error_messages = []
                
                try:
                    # --- 1. PARSE JSON ---
                    full_record = json.loads(line.strip())
                    model_gen_str = full_record.get("model_generation")
                    if not model_gen_str: raise ValueError("'model_generation' is missing.")
                    start_idx, end_idx = model_gen_str.find('{'), model_gen_str.rfind('}')
                    if start_idx == -1 or end_idx == -1: raise ValueError("No JSON object found.")
                    tags_data = json.loads(model_gen_str[start_idx:end_idx+1])

                    # --- 2. MAP, VALIDATE, AND CLEAN 'organs_involved' (NEW LOGIC) ---
                    organs_str = tags_data.get("organs_involved", "")
                    raw_organs = [org.strip() for org in organs_str.replace(" ", "").split(',') if org.strip()]
                    
                    processed_organs = []
                    unmappable_organs = []

                    for org in raw_organs:
                        if org in ORGAN_SYNONYM_MAP:
                            mapped_value = ORGAN_SYNONYM_MAP[org]
                            if isinstance(mapped_value, list):
                                processed_organs.extend(mapped_value)
                                mapping_key = (org, ",".join(mapped_value))
                            else: # is a string
                                processed_organs.append(mapped_value)
                                mapping_key = (org, mapped_value)
                            mapping_summary[mapping_key] += 1
                        elif org in VALID_ORGANS_SET:
                            processed_organs.append(org)
                        elif org.upper() != "N/A":
                            unmappable_organs.append(org)
                    
                    if unmappable_organs:
                        failure_summary['organs_involved'].update(unmappable_organs)
                        # Optionally print a warning for unmapped organs
                        # print(f"[WARNING] Line {i}: Could not map or validate organs: {unmappable_organs}")

                    # Finalize the organ string: unique, sorted, or "N/A"
                    final_organs = sorted(list(set(processed_organs)))
                    validated_organs_str = ",".join(final_organs) if final_organs else "N/A"

                    # --- 3. VALIDATE OTHER FIELDS (STRICT LOGIC) ---
                    task = tags_data.get("task")
                    if task not in VALID_TASKS_SET:
                        is_valid_record = False
                        failure_summary['task'][task] += 1
                        error_messages.append(f"Invalid task: '{task}'")

                    quality = tags_data.get("sample_quality")
                    if quality not in VALID_QUALITIES_SET:
                        is_valid_record = False
                        failure_summary['sample_quality'][quality] += 1
                        error_messages.append(f"Invalid quality: '{quality}'")

                    hardness = tags_data.get("hardness")
                    if hardness not in VALID_HARDNESS_LEVELS_SET:
                        is_valid_record = False
                        failure_summary['hardness'][hardness] += 1
                        error_messages.append(f"Invalid hardness: '{hardness}'")

                    # --- 4. DECISION & OUTPUT ---
                    if is_valid_record:
                        new_record = {
                            "source": full_record.get("source"),
                            "index": full_record.get("index"),
                            "organs_involved": validated_organs_str,
                            "task": task, "sample_quality": quality, "hardness": hardness
                        }
                        f_out.write(json.dumps(new_record) + '\n')
                        
                        success_count += 1
                        success_summary['task'][task] += 1
                        success_summary['sample_quality'][quality] += 1
                        success_summary['hardness'][hardness] += 1
                        if final_organs:
                            success_summary['organs_involved'].update(final_organs)
                    else:
                        failure_summary['validation_errors'] += 1
                        print(f"[ERROR] Line {i}: Validation failed. Skipping. Reason(s): {'; '.join(error_messages)}")

                except (ValueError, json.JSONDecodeError) as e:
                    failure_summary['parsing_errors'] += 1
                    print(f"[ERROR] Line {i}: Parsing failed. Skipping. Reason: {e}")
        
        print_summary_report(success_count, failure_summary, success_summary, mapping_summary)

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'.")

if __name__ == "__main__":
    INPUT_JSONL_PATH = "/mnt/eff_nas/tangzhiwei.tzw/MedEvalKit-V3/output/generaleval_tagging_raw.jsonl"
    OUTPUT_JSONL_PATH = "general_eval_data_tags.jsonl"
    
    process_and_validate_tags(INPUT_JSONL_PATH, OUTPUT_JSONL_PATH)

