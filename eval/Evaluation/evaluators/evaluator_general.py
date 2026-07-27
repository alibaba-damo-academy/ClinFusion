import json
import os
import random
from typing import Any, Dict, List
import re
import time 
from openai import APIError, OpenAI 
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 1. API WRAPPER ---

class APIWrapper:
    """
    A generic API wrapper that conforms to the OpenAI standard.
    It uses asyncio for concurrent API requests and includes robust retry logic.
    """
    def __init__(self, model_path: str, base_url: str):
        self.model_name = model_path

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            raise ValueError("OPENAI_API_KEY environment variable not set or is a placeholder.")

        base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.max_retries = 100


    def _prepare_api_request(self, item: Dict[str, Any]) -> dict:
        """Prepares the payload for the OpenAI-compatible API."""
        messages = item.get("messages", {})
        prompt_text = messages.get("prompt")
        api_messages = [{"role": "user", "content": prompt_text}]

        final_payload = {
            "model": self.model_name,
            "messages": api_messages,
            "max_tokens": 8192,
            "temperature": 0.0
        }

        return final_payload

    def _generate_with_retry(self, payload: dict):
        """Sends a single API request with exponential backoff retry logic."""
        attempt = 0
        request_identifier = str(payload.get('messages'))[:100]

        while attempt < self.max_retries:
            try:
                completion = self.client.chat.completions.create(**payload)
                content = completion.choices[0].message.content
                if content is None:
                    raise APIError(message="Empty response content.", response=None, body=None)
                
                return completion

            except Exception as e:
                attempt += 1
                logger.warning(
                    f"Request failed on attempt {attempt}/{self.max_retries} for '{request_identifier}...'. "
                    f"Error Type: {type(e).__name__}, Details: {e}"
                )
                
                if attempt >= self.max_retries:
                    logger.error(
                        f"Max retries ({self.max_retries}) reached. Failing permanently for request: '{request_identifier}...'"
                    )
                    raise e
                
                sleep_duration = random.uniform(1.0, 5.0) * attempt
                logger.info(f"Waiting for {sleep_duration:.2f} seconds before retrying...")
                time.sleep(sleep_duration)
    
    def generate(self, batch_data: List[Dict[str, Any]]) -> List[str]:
        """
        Runs the generation process for a batch of data synchronously.
        Since this is called from a thread pool, it processes one item at a time within its thread.
        """
        output_texts = []
        # In a thread pool, each call to this 'generate' method will likely have a batch_data of size 1.
        # This loop handles it correctly regardless of batch size.
        for item in batch_data:
            payload = self._prepare_api_request(item)
            try:
                # This will either return a completion or raise an exception after retries.
                completion = self._generate_with_retry(payload)
                output_texts.append(completion.choices[0].message.content.strip())
            except Exception as e:
                # If _generate_with_retry fails, we catch the exception and return an error message.
                error_message = f"Error: API call failed after all retries. Details: {str(e)}"
                logger.error(f"Final failure for item. Appending error message: {error_message}")
                output_texts.append(error_message)
        
        return output_texts
    
def truncate_by_word(text: str, max_words: int = 200) -> str:
    """
    Truncates a string to a maximum number of words without adding a suffix.

    Args:
        text (str): The input string to truncate.
        max_words (int): The maximum number of words to keep. Defaults to 200.

    Returns:
        str: The truncated string, or the original string if it's within the
             word limit.
    """
    # Split the string into a list of words
    words = text.split()

    # If the number of words is less than or equal to the limit,
    # return the original text to preserve formatting.
    if len(words) <= max_words:
        return text

    # Slice the list of words to the max_words limit
    truncated_words = words[:max_words]

    # Join the words back into a string and return
    return ' '.join(truncated_words)


# --- 2. EVALUATOR CLASS ---

class Evaluator:
    """
    Handles the end-to-end evaluation of models for various tasks.
    It prepares prompts for a model-under-test and uses an LLM to evaluate the output.
    """
    def __init__(self, config: dict):
        """
        Initializes the Evaluator with a configuration for its internal LLM.

        Args:
            config (dict): A dictionary containing 'model_path', 'model_config',
                           and 'generation_config' for the evaluator LLM.
        """
        self.evaluator_llm = APIWrapper(
            model_path=config['model_path'],
            base_url=config['base_url']
        )

        self.use_llm_fallback = config.get("use_llm_fallback", False)
        self.use_regex_fallback = config.get("use_regex_fallback", False)

        self.max_words = 200

    def get_prompt(self, data: dict, eval_method: str) -> dict:
        """
        Prepares and augments the data with a prompt for the model being tested.

        Args:
            data (dict): The raw data entry.
            eval_method (str): The evaluation method (e.g., 'report_generation').

        Returns:
            dict: The data augmented with an 'eval_details' dictionary containing the prompt.
        """
        if eval_method == 'report_generation':
            return self._prepare_report_generation_prompt(data)
        elif eval_method == 'mcq':
            return self._prepare_mcq_prompt(data)
        elif eval_method == 'mcq_context':
            return self._prepare_mcq_context_prompt(data)
        elif eval_method == 'open':
            return self._prepare_open_prompt(data)
        else:
            raise NotImplementedError(f"Prompt generation for eval_method '{eval_method}' is not supported.")

    def eval(self, data: dict, eval_method: str) -> dict:
        """
        Performs evaluation by comparing model generation to ground truth.

        Args:
            data (dict): The data entry, now including the model's generation.
            eval_method (str): The evaluation method used.

        Returns:
            dict: The data with evaluation results and metrics added to 'eval_details'.
        """
        if eval_method == 'report_generation':
            return self._eval_report_generation(data)
        elif eval_method == 'mcq' or eval_method == 'mcq_context':
            return self._eval_mcq(data)
        elif eval_method == 'open':
            return self._eval_open(data)
        else:
            raise NotImplementedError(f"Evaluation for eval_method '{eval_method}' is not supported.")

    def _prepare_report_generation_prompt(self, data: dict) -> dict:
        """
        Prepares a context-aware prompt for report generation.

        This function uses an LLM to infer high-level clinical questions from the
        ground truth report, preventing information leakage. It then injects these
        focus areas into a new prompt for the model-under-test.
        It will raise an error if required data is missing.
        """
        # 1. Assert that required data fields are present and not empty.
        assert "images_path" in data and data["images_path"], "Input data is missing a valid 'images_path'."
        assert "ground_truth" in data and data["ground_truth"], "Input data is missing 'ground_truth' for context extraction."

        ground_truth = data.get("ground_truth")

        # 2. Design a new, more sophisticated prompt with an extremely strong emphasis on preventing information leakage.
        # It now frames the task as a critical security function before defining the summarization goal.
        context_extraction_prompt = f"""
        You are an AI with a critical security function. Your task is to process a sensitive radiology report and generate a high-level, completely neutral clinical context for another AI.

        [CRITICAL SECURITY CONSTRAINT]
        Your absolute primary goal is to **PREVENT INFORMATION LEAKAGE**. The context you generate must reveal **WHY** an exam was done and **WHERE** to look, but absolutely **NOTHING** about **WHAT WAS FOUND**. The output must be devoid of any specific findings, measurements, or conclusions from the original report. Any leakage of a finding is a complete failure of your primary function.

        [OBJECTIVE]
        Analyze the [GROUND TRUTH REPORT] and generate a two-part summary:
        1.  **[CLINICAL INDICATION]:** A single, concise sentence inferring the reason for the exam (e.g., post-operative follow-up, evaluation for abdominal pain, cancer screening).
        2.  **[AREAS OF FOCUS]:** A bulleted list of 2-5 **broad, coarse-grained** anatomical regions or systems. You MUST consolidate specific findings into general topics.

        - **Correct Transformation (High-Level, No Leakage):**
          - Findings: "Large left pleural effusion" and "Bibasilar atelectasis" -> Area of Focus: `- Lungs and pleura`
          - Findings: "10mm liver cyst" and "Fatty liver" -> Area of Focus: `- Liver and biliary system`

        - **Incorrect Transformation (Information Leakage):**
          - Do NOT output: `- Pleural effusion`, `- Atelectasis`, `- Liver cyst`. These leak specific findings and violate your primary security constraint.

        [OUTPUT RULES]
        1.  Your output MUST contain BOTH headers: `[CLINICAL INDICATION]` and `[AREAS OF FOCUS]`.
        2.  Do NOT include any preamble, explanation, or summary. Your entire response must be the structured text itself.

        ---
        [EXAMPLE OF ZERO LEAKAGE]
        ---
        [GROUND TRUTH REPORT]:
        Findings: The heart is enlarged. Lungs are clear with no pneumothorax or pleural effusion. There are sternotomy wires.
        Impression: Cardiomegaly.

        [YOUR OUTPUT]:
        [CLINICAL INDICATION]
        Evaluation of cardiothoracic structures, likely for post-surgical follow-up.

        [AREAS OF FOCUS]
        - Heart and great vessels
        - Lungs and pleura
        - Evidence of post-surgical changes

        ---
        [GROUND TRUTH REPORT]:
        {ground_truth}

        [YOUR OUTPUT]:
        """

        # 3. Call the evaluator LLM to generate the structured, high-level context.
        item_for_llm = {"messages": {"prompt": context_extraction_prompt}}
        responses = self.evaluator_llm.generate([item_for_llm])
        extracted_context_string = responses[0].strip()

        # 4. Parse the structured context into 'indication' and 'focus areas'.
        indication_part = "General radiological evaluation." # Default value
        focus_part = "" # Default value
        try:
            indication_header = "[CLINICAL INDICATION]"
            focus_header = "[AREAS OF FOCUS]"
            
            if not (indication_header in extracted_context_string and focus_header in extracted_context_string):
                raise ValueError("LLM-generated context is missing required headers.")

            parsed_indication = extracted_context_string.split(indication_header)[1].split(focus_header)[0].strip()
            parsed_focus = extracted_context_string.split(focus_header)[1].strip()
            
            if not parsed_indication or not parsed_focus:
                raise ValueError("Indication or Focus Areas are empty after parsing.")
            
            indication_part = parsed_indication
            focus_part = parsed_focus

        except (ValueError, IndexError) as e:
            # If parsing fails, fall back to a safe, generic prompt to avoid a hard crash.
            print(f"Warning: Could not parse structured context due to '{e}'. Falling back to generic prompt.")
            focus_part = "General evaluation of all visible structures." # Provide a generic but functional fallback.
            
        # 5. Construct the final, context-aware prompt for the model-under-test.
        final_prompt_instruction = (
            "You are an expert radiologist. Your task is to analyze a medical image and write a report based on the provided clinical context.\n\n"
            f"**Clinical Indication:** {indication_part}\n\n"
            "Based on this indication, you are asked to evaluate the following key areas. Your report must focus on these topics.\n\n"
            "**Areas of Focus:**\n"
            f"{focus_part}\n\n"
            "**CRITICAL INSTRUCTION:** Your generated report MUST ONLY describe findings related to the **Areas of Focus** listed above. "
            "Do NOT include any findings, observations, or comments about areas or structures NOT mentioned in the list. "
            "Your response will be evaluated based on how accurately you perform a focused examination based on the clinical context.\n\n"
            "Generate a complete report with two sections, 'Findings' and 'Impression'. Your output must follow this format EXACTLY:\n"
            "Findings: [Text of your findings here, covering ONLY the focus areas]\n\n"
            "Impression: [Text of your impression here, summarizing ONLY the focus areas]"
        )

        # 6. Populate the eval_details dictionary with the new prompt and extracted context.
        data["eval_details"] = {
            "messages": {
                "prompt": final_prompt_instruction,
                "image": data["images_path"]
            },
            "extracted_context": extracted_context_string,
            "parsed_indication": indication_part,
            "parsed_focus_areas": focus_part
        }
        
        return data

    def _eval_report_generation(self, data: dict) -> dict:
        """
        Handles the LLM-based evaluation for report generation, assessing all key findings (both positive and negative).
        """
        ground_truth = data.get("ground_truth", "")
        try:
            model_generation = data["eval_details"]["model_generation"]
        except KeyError:
            # Asserting the key exists, as per earlier requirements.
            raise KeyError("Input data for evaluation is missing 'model_generation' key.")

        parsed_answer = model_generation.split("</think>")[-1].strip()
        parsed_answer = truncate_by_word(parsed_answer, self.max_words)

        # This completely new prompt guides the LLM to evaluate ALL key findings, not just abnormalities.
        eval_prompt = f"""
        You are an expert radiologist acting as a strict evaluator. Your task is to compare a model-generated report against a ground truth report, focusing exclusively on how well it handles **abnormal findings**.

        [GROUND TRUTH REPORT]:
        ---
        {ground_truth}
        ---

        [MODEL GENERATED REPORT]:
        ---
        {parsed_answer}
        ---

        Now, analyze the reports and categorize findings based on the following strict definitions.

        **Category Definitions:**

        1.  `[MATCHED_ABNORMALITY]`:
            -   **Definition:** The Ground Truth mentions an abnormality, and the Model Report correctly identifies the same abnormality. Minor differences in wording are acceptable.
            -   **Example:** GT: "Cardiomegaly." Model: "The heart is enlarged." -> This is a MATCH.

        2.  `[MISSED_ABNORMALITY]`:
            -   **Definition:** The Ground Truth mentions an abnormality, but the Model Report either completely fails to mention it OR incorrectly states that the area is normal.
            -   **Example 1 (Omission):** GT: "Bibasilar atelectasis." Model: Does not mention the lung bases. -> This is a MISS.
            -   **Example 2 (Contradiction):** GT: "Pleural effusion is present." Model: "No pleural effusion." -> This is a MISS.

        3.  `[HALLUCINATED_ABNORMALITY]`:
            -   **Definition:** The Model Report mentions an abnormality that is **not** present in the Ground Truth. This includes cases where the Ground Truth explicitly states an area is normal, or where the Ground Truth simply does not mention the finding at all (making it unverifiable).
            -   **Example 1 (Contradiction):** GT: "Lungs are clear." Model: "There is a right lower lobe opacity." -> This is a HALLUCINATION.
            -   **Example 2 (Unverifiable):** GT: Does not mention the liver. Model: "Hepatomegaly is noted." -> This is a HALLUCINATION.

        **Your Task & Output Rules:**
        -   Your entire analysis must focus on ABNORMAL findings. Do not evaluate routine negative statements.
        -   **CRITICAL RULE:** Your response MUST contain all three category headers, in this exact order: `[MATCHED_ABNORMALITY]`, `[MISSED_ABNORMALITY]`, `[HALLUCINATED_ABNORMALITY]`.
        -   If a category has no findings to list, you **must still include its header** on a new line, leaving it empty underneath.
        -   Under each header with findings, concisely list them.
        -   Do NOT add any extra explanations, introductions, or summaries. Your response must begin directly with the first header.

        **Examples of Final Output Format:**

        **Example 1: Mixed Findings**
        (This demonstrates a case with an empty category).

        [MATCHED_ABNORMALITY]
        Cardiomegaly
        [MISSED_ABNORMALITY]
        Bibasilar atelectasis
        Pleural effusion
        [HALLUCINATED_ABNORMALITY]

        ---
        **Example 2: A Perfect "Normal" Case**
        (This shows the required output when both reports are normal and there are no abnormalities to list).

        [MATCHED_ABNORMALITY]
        [MISSED_ABNORMALITY]
        [HALLUCINATED_ABNORMALITY]
        """
        
        item_for_llm = {"messages": {"prompt": eval_prompt}}
        responses = self.evaluator_llm.generate([item_for_llm])
        llm_output_str = responses[0]

        try:
            llm_result = self._parse_structured_text_output(llm_output_str, 
                                                            list_headers=["MATCHED_ABNORMALITY", "MISSED_ABNORMALITY", "HALLUCINATED_ABNORMALITY"])

            tp = len(llm_result.get('matched_abnormality', []))
            fp = len(llm_result.get('hallucinated_abnormality', []))
            fn = len(llm_result.get('missed_abnormality', []))

            if tp == 0 and fp == 0 and fn == 0:
                # This scenario is now always treated as a perfect score.
                precision, recall, f1_score = 1.0, 1.0, 1.0
            else:
                # Standard calculation for all other cases.
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            llm_result["parsed_answer"] = parsed_answer
            data["eval_details"]["evaluation_results"] = llm_result
            data["eval_details"]["evaluation_metrics"] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1_score, 4)
            }

        except (json.JSONDecodeError, ValueError) as e:
            data["eval_details"]["judge_error"] = f"Could not parse LLM Judger output: {e}"
            data["eval_details"]["raw_judger_output"] = llm_output_str
            data["eval_details"]["evaluation_metrics"] = {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        return data

    def _prepare_mcq_prompt(self, data: dict) -> dict:
        """Handles prompt creation for multiple-choice questions with a highly restrictive format."""
        question = data.get("question", "")
        options = data.get("options", {})

        options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
        valid_option_keys = ", ".join(options.keys())

        if "images_path" in data and data["images_path"]:
            start_str = "Answer the following multiple-choice question based on the provided image(s).\n\n"
        else:
            start_str = "Answer the following multiple-choice question.\n\n"

        prompt_instruction = (
            f"{start_str}"
            f"Question: {question}\n\n"
            f"Options:\n{options_str}\n\n"
            f"""---
    **CRITICAL INSTRUCTIONS FOR AUTOMATED GRADING:**
    This is a single-choice question. You must select **ONE AND ONLY ONE** option.

    1.  Your entire output must be ONLY the **single uppercase letter** of the correct option. The valid letters for this question are: {valid_option_keys}.
    2.  This single letter MUST be enclosed in `<answer></answer>` tags.
    3.  DO NOT provide any reasoning, explanations, or any text before or after the tags.

    **Correct Format Example:**
    If 'A' is the correct option:
    <answer>A</answer>

    **Incorrect Format Examples (THESE WILL FAIL AUTOMATED GRADING):**
    -   `The correct answer is <answer>A</answer>` (Contains extra text)
    -   `A` (Missing tags)
    -   `<answer>A: Option Text</answer>` (Contains option text, not just the letter)
    -   `<answer>G2</answer>` (Contains the option's value instead of its letter key)
    -   `<answer>BDEFGI</answer>` (**INVALID**: Contains multiple letters. You must choose only one.)

    Provide your single-letter answer now.
    """
        )

        messages = {"prompt": prompt_instruction}
        if "images_path" in data and data["images_path"]:
            messages["image"] = data["images_path"]

        data["eval_details"] = {"messages": messages}
        return data

    def _prepare_mcq_context_prompt(self, data: dict) -> dict:
        """Handles prompt creation for multiple-choice questions with context and a highly restrictive format."""
        question = data.get("question", "")
        context = data.get("context", "")
        options = data.get("options", {})

        options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
        valid_option_keys = ", ".join(options.keys())

        if "images_path" in data and data["images_path"]:
            start_str = "Answer the multiple-choice question based on the provided image(s) and context.\n\n"
        else:
            start_str = "Answer the multiple-choice question based on the provided context.\n\n"

        prompt_instruction = (
            f"{start_str}"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{options_str}\n\n"
            f"""---
**CRITICAL INSTRUCTIONS FOR AUTOMATED GRADING:**
This is a single-choice question. You must select **ONE AND ONLY ONE** option.

1.  Your entire output must be ONLY the **single uppercase letter** of the correct option. The valid letters for this question are: {valid_option_keys}.
2.  This single letter MUST be enclosed in `<answer></answer>` tags.
3.  DO NOT provide any reasoning, explanations, or any text before or after the tags.

**Correct Format Example:**
If 'A' is the correct option:
<answer>A</answer>

**Incorrect Format Examples (THESE WILL FAIL AUTOMATED GRADING):**
-   `The correct answer is <answer>A</answer>` (Contains extra text)
-   `A` (Missing tags)
-   `<answer>A: Option Text</answer>` (Contains option text, not just the letter)
-   `<answer>G2</answer>` (Contains the option's value instead of its letter key)
-   `<answer>BDEFGI</answer>` (**INVALID**: Contains multiple letters. You must choose only one.)

Provide your single-letter answer now.
"""
        )

        messages = {"prompt": prompt_instruction}
        if "images_path" in data and data["images_path"]:
            messages["image"] = data["images_path"]

        data["eval_details"] = {"messages": messages}
        return data

    def _eval_mcq(self, data: dict) -> dict:
        """
        Handles the direct evaluation for multiple-choice questions with strict parsing.
        The model's output MUST be in the format <answer>X</answer> where X is a valid option key.
        """
        ground_truth = str(data.get("ground_truth", "")).strip().upper()
        try:
            model_generation = data["eval_details"]["model_generation"]
        except KeyError:
            data["eval_details"]["evaluation_error"] = "Input data missing 'model_generation' key."
            data["eval_details"]["evaluation_metrics"] = {"accuracy": 0.0}
            return data

        parsed_answer = model_generation.split("</think>")[-1].strip()
        parsed_answer = truncate_by_word(parsed_answer, self.max_words)
        
        try:
            parsed_answer_str = parsed_answer.strip()
            match = re.fullmatch(r"<answer>(.*?)</answer>", parsed_answer_str, re.DOTALL)
            
            if not match:
                error_msg = "Failed to parse answer. The model output did not contain or did not exactly match the required <answer>...</answer> format."
                raise ValueError(error_msg) # 
            
            extracted_content = match.group(1).strip()
            valid_options = data.get("options", {}).keys()

            if extracted_content in valid_options:
                is_correct = (extracted_content == ground_truth)
                accuracy = 1.0 if is_correct else 0.0
                
                data["eval_details"]["evaluation_results"] = {
                    "parsed_answer": extracted_content,
                    "is_correct": is_correct
                }
                data["eval_details"]["evaluation_metrics"] = {"accuracy": accuracy}
            else:
                error_msg = f"Parsed answer '{extracted_content}' is not a valid option key. Valid options are: {list(valid_options)}."
                raise ValueError(error_msg) 

        except ValueError as error: 
            data["eval_details"]["format_error"] = str(error) 

            if self.use_llm_fallback:
                logger.info(f"MCQ format error for item. Attempting LLM fallback.")
                data["eval_details"]["fallback_attempted"] = "llm"
                return self._eval_mcq_fallback(data)
            
            # 2. If LLM fallback is not enabled, try the regex fallback.
            elif self.use_regex_fallback:
                logger.info(f"MCQ format error for item. Attempting Regex fallback.")
                data["eval_details"]["fallback_attempted"] = "regex"
                return self._eval_mcq_regex_fallback(data)
            
            # 3. If no fallbacks are enabled, fail the evaluation for this item.
            else:
                logger.warning(f"MCQ format error and no fallback enabled. Marking as incorrect.")
                data["eval_details"]["evaluation_metrics"] = {"accuracy": 0.0}
                if "evaluation_results" not in data["eval_details"]:
                    data["eval_details"]["evaluation_results"] = {}
                data["eval_details"]["evaluation_results"]["judged_choice"] = "FormatError"

        return data

    def _eval_mcq_fallback(self, data: dict) -> dict:
        """
        Handles the LLM-based evaluation for multiple-choice questions. Ignored any formatting errors.
        """

        ground_truth = str(data.get("ground_truth", "")).strip().upper()
        try:
            model_generation = data["eval_details"]["model_generation"]
        except KeyError:
            data["eval_details"]["evaluation_error"] = "Input data missing 'model_generation' key."
            data["eval_details"]["evaluation_metrics"] = {"accuracy": 0.0}
            return data

        parsed_answer = model_generation.split("</think>")[-1].strip()
        parsed_answer = truncate_by_word(parsed_answer, self.max_words)

        options = data.get("options", {})
        options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
    
        valid_option_keys = list(options.keys())

        question = data.get("question", "")

        # This detailed prompt guides the LLM to act as a strict judge.
        eval_prompt = f"""
        You are an impartial judge. Your task is to determine which single option the "MODEL'S ANSWER" selected for the given multiple-choice question.

        [QUESTION]:
        {question}

        [OPTIONS]:
        {options_str}

        [MODEL'S ANSWER]:
        {parsed_answer}
        ---

        Your task is to identify which option from {valid_option_keys} the model chose.
        Provide your verdict in the following structured format. Do not use JSON.
        - `[CHOICE]`: On the next line, write the single letter of the option the model selected (e.g., A, B, C). If the model's answer is ambiguous, nonsensical, or does not select a valid option, write `None`.
        - `[REASONING]`: On the next line, write a brief explanation for your decision.

        Example 1 (Clear Choice):
        [CHOICE]
        B
        [REASONING]
        The model explicitly stated that 'B' is the correct answer.

        Example 2 (Ambiguous/No Choice):
        [CHOICE]
        None
        [REASONING]
        The model provided a general discussion without committing to a specific option.
        """
        
        item_for_llm = {"messages": {"prompt": eval_prompt}}
        responses = self.evaluator_llm.generate([item_for_llm])
        llm_output_str = responses[0]

        try:
            llm_result = self._parse_structured_text_output(llm_output_str)

            if "choice" not in llm_result:
                raise ValueError("Judger output missing 'choice' key.")
                
            judged_choice = llm_result.get("choice", "None").strip().upper()

            is_correct = (judged_choice == ground_truth)
            accuracy = 1.0 if is_correct else 0.0

            # Merge the new results with existing details
            if "evaluation_results" not in data["eval_details"]:
                data["eval_details"]["evaluation_results"] = {}
                
            data["eval_details"]["evaluation_results"].update({
                "judged_choice": judged_choice,
                "llm_judger_reasoning": llm_result.get("reasoning", ""), # Specific key
                "is_correct": is_correct
            })
            data["eval_details"]["evaluation_metrics"] = {"accuracy": accuracy}
            # Add a specific flag to indicate LLM fallback was successful
            data["eval_details"]["llm_fallback_used"] = True

        except (json.JSONDecodeError, ValueError) as e:
            data["eval_details"]["judge_error"] = f"Could not parse LLM Judger fallback output: {e}"
            data["eval_details"]["raw_judger_output"] = llm_output_str
            data["eval_details"]["evaluation_metrics"] = {"accuracy": 0.0}
            # Still flag that LLM fallback was used, even if it failed, for traceability
            data["eval_details"]["llm_fallback_used"] = True

        return data
    

    def _parse_mcq_with_regex(self, response_text: str, valid_options: List[str]) -> (str, str):
        """
        Parses a free-form text response to find a multiple-choice answer using regex.
        This avoids similarity metrics and uses explainable rules.

        Args:
            response_text (str): The model's generated text.
            valid_options (List[str]): A list of valid option keys (e.g., ['A', 'B', 'C']).

        Returns:
            A tuple of (found_choice, reason). `found_choice` is the option letter or None.
            `reason` explains how the choice was found.
        """
        # Normalize the text and the valid options
        text = response_text.strip().upper()
        valid_options = [opt.upper() for opt in valid_options]
        
        # Rule 1: Look for explicit statements like "The answer is A", "Choice: B", etc.
        # This regex looks for a keyword, optional punctuation, and then a valid option letter.
        # \b is a word boundary to prevent matching 'A' in 'FAIL'.
        keywords = ['answer is', 'choice is', 'option is', 'is:', 'is']
        for keyword in keywords:
            pattern = re.compile(f"\\b{re.escape(keyword)}\\s*[:\\-]?\\s*\\b({'|'.join(valid_options)})\\b")
            match = pattern.search(text)
            if match:
                choice = match.group(1)
                return choice, f"Found choice via keyword '{keyword}'"
        
        # Rule 2: Check if the response STARTS with a single valid option letter.
        # This handles cases like "A. ..." or "B) ..."
        pattern = re.compile(r"^\s*\(?(" + "|".join(valid_options) + r")\b")
        match = pattern.match(text)
        if match:
            choice = match.group(1)
            return choice, "Found choice as the first character in the response"

        # Rule 3: If no specific pattern matches, check if ONLY ONE valid option is mentioned.
        # This is an unambiguous way to determine the choice without similarity scores.
        found = []
        for option in valid_options:
            if re.search(r'\b' + option + r'\b', text):
                found.append(option)
        
        if len(found) == 1:
            return found[0], "Found a single, unambiguous mention of a valid option key"
        
        if len(found) > 1:
            return None, f"Response is ambiguous; mentions multiple options: {', '.join(found)}"

        return None, "No specific choice could be deterministically parsed from the response"

    def _eval_mcq_regex_fallback(self, data: dict) -> dict:
        """
        Handles regex-based evaluation for MCQs when strict parsing fails.
        """
        ground_truth = str(data.get("ground_truth", "")).strip().upper()
        model_generation = data["eval_details"]["model_generation"]

        parsed_answer = model_generation.split("</think>")[-1].strip()
        parsed_answer = truncate_by_word(parsed_answer, self.max_words)
        valid_options = list(data.get("options", {}).keys())

        # Use the new regex parser to get a choice and a reason
        judged_choice, reason = self._parse_mcq_with_regex(parsed_answer, valid_options)

        is_correct = False
        accuracy = 0.0
        if judged_choice:
            is_correct = (judged_choice == ground_truth)
            accuracy = 1.0 if is_correct else 0.0

        # Update evaluation details with results from the regex fallback
        if "evaluation_results" not in data["eval_details"]:
            data["eval_details"]["evaluation_results"] = {}
        
        data["eval_details"]["evaluation_results"].update({
            "answer_to_parse": parsed_answer,
            "judged_choice": judged_choice,
            "regex_judger_reasoning": reason,
            "is_correct": is_correct
        })
        data["eval_details"]["evaluation_metrics"] = {"accuracy": accuracy}
        data["eval_details"]["regex_fallback_used"] = True # Add specific flag

        return data
    

    def _prepare_open_prompt(self, data: dict) -> dict:
        """Handles prompt creation for open-ended questions."""
        question = data.get("question", "")

        if "images_path" not in data:
            start_str = "Analyze the following question and provide a concise, direct answer.\n\n"
        else:
            start_str = "Analyze the following image(s) and provide a concise, direct answer to the following question.\n\n"

        prompt_instruction = (
            f"{start_str}"
            f"Question: {question}\n\n"
            "Answer:"
        )

        messages = {"prompt": prompt_instruction}
        if "images_path" in data and data["images_path"]:
            messages["image"] = data["images_path"]

        data["eval_details"] = {"messages": messages}

        return data

    def _eval_open(self, data: dict) -> dict:
        """Handles the LLM-based evaluation for open-ended questions."""
        question = data.get("question", "")
        ground_truth = data.get("ground_truth", "")
        try:
            model_generation = data["eval_details"]["model_generation"]
        except KeyError:
            data["eval_details"]["evaluation_error"] = "Input data missing 'model_generation' key."
            return data

        parsed_answer = model_generation.split("</think>")[-1].strip()
        parsed_answer = truncate_by_word(parsed_answer, self.max_words)

        # This detailed prompt guides the LLM to act as a strict judge.
        eval_prompt = f"""
        Your task is to determine whether the user's answer is correct based on the provided questions and standard answers (for example, if the user expresses a similar meaning to the standard answer, or another interpretation of the standard answer, it is considered correct.)

        [QUESTION]:
        {question}

        [GROUND TRUTH ANSWER]:
        {ground_truth}

        [MODEL'S ANSWER]:
        {model_generation}
        ---

        Provide your verdict in the following structured format. Do not use JSON.
        - `[JUDGEMENT]`: Write `correct` or `incorrect` on the next line.
        - `[REASONING]`: Write a brief explanation for your decision on the next line.

        Example of the required output format:
        [JUDGEMENT]
        correct
        [REASONING]
        The standard answer is right, and the user's answer is right frontal lobe, they express the same meaning, so it is correct.
        """
        
        item_for_llm = {"messages": {"prompt": eval_prompt}}
        responses = self.evaluator_llm.generate([item_for_llm])
        llm_output_str = responses[0]

        try:
            llm_result = self._parse_structured_text_output(llm_output_str)

            # Ensure the judger's output has the required key
            if "judgement" not in llm_result:
                raise ValueError("Judger output missing 'judgement' key.")

            judgement = llm_result.get("judgement").lower()
            accuracy = 1.0 if judgement == "correct" else 0.0

            llm_result["parsed_answer"] = parsed_answer
            data["eval_details"]["evaluation_results"] = llm_result
            data["eval_details"]["evaluation_metrics"] = {
                "accuracy": accuracy
            }

        except (json.JSONDecodeError, ValueError) as e:
            data["eval_details"]["judge_error"] = f"Could not parse LLM Judger output: {e}"
            data["eval_details"]["raw_judger_output"] = llm_output_str
            data["eval_details"]["evaluation_metrics"] = {"accuracy": 0.0}

        return data
    
    def _parse_structured_text_output(self, llm_output: str, list_headers: list = []) -> dict:
        """
        Parses a structured text output from an LLM into a dictionary.

        The expected format uses headers like [HEADER] followed by content.
        - For list-based headers (MATCHED, ERROR, etc.), it expects one item per line.
        - For string-based headers (JUDGEMENT, REASONING), it expects the content on the next line.

        Args:
            llm_output (str): The raw string output from the LLM.

        Returns:
            dict: The parsed key-value data.
        """
        output_dict = {}
        
        # A robust regex to find all [HEADER] sections and their content.
        # It captures the header name and the content until the next header or end of string.
        pattern = re.compile(r'\[([A-Z_]+)\]\s*(.*?)(?=\s*\[[A-Z_]+\]|\Z)', re.DOTALL)
        
        matches = pattern.findall(llm_output)
        if not matches:
            raise ValueError("Could not find any structured headers like [HEADER] in the output.")

        for header, content in matches:
            header = header.strip().upper()
            content = content.strip()
            key = header.lower()

            if header in list_headers:
                # Split by newline, strip each line, and filter out empty lines
                items = [line.strip() for line in content.split('\n') if line.strip()]
                # Optional: remove leading bullet points if they exist
                items = [re.sub(r'^[*-]\s*', '', item) for item in items]
                output_dict[key] = items
            else: # For headers like JUDGEMENT, REASONING
                # The content is the whole block, cleaned up
                output_dict[key] = content.replace('\n', ' ').strip()
                
        return output_dict
