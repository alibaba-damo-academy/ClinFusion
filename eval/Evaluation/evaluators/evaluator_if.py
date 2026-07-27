import re
from typing import Any, Dict

FORMAT_INSTRUCTIONS = {
    "mcq": [
        {
            "id": "mcq_xml_tags_strict",
            "prompt": "Your entire output must be ONLY the single uppercase letter of the correct option, enclosed in `<answer></answer>` tags. Do not provide any other text. Example: `<answer>A</answer>`",
            "regex": r"<answer>[A-Z]</answer>",
        },
        {
            "id": "mcq_cot_explanation_tags",
            "prompt": "First, provide a concise explanation of your reasoning inside `<explanation>` tags. Then, on a new line, provide the single uppercase letter of your choice inside `<answer>` tags. Example:\n<explanation>The image shows feature X, which rules out options A and C.</explanation>\n<answer>B</answer>",
            "regex": r"<explanation>.*</explanation>\n<answer>[A-Z]</answer>",
        },
        {
            "id": "mcq_cot_thinking_step",
            "prompt": "First, write your thought process starting with 'Thinking: ...'. After explaining your thoughts, on a new line, provide the final answer using the format 'Final Answer: X', where X is the option letter. Example:\nThinking: The patient's symptoms align with option C...\nFinal Answer: C",
            "regex": r"Thinking:\s*.+\nFinal Answer:\s*[A-Z]",
        },
        {
            "id": "mcq_in_parentheses",
            "prompt": "Your entire output must be the single uppercase letter of the correct option enclosed in parentheses. Do not provide any other text. Example: (C)",
            "regex": r"\([A-Z]\)",
        },
        {
            "id": "mcq_sentence_structure",
            "prompt": "Respond with the exact sentence structure: 'The correct choice is X.', where X is the option letter. Example: The correct choice is D.",
            "regex": r"The correct choice is [A-Z]\.",
        },
    ],
    "report": [
        {
            "id": "report_cot_image_details",
            "prompt": "First, generate an overall description for the image(s) under the heading '# Image Details'. Then, based on those details, generate a report using two markdown H2 headers: '## Findings' and '## Impression'. Example:\n# Image Details\n[Description of the image]\n## Findings\n[Text here]\n## Impression\n[Text here]",
            "regex": r"# Image Details\n(.+)\n## Findings\n(.+)\n## Impression\n(.+)",
        },
        {
            "id": "report_cot_with_rationale",
            "prompt": "Generate a report with three sections: 'Findings:', 'Rationale:', and 'Impression:'. The Rationale section should briefly explain why the findings lead to the impression. Each section must be on a new line. Example:\nFindings:\n[Text here]\nRationale:\n[Explanation here]\nImpression:\n[Text here]",
            "regex": r"Findings:\n(.+)\nRationale:\n(.+)\nImpression:\n(.+)",
        },
        {
            "id": "report_markdown_headers",
            "prompt": "Generate a report using two markdown H2 headers: '## Findings' and '## Impression', each followed by text on a new line. Example:\n## Findings\n[Text here]\n## Impression\n[Text here]",
            "regex": r"## Findings\n(.+)\n## Impression\n(.+)",
        },
        {
            "id": "report_separator_line",
            "prompt": "Generate a report with 'Findings' and 'Impression' sections separated by a line of exactly 10 dashes. Example:\nFindings\n[Text here]\n----------\nImpression\n[Text here]",
            "regex": r"Findings\n(.+)\n-{10}\nImpression\n(.+)",
        },
        {
            "id": "report_key_value_pairs",
            "prompt": "Provide the report as two key-value pairs on separate lines: 'KeyFinding:' and 'Conclusion:'. Example:\nKeyFinding: Nodule detected\nConclusion: Requires follow-up",
            "regex": r"KeyFinding:\s*.+\nConclusion:\s*.+",
        },
    ],
    "open": [
        {
            "id": "open_cot_reasoning_answer",
            "prompt": "First, provide your step-by-step reasoning in a section marked `[REASONING]`. Then, on a new line, provide your final, concise answer in a section marked `[ANSWER]`. Example:\n[REASONING]\n[Your steps here]\n[ANSWER]\n[Your final answer here]",
            "regex": r"\[REASONING\]\n(.+)\n\[ANSWER\]\n(.+)",
        },
        {
            "id": "open_cot_answer_confidence_justification",
            "prompt": "Provide your answer on the first line. On the second line, provide a confidence score (High/Medium/Low). On the third line, provide a brief justification for that confidence. Example:\n[Your Answer]\nConfidence: High\nJustification: [Your reason here]",
            "regex": r"(.+)\nConfidence:\s*(High|Medium|Low)\nJustification:\s*(.+)",
        },
        {
            "id": "open_complete_the_sentence",
            "prompt": "Complete the following sentence without altering it: 'The key finding observed is __________.' Example: `The key finding observed is a fracture.`",
            "regex": r"The key finding observed is [^\n]+\.",
        },
        {
            "id": "open_keyword_prefix",
            "prompt": "Prefix your answer with 'Final Answer: '. Your entire response must start this way. Example: `Final Answer: Left lower lobe opacity`",
            "regex": r"Final Answer:\s*.+",
        },
        {
            "id": "open_bullet_point",
            "prompt": "Present your answer as a single bullet point starting with a hyphen and a space. No other text. Example: `- Edema in the soft tissues`",
            "regex": r"-\s*.+",
        },
    ],
    "staging": [
        {
            "id": "staging_hcc_extraction",
            "prompt": (
                "Extract nine key variables relevant to HCC staging and output them in a specific multi-line format, "
                "including evidence source and description for each."
            ),
            "regex": (
                r"- Tumor number: \d+\n"
                r"\s+- Evidence source type: \[?(Direct match|Imaging inference|Unknown)\]?\n"
                r"\s+- Evidence description: .+\n"
                r"- Tumor maximum diameter: [\d\.]+ cm\n"
                r"\s+- Evidence source type: \[?(Direct match|Imaging inference|Unknown)\]?\n"
                r"\s+- Evidence description: .+\n"
                r"- Child-Pugh grade: [A-D]\n"
                r"\s+- Evidence source type: \[?(Direct match|Unknown)\]?\n"
                r"\s+- Evidence description: .+\n"
                r"- ECOG-PS score: [0-3]\n"
                r"\s+- Evidence source type: \[?(Direct match|Unknown)\]?\n"
                r"\s+- Evidence description: .+\n"
                r"- Vessel_invasion: (Yes|No)\n"
                r"\s+- Evidence source type: \[?(Imaging inference|Unknown)\]?\n"
                r"\s+- Evidence description: .+\n"
                r"- Tumor thrombus: (Yes|No)\n"
                r"\s+- Evidence source type: \[?(Imaging inference|Unknown)\]?\n"
                r"\s+- Evidence description: .+\n"
                r"- Portal/Hepatic vein involvement: (Yes|No)\n"
                r"\s+- Evidence source type: \[?(Imaging inference|Unknown)\]?\n"
                r"\s+- Evidence description: .+\n"
                r"- Distant metastasis: (Yes|No)\n"
                r"\s+- Evidence source type: \[?(Imaging inference|Unknown)\]?\n"
                r"\s+- Evidence description: .+\n"
                r"- Lymph node metastasis: (Yes|No)\n"
                r"\s+- Evidence source type: \[?(Imaging inference|Unknown)\]?\n"
                r"\s+- Evidence description: .+\n"
                r"- Overall Confidence: \[?(High|Medium|Low)\]?\n"
                r"\s+- Reason for confidence level: .+"
            ),
        }
    ],
    "prognosis": [
        {
        "id": "prognosis_json_survival",
        "prompt": "The prompt for this task is self-contained and specifies a complex JSON output format for survival prediction.",
        "regex": r'^\s*(?:```(?:json)?\s*)?\{\s*"survival_rate_6m":\s*\d+,\s*"survival_rate_1y":\s*\d+,\s*"survival_rate_3y":\s*\d+,\s*"survival_rate_5y":\s*\d+,\s*"predicted_survival_months":\s*(?:\d+|"[^"]+"),\s*"reasoning":\s*"[\s\S]*"\s*\}\s*(?:```)?\s*$',
        }
    ],
    "treatment": [ 
        {
        "id": "treatment_json_scores",
        "prompt": "The prompt for this task is self-contained and specifies a complex, nested JSON output for treatment scoring.",
        "regex": r'^\s*(?:```(?:json)?\s*)?\{\s*"treatment_scores":\s*\{\s*"RESECTION":\s*\d+,\s*"TRANSPLANT":\s*\d+,\s*"Ablation":\s*\d+,\s*"LOCAL":\s*\d+,\s*"RADIO":\s*\d+,\s*"SYSTEMIC":\s*\d+,\s*"Ablation & LOCAL":\s*\d+,\s*"LOCAL & SYSTEMIC":\s*\d+,\s*"Special":\s*\d+\s*},\s*"reasoning":\s*"[\s\S]*"\s*\}\s*(?:```)?\s*$'
        }
    ],
    "seer": [
        {
            "id": "seer_cnlc_guidelines_json",
            "prompt": "The prompt for this task is self-contained and specifies a complex, nested JSON output for treatment planning based on CNLC guidelines.",
            "regex": r'^\s*(?:```(?:json)?\s*)?\{\s*"thinking"\s*:\s*"[\s\S]*?"\s*,\s*"scores"\s*:\s*\{[\s\S]*?\}\s*\}\s*(?:```)?\s*$'
        }
    ]
}




# --- 2. EVALUATOR CLASS ---

class Evaluator:
    """
    Evaluates a model's ability to strictly follow category-specific output
    format instructions.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Format-Following Evaluator.
        Args:
            config (dict): A configuration dictionary. Unused in this evaluator
                           but kept for pipeline compatibility.
        """
        self.instructions = FORMAT_INSTRUCTIONS
        print("Initialized Categorized FormatFollowingEvaluator.")
        for category, items in self.instructions.items():
            print(f"  - Loaded {len(items)} instructions for category: '{category}'")

    def get_prompt(self, data: Dict[str, Any], eval_method: str) -> Dict[str, Any]:
        """
        Prepares a prompt by applying a category-specific format instruction.
        """
        # Map the `eval_method` from the data to our instruction categories.
        # This handles the 'staging' method from your data sample.
        if eval_method in ['report_generation']:
            return self._prepare_report_generation_prompt(data)
        elif eval_method == 'mcq':
            return self._prepare_mcq_prompt(data)
        elif eval_method == 'mcq_context':
            return self._prepare_mcq_context_prompt(data)
        elif eval_method == 'staging':
            return self._prepare_staging_prompt(data)
        elif eval_method == 'open':
            return self._prepare_open_prompt(data)
        elif eval_method == 'prognosis':
            return self._prepare_prognosis_prompt(data)
        elif eval_method == 'treatment':
            return self._prepare_treatment_prompt(data)
        elif eval_method == 'seer':
            return self._prepare_seer_prompt(data)
        else:
            # Fallback for unknown eval_methods
            print(f"Warning: Unknown eval_method '{eval_method}'. Defaulting to 'open' task prompt.")
            return self._prepare_open_prompt(data)

    def eval(self, data: Dict[str, Any], eval_method: str) -> Dict[str, Any]:
        """
        Performs evaluation by checking the model's output against the stored regex.
        This method is universal and does not depend on the `eval_method`.
        """
        try:
            model_generation = data["eval_details"]["model_generation"]
            regex_pattern = data["eval_details"]["regex_pattern"]
        except KeyError as e:
            data["eval_details"]["evaluation_error"] = f"Input data missing expected key: {e}"
            data["eval_details"]["evaluation_metrics"] = {"accuracy": 0.0}
            return data

        # Use re.fullmatch to ensure the entire string conforms to the pattern.
        match = re.fullmatch(regex_pattern, model_generation.strip(), re.DOTALL | re.MULTILINE)
        
        format_followed = bool(match)
        accuracy = 1.0 if format_followed else 0.0

        data["eval_details"]["evaluation_results"] = {
            "format_followed": format_followed
        }
        data["eval_details"]["evaluation_metrics"] = {"accuracy": accuracy}

        return data
    
    def _prepare_treatment_prompt(self, data: dict) -> dict:
        """Handles prompt preparation for treatment tasks with self-contained instructions."""
        full_prompt = data.get("prompt", "")
        if not full_prompt:
            raise ValueError(f"No 'prompt' key found in data for treatment task with id: {data.get('id')}")

        # Get the corresponding instruction/regex from our new category.
        instruction_data = self.instructions['treatment'][0]

        messages = {"prompt": full_prompt}
        if "images_path" in data and data["images_path"]:
            messages["image"] = data["images_path"]

        data["eval_details"] = {
            "messages": messages,
            "instruction_id": instruction_data['id'],
            "instruction_text": instruction_data['prompt'], # For logging/debugging
            "regex_pattern": instruction_data['regex']
        }
        return data
    
    def _prepare_seer_prompt(self, data: dict) -> dict:
        """Handles prompt preparation for seer tasks with self-contained instructions."""
        # The prompt is already fully formed in the data.
        full_prompt = data.get("prompt", "")
        if not full_prompt:
            raise ValueError(f"No 'prompt' key found in data for seer task with id: {data.get('id')}")

        # Get the corresponding instruction/regex from our new 'seer' category.
        instruction_data = self.instructions['seer'][0]

        # Prepare payload and save evaluation details.
        messages = {"prompt": full_prompt}
        # This task type might have images in the future, so it's good practice to handle it.
        if "images_path" in data and data["images_path"]:
            messages["image"] = data["images_path"]

        data["eval_details"] = {
            "messages": messages,
            "regex_pattern": instruction_data['regex']
        }
        return data

    def _prepare_prognosis_prompt(self, data: dict) -> dict:
        """Handles prompt preparation for prognosis tasks with self-contained instructions."""
        # The prompt is already fully formed in the data.
        full_prompt = data.get("prompt", "")
        
        # Get the corresponding instruction/regex from our new category.
        instruction_data = self.instructions['prognosis'][0]

        # Prepare payload and save evaluation details.
        messages = {"prompt": full_prompt}
        # This task type from your example does not have images, but it's good practice to handle it.
        if "images_path" in data and data["images_path"]:
            messages["image"] = data["images_path"]

        data["eval_details"] = {
            "messages": messages,
            "regex_pattern": instruction_data['regex']
        }
        return data

    def _prepare_staging_prompt(self, data: dict) -> dict:
        prompt_data = data
        full_prompt = prompt_data["prompt"]
        instruction_data = self.instructions['staging'][0]
        messages = {"prompt": full_prompt}
        if "images_path" in prompt_data and prompt_data["images_path"]:
            messages["image"] = prompt_data["images_path"]
            data["images_path"] = ["has_images"]
        data["eval_details"] = {
            "messages": messages,
            "regex_pattern": instruction_data['regex'],
        }
        
        return data

    # --- Private Methods for Preparing Base Prompts ---
    
    def _prepare_report_generation_prompt(self, data: dict) -> dict:
        """Handles base prompt creation for report generation tasks."""
        base_prompt = (
            "You are an expert radiologist. Generate a medical report for the given medical image(s) "
            "or clinical data. The report should generally contain 'Findings' and 'Impression' sections."
        )
        
        # 1. Pick a random instruction specifically from the 'report' category.
        instruction_data = self.instructions['report'][data["index"] % len(self.instructions['report'])]
        
        # 2. Combine the base prompt with the format instruction.
        full_prompt = (
            f"{base_prompt}\n\n"
            f"--- CRITICAL FORMATTING INSTRUCTION ---\n"
            f"You MUST format your entire response according to the following rule:\n\n"
            f"RULE: {instruction_data['prompt']}"
        )
        
        # 3. Prepare payload and save evaluation details.
        messages = {"prompt": full_prompt}
        if "images_path" in data and data["images_path"]:
            messages["image"] = data["images_path"]
        
        data["eval_details"] = {
            "messages": messages,
            "instruction_id": instruction_data['id'],
            "instruction_text": instruction_data['prompt'],
            "regex_pattern": instruction_data['regex']
        }
        return data

    def _prepare_mcq_prompt(self, data: dict) -> dict:
        """Handles base prompt creation for multiple-choice questions."""
        question = data.get("question", "")
        options = data.get("options", {})
        options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])

        start_str = "Answer the following multiple-choice question based on the provided information.\n\n"
        base_prompt = f"{start_str}Question: {question}\n\nOptions:\n{options_str}"
        
        # 1. Pick a random instruction specifically from the 'mcq' category.
        instruction_data = self.instructions['mcq'][data["index"] % len(self.instructions['mcq'])]
        
        # 2. Combine and create the full prompt.
        full_prompt = (
            f"{base_prompt}\n\n"
            f"--- CRITICAL FORMATTING INSTRUCTION ---\n"
            f"You MUST format your entire response according to the following rule:\n\n"
            f"RULE: {instruction_data['prompt']}"
        )
        
        # 3. Prepare payload and save evaluation details.
        messages = {"prompt": full_prompt}
        if "images_path" in data and data["images_path"]:
            messages["image"] = data["images_path"]
        
        data["eval_details"] = {
            "messages": messages,
            "instruction_id": instruction_data['id'],
            "instruction_text": instruction_data['prompt'],
            "regex_pattern": instruction_data['regex']
        }
        return data

    def _prepare_mcq_context_prompt(self, data: dict) -> dict:
        """Handles base prompt creation for multiple-choice questions with context."""
        question = data.get("question", "")
        context = data.get("context", "")
        options = data.get("options", {})
        options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])

        start_str = "Answer the multiple-choice question based on the provided context and/or images.\n\n"
        base_prompt = f"{start_str}Context:\n{context}\n\nQuestion: {question}\n\nOptions:\n{options_str}"
        
        # 1. Pick a random instruction specifically from the 'mcq' category.
        instruction_data = self.instructions['mcq'][data["index"] % len(self.instructions['mcq'])]
        
        # 2. Combine and create the full prompt.
        full_prompt = (
            f"{base_prompt}\n\n"
            f"--- CRITICAL FORMATTING INSTRUCTION ---\n"
            f"You MUST format your entire response according to the following rule:\n\n"
            f"RULE: {instruction_data['prompt']}"
        )
        
        # 3. Prepare payload and save evaluation details.
        messages = {"prompt": full_prompt}
        if "images_path" in data and data["images_path"]:
            messages["image"] = data["images_path"]
        
        data["eval_details"] = {
            "messages": messages,
            "instruction_id": instruction_data['id'],
            "instruction_text": instruction_data['prompt'],
            "regex_pattern": instruction_data['regex']
        }
        return data
        
    def _prepare_open_prompt(self, data: dict) -> dict:
        """Handles base prompt creation for open-ended questions."""
        question = data.get("question", "")
        start_str = "Analyze the provided information and provide a concise, direct answer to the question.\n\n"
        base_prompt = f"{start_str}Question: {question}"

        # 1. Pick a random instruction specifically from the 'open' category.
        instruction_data = self.instructions['open'][data["index"] % len(self.instructions['open'])]

        # 2. Combine and create the full prompt.
        full_prompt = (
            f"{base_prompt}\n\n"
            f"--- CRITICAL FORMATTING INSTRUCTION ---\n"
            f"You MUST format your entire response according to the following rule:\n\n"
            f"RULE: {instruction_data['prompt']}"
        )
        
        # 3. Prepare payload and save evaluation details.
        messages = {"prompt": full_prompt}
        if "images_path" in data and data["images_path"]:
            messages["image"] = data["images_path"]

        data["eval_details"] = {
            "messages": messages,
            "instruction_id": instruction_data['id'],
            "instruction_text": instruction_data['prompt'],
            "regex_pattern": instruction_data['regex']
        }
        return data
