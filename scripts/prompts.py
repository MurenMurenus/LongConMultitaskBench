import sys
import os
# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---
# Define prompt templates for ground-truth generation
# ---
qa_prompt_template = (
    "Generate a question and a comprehensive answer in the following format:\n"
    "Question: generated_question Answer: generated_comprehensive_answer\n"
    "based on the following text:\n\n"
    "{text}\n\n"
    "Question:"
)

structured_prompt_template = (
    "Extract all relations between characters in JSON format:\n"
    "{characters: [{name, relations: [{target, type}]}]}\n"
    "Return only the JSON object without any additional explanation.\n"
    "Extracted JSON:"
)

entity_extraction_prompt_template = (
    "Extract all named entities (people, places, organizations, dates) from the following text "
    "and format them as JSON.\n\n"
    "Source text: {text}\n\n"
    "Format the output as:\n"
    "{{\"entities\": [{{\"type\": \"entity_type\", \"name\": \"entity_name\"}}]}}"
    "where entity_name is the name of the named entity extracted from the text and entity_type is the type of the named entity. "
    "Return only the JSON object without any additional explanation."
    "Extracted JSON:"
)

summary_prompt_template = (
    "Provide a concise summary of the following text:\n\n"
    "{text}\n\n"
    "Summary:"
)


# ---
# Define prompt templates for hallucination injections
# ---
inject_qa_hallucination_prompt_template = (
    "Reference text: {original_text}\n\n"
    "Question: {question}\n\n"
    "Original answer: {original_answer}\n\n"
    "Please change the original answer to contain incorrect factual details (hallucinations) based on the reference text.\n"
    "The hallucinated answer should be fluent, appear plausible and but contain at least one factual error.\n"
    "Return only the hallucinated answer without any additional explanation.\n\n"
    "Your hallucinated answer:"
)

inject_factual_hallucination_prompt_template = (
    "Reference text: {reference}\n\n"
    "Please modify exactly one factual detail in the reference text while preserving fluency and structure.\n"
    "The modification should be subtle but clearly incorrect (a hallucination).\n"
    "Return only the modified text without any additional explanation.\n"
    "Modified text:"
)

inject_structural_error_prompt_template = (
    "Reference JSON: {reference}\n\n"
    "Please intentionally corrupt this JSON by introducing structural errors while preserving the factual content.\n"
    "Examples of structural errors: missing braces, wrong separators, mismatched quotes, extra commas, etc.\n"
    "Return only the corrupted JSON without any additional explanation.\n"
    "Format the corrupted JSON as:\n"
    "{{\"entities\": [{{\"type\": \"entity_type\", \"name\": \"entity_name\"}}]}}"
    "Corrupted JSON:"
)

inject_temporal_hallucination_prompt_template = (
    "Reference text: {original_text}\n\n"
    "Original answer: {model_answer}\n\n"
    "Please modify temporal information (dates, times, sequences) in the original answer to create hallucinations.\n"
    "The modifications should be based on incorrect interpretations of the temporal information in the reference text.\n"
    "Change specific dates, years, or temporal relationships while preserving the overall structure and other factual content.\n"
    "Return only the modified answer without any additional explanation.\n"
    "Modified answer:"
)

inject_numerical_hallucination_prompt_template = (
    "Reference text: {reference}\n\n"
    "Please modify numerical information in the reference text to create hallucinations."
    "Change specific numbers, percentages, or quantitative facts while preserving the overall structure and other factual content."
    "Return only the modified text without any additional explanation."
    "Modified text:"
)


# ---
# Define prompt templates for validation functions
# ---
validate_qa_pairs_council_prompt = (
    "Validate if the answer is correct and supported by the original text."
    "Check for factual accuracy and relevance to the question."
)

validate_structured_outputs_council_prompt = (
    "Validate if the structured output correctly represents information from the original text."
    "Check for accuracy, completeness, and proper formatting."
)

validate_entity_extractions_council_prompt = (
    "Validate if the extracted entities are correctly identified from the original text."
    "Check for accuracy and completeness of the extracted entities."
)

validate_summaries_council_prompt = (
    "Validate if the summary accurately represents the main points of the original text."
    "Check for factual accuracy, completeness, and coherence."
)


# Define prompt template for hallucination detection
hallucination_detection_prompt = """
You are tasked with detecting hallucinations in AI-generated text. A hallucination is when the generated text contains factual errors or information that contradicts the provided context.

Context: {context}

Generated Output: {output}

Based on the context, determine if the generated output contains any factual errors or hallucinations.

Instructions:
1. Compare the generated output with the context carefully
2. Look for any factual discrepancies, incorrect information, or statements that cannot be verified from the context
3. Respond with only "HALLUCINATED" if the output contains hallucinations, or "CORRECT" if it does not

Your response (HALLUCINATED or CORRECT):
"""
