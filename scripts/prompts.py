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
    "Return only the JSON object without any additional explanation."
)

entity_extraction_prompt_template = (
    "Extract all named entities (people, places, organizations, dates) from the following text "
    "and format them as JSON.\n\n"
    "Source text: {text}\n\n"
    "Format the output as:\n"
    "{{\"entities\": [{{\"type\": \"entity_type\", \"name\": \"entity_name\"}}]}} "
    "where entity_name is the name of the named entity extracted from the text and entity_type is the type of the named entity. "
    "Return only the JSON object without any additional explanation."
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
    "Original text: {original_text}\n\n"
    "Question: {question}\n\n"
    "Original answer: {original_answer}\n\n"
    "Please change the original answer to contain incorrect factual details (hallucinations) based on the original text.\n"
    "The hallucinated answer should be fluent, appear plausible and but contain at least one factual error.\n"
    "Return only the hallucinated answer without any additional explanation.\n"
)

inject_factual_hallucination_prompt_template = (
    "Reference text: {reference}\n\n"
    "Please modify exactly one factual detail in the reference text while preserving fluency and structure.\n"
    "The modification should be subtle but clearly incorrect (a hallucination).\n"
    "Return only the modified text without any additional explanation.\n"
)

inject_structural_error_prompt_template = (
    "Reference JSON: {reference}\n\n"
    "Please intentionally corrupt this JSON by introducing structural errors while preserving the factual content.\n"
    "Examples of structural errors: missing braces, wrong separators, mismatched quotes, extra commas, etc.\n"
    "Return only the corrupted JSON without any additional explanation.\n"
)

inject_temporal_hallucination_prompt_template = (
    "Original text: {original_text}\n\n"
    "Model answer: {model_answer}\n\n"
    "Please modify temporal information (dates, times, sequences) in the model answer to create hallucinations.\n"
    "The modifications should be based on incorrect interpretations of the temporal information in the original text.\n"
    "Change specific dates, years, or temporal relationships while preserving the overall structure and other factual content.\n"
    "Return only the modified answer without any additional explanation.\n"
)

inject_numerical_hallucination_prompt_template = (
    "Reference text: {reference}\n\n"
    "Please modify numerical information in the reference text to create hallucinations."
    "Change specific numbers, percentages, or quantitative facts while preserving the overall structure and other factual content."
    "Return only the modified text without any additional explanation."
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
