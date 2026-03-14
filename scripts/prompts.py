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
    "{text}\n"
    "Don't explain the question or the answer. Return only the question and the answer without any additional explanation.\n\n"
    "Question:"
)

structured_prompt_template = (
    "Extract all relations between characters in JSON format:\n"
    "{characters: [{name, relations: [{target, type}]}]}\n"
    "Return only the JSON object without any additional explanation.\n"
    "Extracted JSON:"
)

entity_extraction_prompt_template = (
    "Extract all named entities (people, places, dates and other types of entities you can extract) from the following text "
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

inject_entity_hallucination_prompt_template = (
    "Entities: {entities}\n\n"
    "Please modify entities to create hallucinations."
    "Change types of entities, names of entities, or relationships between entities while preserving the overall structure and other factual content."
    "Return only the modified JSON without any additional explanation."
    "Modified JSON:"
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
    "Ensure that all relations between characters are correctly identified and formatted."
    "Don't check for completeness, because the text is large and contains of many characters and relations."
)

validate_entity_extractions_council_prompt = (
    "Validate if the extracted entities are correctly identified from the original text."
    "Check for accuracy of the extracted entities."
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


# Define task-specific hallucination detection prompts
qa_hallucination_detection_prompt = """
You are tasked with detecting hallucinations in AI-generated question-answering responses. A hallucination is when the answer contains factual errors or information that contradicts the provided context.

Context: {context}

Question: {prompt}

Generated Answer: {output}

Based on the context and the question, determine if the generated answer contains any factual errors or hallucinations.

Instructions:
1. Compare the generated answer with the context carefully
2. Check if the answer correctly addresses the question based on the context
3. Look for any factual discrepancies, incorrect information, or statements that cannot be verified from the context
4. Pay special attention to details like names, dates, events, and relationships mentioned in the answer
5. Respond with only "HALLUCINATED" if the answer contains hallucinations, or "CORRECT" if it does not

Your response (HALLUCINATED or CORRECT):
"""

structured_hallucination_detection_prompt = """
You are tasked with detecting hallucinations in AI-generated structured outputs (JSON format). A hallucination is when the structured output contains entities, relationships, or information that contradicts the provided context.

Context: {context}

Prompt: {prompt}

Generated Structured Output: {output}

Based on the context and the prompt, determine if the generated structured output contains any factual errors or hallucinations.

Instructions:
1. Compare the generated structured output with the context carefully
2. Check if all entities, relationships, and information in the output can be verified from the context
3. Look for any fictional entities, incorrect relationships, or fabricated information
4. Validate the structure and format of the JSON output
5. Respond with only "HALLUCINATED" if the output contains hallucinations, or "CORRECT" if it does not

Your response (HALLUCINATED or CORRECT):
"""

entity_hallucination_detection_prompt = """
You are tasked with detecting hallucinations in AI-generated entity extraction outputs. A hallucination is when the extracted entities contain names, types, or information that contradicts the provided context.

Context: {context}

Prompt: {prompt}

Generated Entity Extraction: {output}

Based on the context and the prompt, determine if the generated entity extraction contains any factual errors or hallucinations.

Instructions:
1. Compare the extracted entities with the context carefully
2. Check if all extracted entities actually exist in the context
3. Verify that entity types are correctly identified
4. Look for any fictional entities, incorrect entity types, or fabricated information
5. Validate the structure and format of the JSON output
6. Respond with only "HALLUCINATED" if the extraction contains hallucinations, or "CORRECT" if it does not

Your response (HALLUCINATED or CORRECT):
"""

summary_hallucination_detection_prompt = """
You are tasked with detecting hallucinations in AI-generated summaries. A hallucination is when the summary contains factual errors or information that contradicts the provided context.

Context: {context}

Generated Summary: {output}

Based on the context, determine if the generated summary contains any factual errors or hallucinations.

Instructions:
1. Compare the generated summary with the context carefully
2. Check if all key points and facts in the summary are supported by the context
3. Look for any factual discrepancies, incorrect information, or statements that cannot be verified from the context
4. Pay special attention to details like names, dates, events, and relationships mentioned in the summary
5. Ensure the summary accurately represents the main points of the original text
6. Respond with only "HALLUCINATED" if the summary contains hallucinations, or "CORRECT" if it does not

Your response (HALLUCINATED or CORRECT):
"""
