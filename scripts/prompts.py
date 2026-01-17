import sys
import os
# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define prompt templates for ground-truth generation
qa_prompt_template = (
    "Generate a question and a comprehensive answer in the following format:\n"
    "Question: generated_question Answer: generated_comprehensive_answer\n"
    "based on the following text:\n\n"
    "{text}\n\n"
    "Question:"
)

structured_prompt_template = (
    "Extract all relations between characters in JSON format:\n"
    "{characters: [{name, relations: [{target, type}]}]}"
)

entity_extraction_prompt_template = (
    "Extract all named entities (people, places, organizations, dates) from the following text "
    "and format them as JSON:\n\n"
    "{text}\n\n"
    "Format the output as:\n"
    "{{\"entities\": [{{\"type\": \"entity_type\", \"name\": \"entity_name\"}}]}}"
)

summary_prompt_template = (
    "Provide a concise summary of the following text:\n\n"
    "{text}\n\n"
    "Summary:"
)
