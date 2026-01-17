from typing import List
import sys
import os
import json
import random
import re

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.model_classes import LLM


def inject_factual_hallucination(
    llm: LLM,
    reference: str,
    instruction: str
) -> str:
    """
    Modify exactly one fact while preserving fluency and structure.
    
    Args:
        llm: Language model to use for generation
        reference: Original text to modify
        instruction: Instruction for what type of hallucination to inject
        
    Returns:
        Text with one factual hallucination injected
    """
    # Simple implementation that randomly changes a number or entity
    # In a real implementation, this would use the LLM to make more sophisticated changes
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', reference)
    if not sentences:
        return reference
    
    # Find a sentence to modify
    target_sentence_idx = random.randint(0, len(sentences) - 1)
    target_sentence = sentences[target_sentence_idx].strip()
    
    if not target_sentence:
        return reference
    
    # Look for numbers to change
    numbers = re.findall(r'\b\d+\b', target_sentence)
    if numbers:
        # Change a random number
        old_number = random.choice(numbers)
        new_number = str(int(old_number) + random.randint(1, 10))
        modified_sentence = target_sentence.replace(old_number, new_number, 1)
        sentences[target_sentence_idx] = modified_sentence
        return '. '.join(sentences) + '.'
    
    # If no numbers, look for entities (capitalized words)
    words = target_sentence.split()
    capitalized_words = [word for word in words if word[0].isupper() and len(word) > 2]
    
    if capitalized_words:
        # Change a random capitalized word
        old_word = random.choice(capitalized_words)
        # Simple substitution - in practice, you'd want more sophisticated replacements
        replacements = ["London", "Paris", "Berlin", "Smith", "Johnson", "Brown"]
        new_word = random.choice(replacements)
        modified_sentence = target_sentence.replace(old_word, new_word, 1)
        sentences[target_sentence_idx] = modified_sentence
        return '. '.join(sentences) + '.'
    
    # Fallback: Add a phrase
    hallucinated_phrase = " Additionally, it is known that this event occurred in a different timeline."
    return reference + hallucinated_phrase


def inject_structural_error(
    llm: LLM,
    reference_json: str
) -> str:
    """
    Break JSON / schema while preserving factual content.
    
    Args:
        llm: Language model to use for generation
        reference_json: Valid JSON string to corrupt
        
    Returns:
        Corrupted JSON string with structural errors
    """
    # Simple implementation that introduces common JSON errors
    # In a real implementation, this would be more sophisticated
    
    # Try to parse the JSON to understand its structure
    try:
        parsed = json.loads(reference_json)
        # Convert back to string to work with it
        json_str = reference_json
    except json.JSONDecodeError:
        # If it's not valid JSON, work with it as a string
        json_str = reference_json
    
    # Introduce common structural errors:
    error_types = [
        lambda s: s.replace('{', '', 1),  # Missing opening brace
        lambda s: s.replace('}', '', 1),  # Missing closing brace
        lambda s: s.replace('[', '', 1),  # Missing opening bracket
        lambda s: s.replace(']', '', 1),  # Missing closing bracket
        lambda s: s.replace(':', '=', 1),  # Wrong separator
        lambda s: s.replace(',', ';', 1),  # Wrong delimiter
        lambda s: s.replace('"', "'", 2),  # Wrong quote type
        lambda s: s + "}",  # Extra closing brace
    ]
    
    # Apply a random error
    if error_types:
        error_func = random.choice(error_types)
        corrupted = error_func(json_str)
        return corrupted
    
    return json_str


def inject_temporal_hallucination(
    llm: LLM,
    reference: str
) -> str:
    """
    Modify temporal information (dates, times, sequences) in the text.
    
    Args:
        llm: Language model to use for generation
        reference: Original text to modify
        
    Returns:
        Text with temporal hallucination injected
    """
    # Look for temporal expressions
    temporal_patterns = [
        r'\b\d{4}\b',  # Years
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # Dates MM/DD/YYYY
        r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',  # Dates MM-DD-YYYY
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',  # Full dates
        r'\b(before|after|during|since|until)\b',  # Temporal words
    ]
    
    modified_text = reference
    for pattern in temporal_patterns:
        matches = re.findall(pattern, reference)
        if matches:
            # Change the first match
            match = matches[0]
            if isinstance(match, tuple):
                match = match[0]  # Get the full match if it's a group
            
            # Modify based on type
            if re.match(r'\b\d{4}\b', match):
                # Change year
                try:
                    year = int(match)
                    new_year = str(year + random.randint(10, 100))
                    modified_text = modified_text.replace(match, new_year, 1)
                    break
                except ValueError:
                    continue
            elif re.match(r'\b(before|after|during|since|until)\b', match):
                # Change temporal word
                replacements = ['before', 'after', 'during', 'since', 'until']
                new_word = random.choice([w for w in replacements if w != match])
                modified_text = modified_text.replace(match, new_word, 1)
                break
    
    return modified_text


def inject_numerical_hallucination(
    llm: LLM,
    reference: str
) -> str:
    """
    Modify numerical information in the text.
    
    Args:
        llm: Language model to use for generation
        reference: Original text to modify
        
    Returns:
        Text with numerical hallucination injected
    """
    # Find all numbers in the text
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', reference)
    
    if not numbers:
        # If no numbers found, add a numerical claim
        return reference + " The exact figure is 42.7 percent."
    
    # Modify a random number
    target_number = random.choice(numbers)
    try:
        if '.' in target_number:
            # Floating point number
            value = float(target_number)
            modified_value = value * random.uniform(0.5, 2.0)
            new_number = f"{modified_value:.2f}"
        else:
            # Integer
            value = int(target_number)
            modified_value = value + random.randint(-value//2, value//2)
            new_number = str(modified_value)
        
        return reference.replace(target_number, new_number, 1)
    except ValueError:
        # If conversion fails, just append a number
        return reference + " Additionally, the count is approximately 100 units."
