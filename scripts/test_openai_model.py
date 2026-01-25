import sys
import os

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.models.model_classes import OpenAILLM, LLMCouncil
from scripts.models.data_classes import LLMOutput, CouncilDecision

def test_openai_model():
    """Test the OpenAI model implementation."""
    try:
        # Initialize the OpenAI model
        openai_model = OpenAILLM(name="test-gpt-3.5", model_name="gpt-3.5-turbo")
        print("OpenAI model initialized successfully!")
        
        # Test generation
        prompt = "What is the capital of France?"
        context = "Geography question"
        
        output = openai_model.generate(prompt, context)
        print(f"Generated output: {output.text}")
        print(f"Metadata: {output.metadata}")
        
        return True
    except Exception as e:
        print(f"Error testing OpenAI model: {e}")
        return False

def test_council_with_openai():
    """Test the LLMCouncil with OpenAI model as judge."""
    try:
        # Initialize models
        openai_judge = OpenAILLM(name="gpt-3.5-turbo", model_name="gpt-3.5-turbo")
        council = LLMCouncil(judges=[openai_judge])
        
        # Test council verification
        instruction = "Name the capital of France."
        reference = "The capital of France is Paris."
        candidate = "Paris is the capital city of France."
        
        decision = council.verify(instruction, reference, candidate)
        print(f"Council decision: Approved = {decision.approved}")
        print(f"Votes: {decision.votes}")
        print(f"Rationale: {decision.rationale}")
        
        return True
    except Exception as e:
        print(f"Error testing LLMCouncil with OpenAI: {e}")
        return False

if __name__ == "__main__":
    print("Testing OpenAI model implementation...")
    
    # Test individual model
    model_success = test_openai_model()
    
    # Test council with OpenAI judge
    council_success = test_council_with_openai()
    
    if model_success and council_success:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed.")
