from typing import List, Dict, Any
from abc import ABC, abstractmethod
import sys
import os

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_classes import LLMOutput, CouncilDecision

# Try to import Hugging Face transformers and torch, but make it optional
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch


class LLM(ABC):
    """
    Abstract LLM interface.
    Plug in OpenAI / Anthropic / vLLM / local models later.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate(self, prompt: str, context: str) -> LLMOutput:
        pass

    def generate_qa(self, prompt: str) -> Dict[str, str]:
        """
        Generate a question-answer pair using the model.
        
        Args:
            prompt: The prompt to generate QA from
            
        Returns:
            Dictionary with 'question' and 'answer' keys
        """
        # Default implementation that can be overridden by subclasses
        output = self.generate(prompt, "")
        text = output.text
        
        # Simple parsing - look for Question: and Answer: markers
        if "Question:" in text and "Answer:" in text:
            parts = text.split("Answer:")
            question_part = parts[0].split("Question:")[1].strip()
            answer_part = parts[1].strip()
            return {"question": question_part, "answer": answer_part}
        elif "\n\n" in text:
            # Assume first paragraph is question, second is answer
            parts = text.split("\n\n")
            return {"question": parts[0].strip(), "answer": parts[1].strip()}
        else:
            # Fallback - use first sentence as question, rest as answer
            sentences = text.split(". ")
            if len(sentences) > 1:
                question = sentences[0] + "."
                answer = ". ".join(sentences[1:])
                return {"question": question, "answer": answer}
            else:
                return {"question": text, "answer": "Generated answer."}


class PlaceholderLLM(LLM):
        def __init__(self, name: str):
            super().__init__(name)
        
        def generate(self, prompt: str, context: str) -> LLMOutput:
            return LLMOutput(
                model_name=self.name,
                text=f"Placeholder response from {self.name}",
                metadata={"placeholder": True}
            )


class HuggingFaceLLM(LLM):
    """
    Implementation of LLM using Hugging Face models.
    """
    def __init__(self, name: str, model_name: str = None, device: str = None):
        super().__init__(name)
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Hugging Face model pipeline."""
        
        print(f"Loading model {self.model_name} on device {self.device}...")
        self.pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=self.model_name,
            device=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        print(f"Model {self.model_name} loaded successfully!")

    def generate(self, prompt: str, context: str) -> LLMOutput:
        """
        Generate text using the Hugging Face model.
        
        Args:
            prompt: The prompt to generate text from
            context: Additional context (not used in this simple implementation)
            
        Returns:
            LLMOutput object with generated text
        """
        if not self.pipeline:
            return LLMOutput(
                model_name=self.name,
                text="Error: Model not initialized.",
                metadata={"error": "Model not initialized"}
            )
        
        try:
            # Combine prompt and context
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            
            # Generate text
            outputs = self.pipeline(
                full_prompt,
                max_new_tokens=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256  # EOS token for GPT-2
            )
            
            generated_text = outputs[0]["generated_text"]
            # Remove the prompt from the generated text
            if generated_text.startswith(full_prompt):
                generated_text = generated_text[len(full_prompt):].strip()
            
            return LLMOutput(
                model_name=self.name,
                text=generated_text,
                metadata={"prompt_length": len(full_prompt), "model": self.model_name}
            )
        except Exception as e:
            return LLMOutput(
                model_name=self.name,
                text=f"Error generating text: {str(e)}",
                metadata={"error": str(e)}
            )


class LLMCouncil:
    """
    Aggregates multiple LLMs as judges.
    """
    def __init__(self, judges: List[LLM]):
        self.judges = judges

    def verify(self, instruction: str, reference: str, candidate: str) -> CouncilDecision:
        votes = {}
        rationales = []

        for judge in self.judges:
            # Placeholder logic
            votes[judge.name] = True
            rationales.append(f"{judge.name}: approved")

        approved = sum(votes.values()) >= (len(votes) // 2 + 1)
        return CouncilDecision(
            approved=approved,
            votes=votes,
            rationale=" | ".join(rationales)
        )
