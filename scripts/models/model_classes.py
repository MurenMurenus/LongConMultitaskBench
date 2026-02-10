from typing import List, Dict
from abc import ABC, abstractmethod
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.models.data_classes import LLMOutput, CouncilDecision

from transformers import pipeline
import torch
import os


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
            dtype=torch.float16 if self.device == "cuda" else torch.float32
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
                # temperature=0.7,
                do_sample=False,
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


class OpenAILLM(LLM):
    """
    Implementation of LLM using OpenAI API.
    """
    def __init__(self, name: str, model_name: str = "gpt-3.5-turbo", api_key: str = None):
        super().__init__(name)
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either as api_key parameter or OPENAI_API_KEY environment variable")

        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, context: str) -> LLMOutput:
        """
        Generate text using the OpenAI API.
        
        Args:
            prompt: The prompt to generate text from
            context: Additional context to include with the prompt
            
        Returns:
            LLMOutput object with generated text
        """
        try:
            # Combine prompt and context
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            
            # Create messages for chat completion
            messages = [
                {"role": "user", "content": full_prompt}
            ]
            
            # Generate text using OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            return LLMOutput(
                model_name=self.name,
                text=generated_text,
                metadata={
                    "model": self.model_name,
                    "prompt_length": len(full_prompt),
                    "completion_tokens": response.usage.completion_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens
                }
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
        rationales = {}

        for judge in self.judges:
            try:
                # Create evaluation prompt
                eval_prompt = f"""
                Instruction: {instruction}
                
                Original text: {reference}
                
                Candidate Answer: {candidate}
                
                Based on the instruction and original text, evaluate whether the candidate answer is correct and relevant.
                
                Provide your evaluation in the following format:
                Vote: [Yes/No] - Whether the candidate answer is acceptable.
                Rationale: [Your rationale] - Your explanation for the vote.
                
                """
                evaluation = judge.generate(eval_prompt, "")
                eval_text = evaluation.text
                
                vote = False
                rationale = "No evaluation response"
                
                if "Vote:" in eval_text:
                    # Extract vote and rationale
                    lines = eval_text.split('\n')
                    vote_line = next((line for line in lines if line.strip().startswith("Vote:")), "")
                    rationale_line = next((line for line in lines if line.strip().startswith("Rationale:")), "")
                    
                    vote = "yes" in vote_line.lower() if vote_line else False
                    rationale = rationale_line.replace("Rationale:", "").strip() if rationale_line else eval_text
                else:
                    # Fallback - try to determine from general text
                    vote = "yes" in eval_text.lower() or "correct" in eval_text.lower() or "acceptable" in eval_text.lower()
                    rationale = eval_text.strip()
                
                votes[judge.name] = vote
                rationales[judge.name] = rationale
            except Exception as e:
                votes[judge.name] = False
                rationales[judge.name] = ("Error during evaluation - {str(e)}")

        approved = sum(votes.values()) >= (len(votes) // 2 + 1)
        return CouncilDecision(
            approved=approved,
            votes=votes,
            rationales=rationales
        )
