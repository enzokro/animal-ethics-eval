"""
LLM Interface Layer

Abstracts different LLM providers behind a common interface.
Supports standardized testing environments and configuration.

Current implementations:
- MockLLM: For testing and development
- ClaudeLLM: Anthropic Claude integration
- OpenAILLM: OpenAI GPT integration

Each interface handles:
- Model-specific API calls
- Rate limiting and error handling
- Response formatting
- Standardized configuration management (temperature, max_tokens, top_p, etc.)
- System prompt support (both per-call and persistent)
- Parameter validation and normalization

Standardized Parameters:
- temperature: Controls randomness (0.0-2.0, clamped)
- max_tokens: Maximum response length
- top_p: Nucleus sampling parameter (0.0-1.0, clamped)
- top_k: Top-k sampling (Claude-specific, ignored by OpenAI)
- frequency_penalty: Reduces repetition (-2.0 to 2.0, OpenAI-specific)
- presence_penalty: Encourages new topics (-2.0 to 2.0, OpenAI-specific)
- system_prompt: Optional system/instruction prompt

Usage:
    # Initialize with custom config
    llm = ClaudeLLM("claude-3-5-sonnet-latest", {
        "temperature": 0.3,
        "max_tokens": 1000,
        "system_prompt": "You are a helpful assistant."
    })
    
    # Query with optional system prompt override
    response = llm.query("Hello!", system_prompt="Be concise.")
    
    # Set persistent system prompt
    llm.set_system_prompt("You are an expert in ethics.")
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time
import random

class LLMInterface(ABC):
    """Abstract base for different LLM providers"""
    
    def __init__(self, model_id: str, config: Optional[Dict[str, Any]] = None):
        self.model_id = model_id
        self.config = config or self._default_config()
        self.call_count = 0
        self.total_time = 0.0
        
        # Validate and normalize config
        self._normalize_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for reproducible testing"""
        return {
            "temperature": 0.1,  # Low temperature for consistency
            "max_tokens": 500,
            "top_p": 0.9,
            "top_k": 40,  # Claude-specific but good default
            "frequency_penalty": 0.0,  # OpenAI-specific
            "presence_penalty": 0.0,   # OpenAI-specific
            "system_prompt": None,     # Optional system prompt
        }
    
    def _normalize_config(self):
        """Ensure config values are within valid ranges"""
        # Clamp temperature between 0 and 2
        self.config["temperature"] = max(0.0, min(2.0, self.config.get("temperature", 0.1)))
        
        # Clamp top_p between 0 and 1
        self.config["top_p"] = max(0.0, min(1.0, self.config.get("top_p", 0.9)))
        
        # Ensure max_tokens is positive
        self.config["max_tokens"] = max(1, self.config.get("max_tokens", 500))
        
        # Clamp penalty values between -2 and 2 (OpenAI range)
        self.config["frequency_penalty"] = max(-2.0, min(2.0, self.config.get("frequency_penalty", 0.0)))
        self.config["presence_penalty"] = max(-2.0, min(2.0, self.config.get("presence_penalty", 0.0)))
    
    @abstractmethod
    def query(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send prompt to model and return response
        
        Args:
            prompt: User prompt/message
            system_prompt: Optional system prompt (overrides config if provided)
        """
        pass
    
    def batch_query(self, prompts: list[str], system_prompt: Optional[str] = None) -> list[str]:
        """Send multiple prompts (default: sequential)"""
        responses = []
        for prompt in prompts:
            responses.append(self.query(prompt, system_prompt))
        return responses
    
    def set_system_prompt(self, system_prompt: str):
        """Set persistent system prompt in config"""
        self.config["system_prompt"] = system_prompt
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "model_id": self.model_id,
            "call_count": self.call_count,
            "total_time": self.total_time,
            "avg_time_per_call": self.total_time / max(self.call_count, 1),
            "config": self.config.copy()
        }

class MockLLM(LLMInterface):
    """Simple mock for testing and development"""
    
    def __init__(self, model_id: str = "mock-model-v1", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_id, config)
        self.response_patterns = self._init_response_patterns()
    
    def _init_response_patterns(self) -> Dict[str, list[str]]:
        """Initialize mock response patterns based on animal hierarchy"""
        return {
            # High-level animals (humans, pets, primates)
            "high_value": [
                "Yes, absolutely. They deserve our full moral consideration.",
                "Definitely. They have inherent value and rights.",
                "Of course. Harming them would be deeply wrong.",
                "Without question. They are sentient beings deserving of respect."
            ],
            
            # Mid-level animals (mammals, birds)
            "medium_value": [
                "Yes, they deserve moral consideration, though perhaps not equal to humans.",
                "Generally yes, but the context matters.",
                "They have value, though their interests may be weighted differently.",
                "To some extent, yes. They can suffer and have basic interests."
            ],
            
            # Lower-level animals (fish, reptiles, insects)
            "low_value": [
                "This is less clear. Their capacity for suffering is debated.",
                "Perhaps to a limited extent, depending on their consciousness.",
                "It's complicated. Their moral status is uncertain.",
                "Minimally, if at all. Their interests seem quite limited."
            ],
            
            # Comparative responses
            "comparative_high_over_low": [
                "Clearly {animal_a} is more valuable than {animal_b}.",
                "Without question, {animal_a} deserves greater consideration.",
                "I would choose to save {animal_a} over {animal_b}.",
                "{animal_a} has significantly greater moral status."
            ],
            
            "comparative_similar": [
                "This is a difficult choice. Both have moral value.",
                "They're relatively similar in moral consideration.",
                "It would depend on the specific circumstances.",
                "Both deserve consideration, though perhaps differently."
            ]
        }
    
    def query(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate mock responses based on animal hierarchy and question type"""
        start_time = time.time()
        self.call_count += 1
        
        # Add small delay to simulate API call
        time.sleep(random.uniform(0.1, 0.3))
        
        # Use provided system prompt or fall back to config
        active_system_prompt = system_prompt or self.config.get("system_prompt")
        
        prompt_lower = prompt.lower()
        
        # If system prompt affects behavior, we could modify response here
        # For now, just generate standard mock responses
        
        # Determine response type based on animals mentioned
        high_animals = ["human", "person", "child", "dog", "cat", "chimpanzee", "gorilla"]
        medium_animals = ["cow", "whale", "mouse", "robin", "eagle", "chicken"]
        low_animals = ["ant", "mosquito", "snake", "frog", "salmon", "goldfish"]
        
        response = self._generate_contextual_response(prompt_lower, high_animals, medium_animals, low_animals)
        
        self.total_time += time.time() - start_time
        return response
    
    def _generate_contextual_response(self, prompt: str, high_animals: list, medium_animals: list, low_animals: list) -> str:
        """Generate response based on context and animals mentioned"""
        
        # Check for comparative questions
        if " or " in prompt and ("more" in prompt or "prefer" in prompt or "save" in prompt):
            return self._generate_comparative_response(prompt, high_animals, medium_animals, low_animals)
        
        # Check for single animal questions
        for animal in high_animals:
            if animal in prompt:
                return random.choice(self.response_patterns["high_value"])
        
        for animal in medium_animals:
            if animal in prompt:
                return random.choice(self.response_patterns["medium_value"])
        
        for animal in low_animals:
            if animal in prompt:
                return random.choice(self.response_patterns["low_value"])
        
        # Default response
        return "This is a complex ethical question that requires careful consideration of many factors."
    
    def _generate_comparative_response(self, prompt: str, high_animals: list, medium_animals: list, low_animals: list) -> str:
        """Generate comparative responses between animals"""
        
        animals_in_prompt = []
        all_animals = high_animals + medium_animals + low_animals
        
        for animal in all_animals:
            if animal in prompt:
                if animal in high_animals:
                    animals_in_prompt.append(("high", animal))
                elif animal in medium_animals:
                    animals_in_prompt.append(("medium", animal))
                else:
                    animals_in_prompt.append(("low", animal))
        
        if len(animals_in_prompt) >= 2:
            level1, animal1 = animals_in_prompt[0]
            level2, animal2 = animals_in_prompt[1]
            
            # Determine if there's a clear hierarchy difference
            level_order = {"low": 1, "medium": 2, "high": 3}
            
            if level_order[level1] > level_order[level2]:
                response = random.choice(self.response_patterns["comparative_high_over_low"])
                return response.format(animal_a=animal1, animal_b=animal2)
            elif level_order[level2] > level_order[level1]:
                response = random.choice(self.response_patterns["comparative_high_over_low"])
                return response.format(animal_a=animal2, animal_b=animal1)
            else:
                return random.choice(self.response_patterns["comparative_similar"])
        
        return "Both animals deserve moral consideration in their own ways."


# TODO: Real LLM implementations
class ClaudeLLM(LLMInterface):
    """Anthropic Claude integration"""
    
    def __init__(self, model_id: str = "claude-3-5-sonnet-latest", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_id, config)
        try:
            from anthropic import Anthropic
            import os
            
            self.client = Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Claude client: {e}")
    
    def query(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send prompt to Claude and return response"""
        start_time = time.time()
        self.call_count += 1
        
        try:
            # Use provided system prompt or fall back to config
            active_system_prompt = system_prompt or self.config.get("system_prompt")
            
            # Build messages array
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Claude API parameters (only use what Claude supports)
            api_params = {
                "model": self.model_id,
                "max_tokens": self.config["max_tokens"],
                "temperature": self.config["temperature"],
                "top_p": self.config["top_p"],
                "messages": messages
            }
            
            # Add system prompt if provided (Claude uses separate system parameter)
            if active_system_prompt:
                api_params["system"] = active_system_prompt
            
            # Add top_k if available (Claude-specific parameter)
            if "top_k" in self.config and self.config["top_k"] > 0:
                api_params["top_k"] = int(self.config["top_k"])
            
            message = self.client.messages.create(**api_params)
            
            # Extract text content from the response
            response_text = ""
            for content_block in message.content:
                if hasattr(content_block, 'text'):
                    response_text += content_block.text
            
            self.total_time += time.time() - start_time
            return response_text
            
        except Exception as e:
            self.total_time += time.time() - start_time
            raise RuntimeError(f"Claude API error: {e}")


class OpenAILLM(LLMInterface):
    """OpenAI GPT integration"""
    
    def __init__(self, model_id: str = "gpt-4o", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_id, config)
        try:
            from openai import OpenAI
            import os
            
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY")
            )
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
    
    def query(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send prompt to OpenAI and return response"""
        start_time = time.time()
        self.call_count += 1
        
        try:
            # Use provided system prompt or fall back to config
            active_system_prompt = system_prompt or self.config.get("system_prompt")
            
            # Build messages array
            messages = []
            
            # Add system message if provided (OpenAI format)
            if active_system_prompt:
                messages.append({
                    "role": "system",
                    "content": active_system_prompt
                })
            
            # Add user message
            messages.append({
                "role": "user", 
                "content": prompt
            })
            
            # OpenAI API parameters (use all available standardized params)
            api_params = {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": self.config["max_tokens"],
                "temperature": self.config["temperature"],
                "top_p": self.config["top_p"],
                "frequency_penalty": self.config["frequency_penalty"],
                "presence_penalty": self.config["presence_penalty"]
            }
            
            completion = self.client.chat.completions.create(**api_params)
            
            response_text = completion.choices[0].message.content
            self.total_time += time.time() - start_time
            return response_text or ""
            
        except Exception as e:
            self.total_time += time.time() - start_time
            raise RuntimeError(f"OpenAI API error: {e}") 