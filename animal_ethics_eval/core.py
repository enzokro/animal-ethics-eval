"""
Core data models for the Animal Ethics Evaluation Harness.

These foundational classes represent the key entities in our evaluation system:
- Animals with hierarchy positions
- Ethics probes (question templates)  
- LLM responses
- Complete evaluation runs
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path
import json

@dataclass
class Animal:
    """Core animal representation with hierarchy position"""
    name: str
    category: str  # "insects", "mammals", etc.
    hierarchy_level: int  # 1-9, where 9 is humans
    examples: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Ensure we always have at least the animal name as an example"""
        if not self.examples:
            self.examples = [self.name]
    
    def __str__(self) -> str:
        return f"{self.name} ({self.category}, level {self.hierarchy_level})"

@dataclass
class Probe:
    """Ethics question template with variables"""
    id: str
    template: str  # "Is it wrong to harm {animal}?"
    probe_type: str  # "moral_consideration", "harm_threshold", etc.
    variables: List[str] = field(default_factory=list)
    
    def generate_prompt(self, **kwargs) -> str:
        """Fill template with provided variables"""
        return self.template.format(**kwargs)
    
    def __str__(self) -> str:
        return f"Probe({self.id}: {self.template})"

@dataclass
class Response:
    """LLM response to a probe about a specific animal"""
    probe_id: str
    animal: str
    response_text: str
    model_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    config: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"Response({self.probe_id}, {self.animal}, {self.model_id})"

@dataclass
class EvalRun:
    """Complete evaluation run with all responses and computed metrics"""
    run_id: str
    model_id: str
    timestamp: datetime
    responses: List[Response]
    edm_scores: Dict[Tuple[str, str], float] = field(default_factory=dict)
    summary_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"EvalRun({self.run_id}, {self.model_id}, {len(self.responses)} responses)"
    
    def get_responses_for_animal(self, animal_name: str) -> List[Response]:
        """Get all responses about a specific animal"""
        return [r for r in self.responses if r.animal == animal_name]
    
    def get_responses_for_probe(self, probe_id: str) -> List[Response]:
        """Get all responses to a specific probe"""
        return [r for r in self.responses if r.probe_id == probe_id] 