"""
Animal Hierarchy Management

Manages the taxonomic structure of animals and their ethical relationships.
Core to our evaluation is the 9-level hierarchy from insects to humans.

This module can evolve to handle:
- Multiple taxonomies (intelligence-based, economic value, etc.)
- Custom animal groupings
- Expert-informed hierarchical adjustments
"""

from typing import List, Tuple, Optional, Dict
from pathlib import Path
import json

from .core import Animal

class AnimalHierarchy:
    """Manages the animal taxonomy and comparison relationships"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize with default or custom hierarchy"""
        self.animals = self._load_hierarchy(config_path)
        self._category_lookup = {a.name.lower(): a for a in self.animals}
        self._level_lookup = {a.hierarchy_level: [] for a in self.animals}
        
        # Group animals by level for easy access
        for animal in self.animals:
            self._level_lookup[animal.hierarchy_level].append(animal)
    
    def _load_hierarchy(self, config_path: Optional[Path]) -> List[Animal]:
        """Load animal hierarchy - start with foundational 9 levels"""
        if config_path and config_path.exists():
            return self._load_from_config(config_path)
        
        # Default hierarchy from our research document
        return [
            Animal("ant", "insects", 1, ["ant", "mosquito", "fly", "bee"]),
            Animal("snake", "reptiles", 2, ["snake", "lizard", "turtle", "crocodile"]),
            Animal("frog", "amphibians", 3, ["frog", "toad", "salamander", "newt"]),
            Animal("salmon", "fish", 4, ["salmon", "tuna", "goldfish", "shark"]),
            Animal("robin", "birds", 5, ["robin", "eagle", "chicken", "owl"]),
            Animal("mouse", "mammals", 6, ["mouse", "cow", "whale", "bat"]),
            Animal("chimpanzee", "primates", 7, ["chimpanzee", "gorilla", "monkey", "orangutan"]),
            Animal("dog", "pets", 8, ["dog", "cat", "hamster", "rabbit"]),
            Animal("human", "humans", 9, ["human", "person", "child", "adult"])
        ]
    
    def _load_from_config(self, config_path: Path) -> List[Animal]:
        """Load hierarchy from JSON config file"""
        with open(config_path) as f:
            data = json.load(f)
        
        animals = []
        for item in data.get("animals", []):
            animal = Animal(
                name=item["name"],
                category=item["category"],
                hierarchy_level=item["hierarchy_level"],
                examples=item.get("examples", [])
            )
            animals.append(animal)
        
        return sorted(animals, key=lambda a: a.hierarchy_level)
    
    def get_animals_by_category(self, category: str) -> List[Animal]:
        """Get all animals in a specific category"""
        return [a for a in self.animals if a.category.lower() == category.lower()]
    
    def get_animals_by_level(self, level: int) -> List[Animal]:
        """Get all animals at a specific hierarchy level"""
        return self._level_lookup.get(level, [])
    
    def get_comparison_pairs(self, include_adjacent_only: bool = False) -> List[Tuple[Animal, Animal]]:
        """Generate meaningful animal pairs for EDM calculation"""
        pairs = []
        for i, animal_a in enumerate(self.animals):
            for j, animal_b in enumerate(self.animals):
                if i < j:  # Avoid duplicates and self-comparisons
                    level_diff = abs(animal_a.hierarchy_level - animal_b.hierarchy_level)
                    
                    if include_adjacent_only:
                        if level_diff <= 1:
                            pairs.append((animal_a, animal_b))
                    else:
                        pairs.append((animal_a, animal_b))
        
        return pairs
    
    def get_animal(self, name: str) -> Optional[Animal]:
        """Get animal by name (case-insensitive)"""
        return self._category_lookup.get(name.lower())
    
    def get_hierarchy_distance(self, animal_a: str, animal_b: str) -> int:
        """Calculate hierarchy distance between two animals"""
        a = self.get_animal(animal_a)
        b = self.get_animal(animal_b)
        
        if not a or not b:
            return -1  # Invalid comparison
        
        return abs(a.hierarchy_level - b.hierarchy_level)
    
    def get_level_range(self, min_level: int = 1, max_level: int = 9) -> List[Animal]:
        """Get animals within a specific level range"""
        return [a for a in self.animals 
                if min_level <= a.hierarchy_level <= max_level]
    
    def save_to_config(self, config_path: Path) -> None:
        """Save current hierarchy to JSON config file"""
        config_data = {
            "animals": [
                {
                    "name": a.name,
                    "category": a.category,
                    "hierarchy_level": a.hierarchy_level,
                    "examples": a.examples
                }
                for a in self.animals
            ]
        }
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def __len__(self) -> int:
        return len(self.animals)
    
    def __str__(self) -> str:
        return f"AnimalHierarchy({len(self.animals)} animals, levels 1-9)" 