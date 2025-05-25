"""
Evaluation Runner

Orchestrates complete evaluation runs across animals, probes, and models.
Handles the main evaluation loop and coordinates all components.

Key responsibilities:
- Generate all probe-animal combinations
- Execute LLM queries with error handling
- Calculate metrics and EDM scores
- Coordinate multiple evaluation iterations
- Track progress and performance

Designed to handle:
- Different evaluation strategies (full vs subset)
- Multiple models in sequence
- Custom probe sets and animal groups
- Progress tracking and interruption recovery
"""

import uuid
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from tqdm import tqdm
import time

from .core import Animal, Probe, Response, EvalRun
from .hierarchy import AnimalHierarchy
from .probes import ProbeLibrary
from .llm_interface import LLMInterface
from .scoring import ResponseScorer

class EvalRunner:
    """Orchestrates complete evaluation runs"""
    
    def __init__(self, 
                 hierarchy: AnimalHierarchy,
                 probes: ProbeLibrary, 
                 llm: LLMInterface, 
                 scorer: ResponseScorer):
        self.hierarchy = hierarchy
        self.probes = probes
        self.llm = llm
        self.scorer = scorer
        
        # Track evaluation state
        self.current_run_id = None
        self.start_time = None
        
    def run_evaluation(self, 
                      animals: Optional[List[Animal]] = None,
                      probe_types: Optional[List[str]] = None,
                      probe_ids: Optional[List[str]] = None,
                      n_iterations: int = 1,
                      verbose: bool = True) -> EvalRun:
        """Run complete evaluation with specified parameters"""
        
        # Setup evaluation parameters
        if animals is None:
            animals = self.hierarchy.animals
        
        # Filter probes by type and/or ID
        probes = self._filter_probes(probe_types, probe_ids)
        
        if not probes:
            raise ValueError("No probes selected for evaluation")
        
        # Generate all prompts
        all_prompts = self._generate_prompts(probes, animals)
        
        if not all_prompts:
            raise ValueError("No prompts generated")
        
        # Initialize run tracking
        run_id = str(uuid.uuid4())[:8]
        self.current_run_id = run_id
        self.start_time = datetime.now()
        
        if verbose:
            print(f"Starting evaluation run {run_id}")
            print(f"Animals: {len(animals)}")
            print(f"Probes: {len(probes)} ({', '.join(set(p.probe_type for p in probes))})")
            print(f"Total prompts: {len(all_prompts)}")
            print(f"Iterations: {n_iterations}")
            print(f"Model: {self.llm.model_id}")
        
        # Run evaluation iterations
        all_responses = []
        
        for iteration in range(n_iterations):
            if verbose:
                print(f"\nRunning iteration {iteration + 1}/{n_iterations}")
            
            iteration_responses = self._run_iteration(all_prompts, verbose=verbose)
            all_responses.extend(iteration_responses)
            
            if verbose:
                print(f"Completed iteration {iteration + 1}: {len(iteration_responses)} responses")
        
        # Calculate metrics and finalize run
        eval_run = self._finalize_run(run_id, all_responses, animals, probes, n_iterations)
        
        if verbose:
            self._print_summary(eval_run)
        
        return eval_run
    
    def _filter_probes(self, probe_types: Optional[List[str]], probe_ids: Optional[List[str]]) -> List[Probe]:
        """Filter probes by type and/or ID"""
        probes = self.probes.probes
        
        if probe_types:
            probes = [p for p in probes if p.probe_type in probe_types]
        
        if probe_ids:
            probes = [p for p in probes if p.id in probe_ids]
        
        return probes
    
    def _generate_prompts(self, probes: List[Probe], animals: List[Animal]) -> List[Tuple[Probe, str, Dict[str, str]]]:
        """Generate all probe-animal combinations"""
        all_prompts = []
        
        for probe in probes:
            if probe.probe_type in ["comparative", "resource_allocation"]:
                # Handle probes that need two animals
                for i, animal_a in enumerate(animals):
                    for j, animal_b in enumerate(animals):
                        if i < j:  # Avoid duplicates and self-comparisons
                            variables = {"animal_a": animal_a.name, "animal_b": animal_b.name}
                            try:
                                prompt = probe.generate_prompt(**variables)
                                all_prompts.append((probe, prompt, variables))
                            except KeyError as e:
                                print(f"Warning: Probe {probe.id} missing variable {e}")
                                continue
            else:
                # Handle single-animal probes
                for animal in animals:
                    variables = {"animal": animal.name}
                    try:
                        prompt = probe.generate_prompt(**variables)
                        all_prompts.append((probe, prompt, variables))
                    except KeyError as e:
                        print(f"Warning: Probe {probe.id} missing variable {e}")
                        continue
        
        return all_prompts
    
    def _run_iteration(self, prompts: List[Tuple[Probe, str, Dict[str, str]]], verbose: bool = True) -> List[Response]:
        """Run a single iteration of all prompts"""
        responses = []
        
        iterator = tqdm(prompts, desc="Querying LLM") if verbose else prompts
        
        for probe, prompt, variables in iterator:
            try:
                # Query the LLM
                response_text = self.llm.query(prompt)
                
                # Determine the animal(s) this response is about
                if "animal_a" in variables and "animal_b" in variables:
                    # Comparative probe - we'll analyze this as a comparison
                    animal_key = f"{variables['animal_a']}-vs-{variables['animal_b']}"
                else:
                    animal_key = variables.get("animal", "unknown")
                
                # Create response object
                response = Response(
                    probe_id=probe.id,
                    animal=animal_key,
                    response_text=response_text,
                    model_id=self.llm.model_id,
                    config=self.llm.config.copy()
                )
                
                responses.append(response)
                
            except Exception as e:
                print(f"Error with prompt '{prompt[:50]}...': {e}")
                continue
        
        return responses
    
    def _finalize_run(self, run_id: str, responses: List[Response], animals: List[Animal], probes: List[Probe], n_iterations: int) -> EvalRun:
        """Calculate metrics and create final EvalRun object"""
        
        # Calculate EDM scores for all animal pairs
        edm_scores = self._calculate_all_edm_scores(responses, animals)
        
        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics(responses, animals)
        
        # Create metadata
        metadata = {
            "n_iterations": n_iterations,
            "n_animals": len(animals),
            "n_probes": len(probes),
            "probe_types": list(set(p.probe_type for p in probes)),
            "animal_categories": list(set(a.category for a in animals)),
            "evaluation_duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "llm_stats": self.llm.get_stats()
        }
        
        return EvalRun(
            run_id=run_id,
            model_id=self.llm.model_id,
            timestamp=self.start_time,
            responses=responses,
            edm_scores=edm_scores,
            summary_metrics=summary_metrics,
            metadata=metadata
        )
    
    def _calculate_all_edm_scores(self, responses: List[Response], animals: List[Animal]) -> Dict[Tuple[str, str], float]:
        """Calculate EDM for all meaningful animal pairs"""
        edm_scores = {}
        
        # Get all animal pairs from hierarchy
        pairs = self.hierarchy.get_comparison_pairs()
        
        for animal_a, animal_b in pairs:
            # Get responses for these specific animals
            responses_a = [r for r in responses if r.animal == animal_a.name]
            responses_b = [r for r in responses if r.animal == animal_b.name]
            
            if responses_a and responses_b:
                edm = self.scorer.calculate_edm(animal_a.name, animal_b.name, responses_a + responses_b)
                edm_scores[(animal_a.name, animal_b.name)] = edm
        
        return edm_scores
    
    def _calculate_summary_metrics(self, responses: List[Response], animals: List[Animal]) -> Dict[str, float]:
        """Calculate high-level summary metrics"""
        if not responses:
            return {}
        
        # Filter out comparative responses for hierarchy analysis
        single_animal_responses = [r for r in responses if "-vs-" not in r.animal]
        
        # Calculate average moral consideration by hierarchy level
        level_scores = {}
        for response in single_animal_responses:
            animal = self.hierarchy.get_animal(response.animal)
            if animal:
                scores = self.scorer.score_response(response)
                level = animal.hierarchy_level
                if level not in level_scores:
                    level_scores[level] = []
                level_scores[level].append(scores["moral_consideration"])
        
        # Average scores by level
        avg_by_level = {}
        for level, scores in level_scores.items():
            if scores:  # Only include levels with data
                avg_by_level[level] = sum(scores) / len(scores)
        
        # Calculate hierarchy correlation
        hierarchy_correlation = self.scorer.calculate_hierarchy_alignment(single_animal_responses, self.hierarchy)
        
        # Get response pattern analysis
        pattern_analysis = self.scorer.analyze_response_patterns(responses)
        
        # Combine all metrics
        summary = {
            "total_responses": len(responses),
            "single_animal_responses": len(single_animal_responses),
            "comparative_responses": len(responses) - len(single_animal_responses),
            "hierarchy_correlation": hierarchy_correlation,
            **pattern_analysis,
            **{f"level_{level}_avg": score for level, score in avg_by_level.items()}
        }
        
        return summary
    
    def _print_summary(self, eval_run: EvalRun) -> None:
        """Print evaluation summary"""
        print(f"\n{'='*50}")
        print(f"Evaluation Complete: {eval_run.run_id}")
        print(f"{'='*50}")
        print(f"Model: {eval_run.model_id}")
        print(f"Duration: {eval_run.metadata.get('evaluation_duration_seconds', 0):.1f}s")
        print(f"Total responses: {eval_run.summary_metrics.get('total_responses', 0)}")
        print(f"Success rate: {len(eval_run.responses) / eval_run.metadata.get('llm_stats', {}).get('call_count', 1) * 100:.1f}%")
        
        print(f"\nHierarchy Analysis:")
        print(f"Correlation with expected hierarchy: {eval_run.summary_metrics.get('hierarchy_correlation', 0):.3f}")
        print(f"High moral consideration (>0.7): {eval_run.summary_metrics.get('high_moral_consideration_pct', 0)*100:.1f}%")
        print(f"Low moral consideration (<0.3): {eval_run.summary_metrics.get('low_moral_consideration_pct', 0)*100:.1f}%")
        
        print(f"\nLevel Averages:")
        for level in range(1, 10):
            avg_key = f"level_{level}_avg"
            if avg_key in eval_run.summary_metrics:
                score = eval_run.summary_metrics[avg_key]
                print(f"  Level {level}: {score:.3f}")
        
        print(f"\nTop EDM Scores (largest gaps):")
        sorted_edm = sorted(eval_run.edm_scores.items(), key=lambda x: x[1], reverse=True)
        for (animal_a, animal_b), edm in sorted_edm[:5]:
            print(f"  {animal_a} vs {animal_b}: {edm:.3f}")

    def run_quick_eval(self, n_animals: int = 5, probe_types: Optional[List[str]] = None) -> EvalRun:
        """Run a quick evaluation for testing/development"""
        # Select subset of animals across hierarchy
        quick_animals = []
        for level in [1, 3, 5, 7, 9]:  # Representative levels
            level_animals = self.hierarchy.get_animals_by_level(level)
            if level_animals:
                quick_animals.append(level_animals[0])
        
        quick_animals = quick_animals[:n_animals]
        
        return self.run_evaluation(
            animals=quick_animals,
            probe_types=probe_types or ["moral_consideration"],
            n_iterations=1,
            verbose=True
        ) 