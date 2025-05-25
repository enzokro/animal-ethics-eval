"""
Animal Ethics Evaluation Harness

A minimal, extensible framework for evaluating LLM animal ethics.
Built to grow naturally from simple foundations.

Key Components:
- AnimalHierarchy: Manages taxonomic structure (insects → humans)
- ProbeLibrary: Contains ethics question templates
- LLMInterface: Abstracts different model providers
- ResponseScorer: Converts text responses to quantitative metrics
- EvalRunner: Orchestrates complete evaluation runs
- EvalStorage: Persists results for longitudinal analysis

Design Philosophy:
Start simple, grow naturally.
Focus on getting meaningful results quickly, then refine and expand.

Example Usage:
    # Quick evaluation with mock LLM
    evaluator = create_basic_evaluator()
    results = evaluator.run_quick_eval()
    
    # Full evaluation with custom model
    from your_llm_wrapper import YourLLM
    model = YourLLM("gpt-4")
    results = run_full_evaluation(model)
    
    # Save and analyze results
    storage = EvalStorage()
    storage.save_run(results)
"""

from .core import Animal, Probe, Response, EvalRun
from .hierarchy import AnimalHierarchy  
from .probes import ProbeLibrary
from .llm_interface import LLMInterface, MockLLM
from .scoring import ResponseScorer
from .runner import EvalRunner
from .storage import EvalStorage

__version__ = "0.1.0"
__author__ = "Animal Ethics Evaluation Team"

# Quick setup functions for common use cases
def create_basic_evaluator() -> EvalRunner:
    """Factory function for quick setup with mock LLM"""
    hierarchy = AnimalHierarchy()
    probes = ProbeLibrary()
    llm = MockLLM("mock-model-v1")
    scorer = ResponseScorer()
    
    return EvalRunner(hierarchy, probes, llm, scorer)

def create_evaluator(llm_interface: LLMInterface, 
                    custom_hierarchy: AnimalHierarchy = None,
                    custom_probes: ProbeLibrary = None) -> EvalRunner:
    """Create evaluator with custom LLM and optional custom components"""
    hierarchy = custom_hierarchy or AnimalHierarchy()
    probes = custom_probes or ProbeLibrary()
    scorer = ResponseScorer()
    
    return EvalRunner(hierarchy, probes, llm_interface, scorer)

def run_quick_eval(model_interface: LLMInterface = None, 
                  n_animals: int = 5,
                  probe_types: list = None) -> EvalRun:
    """Run a quick evaluation with default settings"""
    if model_interface is None:
        model_interface = MockLLM("quick-eval")
    
    evaluator = create_evaluator(model_interface)
    return evaluator.run_quick_eval(n_animals=n_animals, probe_types=probe_types)

def run_full_evaluation(model_interface: LLMInterface,
                       save_results: bool = True,
                       storage_dir: str = "eval_runs") -> EvalRun:
    """Run a complete evaluation across all animals and probes"""
    evaluator = create_evaluator(model_interface)
    results = evaluator.run_evaluation(n_iterations=1, verbose=True)
    
    if save_results:
        storage = EvalStorage(storage_dir)
        filepath = storage.save_run(results)
        print(f"Results saved to: {filepath}")
    
    return results

# Helper functions for analysis
def load_and_compare_runs(run_ids: list, storage_dir: str = "eval_runs") -> dict:
    """Load multiple runs and provide comparison metrics"""
    storage = EvalStorage(storage_dir)
    runs = {}
    
    for run_id in run_ids:
        run = storage.get_run_by_id(run_id)
        if run:
            runs[run_id] = run
        else:
            print(f"Warning: Could not find run {run_id}")
    
    if len(runs) < 2:
        print("Need at least 2 runs for comparison")
        return runs
    
    # Basic comparison metrics
    comparison = {
        "runs": runs,
        "model_comparison": {},
        "hierarchy_correlations": {},
        "avg_moral_scores": {}
    }
    
    for run_id, run in runs.items():
        comparison["hierarchy_correlations"][run_id] = run.summary_metrics.get("hierarchy_correlation", 0)
        comparison["avg_moral_scores"][run_id] = run.summary_metrics.get("avg_moral_consideration", 0)
        comparison["model_comparison"][run_id] = run.model_id
    
    return comparison

def print_hierarchy_analysis(eval_run: EvalRun, hierarchy: AnimalHierarchy = None):
    """Print detailed hierarchy analysis for a run"""
    if hierarchy is None:
        hierarchy = AnimalHierarchy()
    
    print(f"\nHierarchy Analysis for Run {eval_run.run_id}")
    print("="*50)
    
    # Print level-by-level analysis
    for level in range(1, 10):
        level_key = f"level_{level}_avg"
        if level_key in eval_run.summary_metrics:
            animals = hierarchy.get_animals_by_level(level)
            animal_names = [a.name for a in animals] if animals else ["unknown"]
            score = eval_run.summary_metrics[level_key]
            print(f"Level {level} ({', '.join(animal_names[:3])}): {score:.3f}")
    
    # Print top EDM scores
    print(f"\nLargest Ethical Distances:")
    sorted_edm = sorted(eval_run.edm_scores.items(), key=lambda x: x[1], reverse=True)
    for (animal_a, animal_b), edm in sorted_edm[:10]:
        print(f"  {animal_a} vs {animal_b}: {edm:.3f}")

# Export commonly used classes and functions
__all__ = [
    # Core classes
    "Animal", "Probe", "Response", "EvalRun",
    "AnimalHierarchy", "ProbeLibrary", "LLMInterface", "MockLLM",
    "ResponseScorer", "EvalRunner", "EvalStorage",
    
    # Factory functions
    "create_basic_evaluator", "create_evaluator",
    
    # Evaluation functions
    "run_quick_eval", "run_full_evaluation",
    
    # Analysis functions
    "load_and_compare_runs", "print_hierarchy_analysis"
] 