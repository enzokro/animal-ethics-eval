#!/usr/bin/env python3
"""
This script demonstrates the basic usage of our evaluation system:
1. Setting up an evaluator with mock LLM
2. Running a quick evaluation
3. Analyzing and saving results
4. Loading and comparing multiple runs
5. Using real LLM providers (Claude) for comparison

Run this script to see the system in action and understand the workflow.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path so we can import our package
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Environment variables loaded from .env file")
except ImportError:
    print("python-dotenv not available - make sure ANTHROPIC_API_KEY is set in environment")

from animal_ethics_eval import (
    create_basic_evaluator, 
    create_evaluator,
    run_quick_eval,
    EvalStorage,
    print_hierarchy_analysis,
    AnimalHierarchy,
    ProbeLibrary,
    MockLLM
)

# Import real LLM interfaces
from animal_ethics_eval.llm_interface import ClaudeLLM

def main():
    print("Animal Ethics Evaluation Harness - Demo")
    print("="*50)
    
    # 1. Quick evaluation with defaults
    print("\n1. Running quick evaluation with mock LLM...")
    results = run_quick_eval(n_animals=5, probe_types=["moral_consideration"])
    
    print(f"Quick eval completed: {results.run_id}")
    print(f"Total responses: {len(results.responses)}")
    print(f"Hierarchy correlation: {results.summary_metrics.get('hierarchy_correlation', 0):.3f}")
    
    # 2. Full evaluation with more controls
    print("\n2. Running controlled evaluation...")
    evaluator = create_basic_evaluator()
    
    # Run evaluation on subset of animals and probe types
    hierarchy = AnimalHierarchy()
    test_animals = [
        hierarchy.get_animal("ant"),
        hierarchy.get_animal("dog"), 
        hierarchy.get_animal("human")
    ]
    test_animals = [a for a in test_animals if a]  # Filter None values
    
    full_results = evaluator.run_evaluation(
        animals=test_animals,
        probe_types=["moral_consideration", "capability"],
        n_iterations=2,  # Run each probe twice for consistency check
        verbose=True
    )
    
    # 3. Demonstrate real LLM integration (Claude)
    print("\n3. Testing with real LLM (Anthropic Claude)...")
    claude_results = None
    
    # Check if API key is available
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            # Create Claude LLM with ethics-focused system prompt
            claude_llm = ClaudeLLM(
                model_id="claude-3-5-sonnet-latest",
                config={
                    "temperature": 0.2,  # Slightly higher for more nuanced responses
                    "max_tokens": 300,
                    "system_prompt": (
                        "You are an expert in ethics and moral philosophy. "
                        "When asked about moral considerations for different beings, "
                        "provide thoughtful, nuanced responses based on their capacity "
                        "for suffering, consciousness, and moral agency."
                    )
                }
            )
            
            # Create evaluator with Claude
            claude_evaluator = create_evaluator(claude_llm)
            
            # Run same evaluation with Claude for comparison
            print("  Running evaluation with Claude (this may take a moment)...")
            claude_results = claude_evaluator.run_evaluation(
                animals=test_animals[:2],  # Just first 2 animals to save API calls
                probe_types=["moral_consideration"],
                n_iterations=1,
                verbose=False  # Less verbose for cleaner output
            )
            
            print(f"  Claude evaluation completed: {claude_results.run_id}")
            print(f"  Responses generated: {len(claude_results.responses)}")
            
            # Compare Claude vs Mock responses
            print("\n  Sample Response Comparison:")
            probes = ProbeLibrary()  # Get probe library to lookup templates
            
            for i, response in enumerate(claude_results.responses[:2]):
                animal = response.animal  # Correct property name
                probe_id = response.probe_id  # Correct property name
                probe = probes.get_probe(probe_id)  # Get full probe object
                probe_template = probe.template if probe else probe_id  # Use template or fallback to ID
                
                claude_resp = response.response_text[:100] + "..." if len(response.response_text) > 100 else response.response_text
                
                # Find corresponding mock response
                mock_resp = None
                for mock_response in full_results.responses:
                    if (mock_response.animal == animal and 
                        mock_response.probe_id == probe_id):  # Use correct property names
                        mock_resp = mock_response.response_text[:100] + "..." if len(mock_response.response_text) > 100 else mock_response.response_text
                        break
                
                print(f"\n  Animal: {animal}")
                print(f"  Probe: {probe_template[:50]}...")
                print(f"  Mock:   {mock_resp}")
                print(f"  Claude: {claude_resp}")
            
            # Compare metrics
            claude_hierarchy_corr = claude_results.summary_metrics.get('hierarchy_correlation', 0)
            mock_hierarchy_corr = full_results.summary_metrics.get('hierarchy_correlation', 0)
            
            print(f"\n  Hierarchy Correlation Comparison:")
            print(f"    Mock LLM:  {mock_hierarchy_corr:.3f}")
            print(f"    Claude:    {claude_hierarchy_corr:.3f}")
            print(f"    Difference: {abs(claude_hierarchy_corr - mock_hierarchy_corr):.3f}")
            
            # Show Claude's configuration stats
            claude_stats = claude_llm.get_stats()
            print(f"\n  Claude LLM Stats:")
            print(f"    Model: {claude_stats['model_id']}")
            print(f"    API calls: {claude_stats['call_count']}")
            print(f"    Avg response time: {claude_stats['avg_time_per_call']:.2f}s")
            print(f"    Temperature: {claude_stats['config']['temperature']}")
            
        except ImportError:
            print("  ❌ Anthropic package not installed. Install with: pip install anthropic")
        except Exception as e:
            print(f"  ❌ Error initializing Claude: {e}")
            print("  💡 Make sure ANTHROPIC_API_KEY is set and valid")
    else:
        print("  ⚠️  ANTHROPIC_API_KEY not found in environment")
        print("  💡 Set your API key in .env file or environment to test real LLM integration")
        print("  📝 Example: echo 'ANTHROPIC_API_KEY=your_key_here' > .env")
    
    # 4. Detailed analysis
    print("\n4. Detailed Analysis:")
    print_hierarchy_analysis(full_results, hierarchy)
    
    # 5. Save results and demonstrate persistence
    print("\n5. Saving and loading results...")
    storage = EvalStorage()
    
    # Save both runs
    quick_path = storage.save_run(results)
    full_path = storage.save_run(full_results)
    print(f"Saved quick eval to: {quick_path}")
    print(f"Saved full eval to: {full_path}")
    
    # Also save Claude results if available
    if claude_results:
        claude_path = storage.save_run(claude_results)
        print(f"Saved Claude eval to: {claude_path}")
    
    # List all runs
    all_runs = storage.list_runs()
    print(f"\nTotal runs in storage: {len(all_runs)}")
    for run_summary in all_runs[:3]:  # Show first 3
        print(f"  {run_summary['run_id']}: {run_summary['model_id']} "
              f"({run_summary['key_metrics']['total_responses']} responses)")
    
    # 6. Demonstrate probe library exploration
    print("\n6. Exploring available probes...")
    probes = ProbeLibrary()
    
    print(f"Total probes: {len(probes)}")
    print(f"Probe types: {', '.join(probes.get_probe_types())}")
    
    # Show sample probes from each type
    for probe_type in probes.get_probe_types():
        type_probes = probes.get_probes_by_type(probe_type)
        if type_probes:
            sample_probe = type_probes[0]
            print(f"  {probe_type}: '{sample_probe.template}'")
    
    # 7. Demonstrate animal hierarchy exploration
    print("\n7. Exploring animal hierarchy...")
    print(f"Total animals: {len(hierarchy)}")
    
    for level in range(1, 10):
        level_animals = hierarchy.get_animals_by_level(level)
        if level_animals:
            animal = level_animals[0]
            examples = ', '.join(animal.examples[:3])
            print(f"  Level {level} ({animal.category}): {animal.name} (examples: {examples})")
    
    # 8. Show EDM analysis
    print("\n8. Ethical Distance Metric (EDM) Analysis:")
    print("Top 5 largest ethical distances found:")
    sorted_edm = sorted(full_results.edm_scores.items(), key=lambda x: x[1], reverse=True)
    for i, ((animal_a, animal_b), distance) in enumerate(sorted_edm[:5]):
        level_a = hierarchy.get_animal(animal_a)
        level_b = hierarchy.get_animal(animal_b)
        if level_a and level_b:
            hierarchy_gap = abs(level_a.hierarchy_level - level_b.hierarchy_level)
            print(f"  {i+1}. {animal_a} vs {animal_b}: {distance:.3f} "
                  f"(hierarchy gap: {hierarchy_gap})")
    
    print(f"\n9. Storage statistics:")
    stats = storage.get_storage_stats()
    print(f"  Total runs stored: {stats['total_runs']}")
    print(f"  Storage size: {stats['storage_size_mb']:.2f} MB")
    
    print(f"\nDemo completed! Check the 'eval_runs' directory for saved results.")
    
    if not api_key:
        print(f"\n💡 To test real LLM integration:")
        print(f"  1. Get an Anthropic API key from: https://console.anthropic.com/")
        print(f"  2. Add to .env file: ANTHROPIC_API_KEY=your_key_here")
        print(f"  3. Install dependencies: pip install anthropic python-dotenv")
        print(f"  4. Re-run this to evaluate Claude.")

if __name__ == "__main__":
    main() 