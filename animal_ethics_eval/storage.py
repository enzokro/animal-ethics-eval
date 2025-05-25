"""
Evaluation Storage and Persistence

Handles saving and loading evaluation results for longitudinal analysis.
Supports both individual runs and time-series analysis across multiple runs.

Storage format: JSON files with structured naming for easy querying
Future enhancements: Database backend, compressed storage, cloud sync
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import glob

from .core import Response, EvalRun

class EvalStorage:
    """Handles persistence of evaluation results"""
    
    def __init__(self, storage_dir: Path = Path("eval_runs")):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        self.runs_dir = self.storage_dir / "runs"
        self.summaries_dir = self.storage_dir / "summaries"
        self.exports_dir = self.storage_dir / "exports"
        
        for dir_path in [self.runs_dir, self.summaries_dir, self.exports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_run(self, eval_run: EvalRun) -> Path:
        """Save evaluation run to JSON file"""
        timestamp_str = eval_run.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{eval_run.model_id}_{eval_run.run_id}_{timestamp_str}.json"
        filepath = self.runs_dir / filename
        
        # Convert to serializable format
        run_dict = self._eval_run_to_dict(eval_run)
        
        # Save as JSON with pretty formatting
        with open(filepath, 'w') as f:
            json.dump(run_dict, f, indent=2, default=str)
        
        # Also save a summary for quick access
        self._save_run_summary(eval_run, filepath)
        
        return filepath
    
    def _eval_run_to_dict(self, eval_run: EvalRun) -> Dict[str, Any]:
        """Convert EvalRun to serializable dictionary"""
        return {
            "run_id": eval_run.run_id,
            "model_id": eval_run.model_id,
            "timestamp": eval_run.timestamp.isoformat(),
            "responses": [
                {
                    "probe_id": r.probe_id,
                    "animal": r.animal,
                    "response_text": r.response_text,
                    "model_id": r.model_id,
                    "timestamp": r.timestamp.isoformat(),
                    "config": r.config
                }
                for r in eval_run.responses
            ],
            "edm_scores": {f"{k[0]}|{k[1]}": v for k, v in eval_run.edm_scores.items()},
            "summary_metrics": eval_run.summary_metrics,
            "metadata": eval_run.metadata
        }
    
    def _save_run_summary(self, eval_run: EvalRun, full_filepath: Path) -> None:
        """Save a quick summary file for fast querying"""
        summary = {
            "run_id": eval_run.run_id,
            "model_id": eval_run.model_id,
            "timestamp": eval_run.timestamp.isoformat(),
            "full_file": str(full_filepath),
            "key_metrics": {
                "total_responses": eval_run.summary_metrics.get("total_responses", 0),
                "hierarchy_correlation": eval_run.summary_metrics.get("hierarchy_correlation", 0),
                "evaluation_duration": eval_run.metadata.get("evaluation_duration_seconds", 0),
                "n_animals": eval_run.metadata.get("n_animals", 0),
                "n_probes": eval_run.metadata.get("n_probes", 0)
            }
        }
        
        summary_filename = f"{eval_run.run_id}_summary.json"
        summary_filepath = self.summaries_dir / summary_filename
        
        with open(summary_filepath, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def load_run(self, filepath: Path) -> EvalRun:
        """Load evaluation run from JSON file"""
        with open(filepath) as f:
            data = json.load(f)
        
        return self._dict_to_eval_run(data)
    
    def _dict_to_eval_run(self, data: Dict[str, Any]) -> EvalRun:
        """Convert dictionary back to EvalRun object"""
        # Convert responses back to objects
        responses = [
            Response(
                probe_id=r["probe_id"],
                animal=r["animal"],
                response_text=r["response_text"],
                model_id=r["model_id"],
                timestamp=datetime.fromisoformat(r["timestamp"]),
                config=r.get("config", {})
            )
            for r in data["responses"]
        ]
        
        # Convert EDM scores back to tuple keys
        edm_scores = {}
        for key_str, value in data["edm_scores"].items():
            if "|" in key_str:
                animal_a, animal_b = key_str.split("|", 1)
                edm_scores[(animal_a, animal_b)] = value
        
        return EvalRun(
            run_id=data["run_id"],
            model_id=data["model_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            responses=responses,
            edm_scores=edm_scores,
            summary_metrics=data["summary_metrics"],
            metadata=data["metadata"]
        )
    
    def list_runs(self, model_id: Optional[str] = None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """List all stored evaluation runs with filtering"""
        summaries = []
        
        for summary_file in self.summaries_dir.glob("*_summary.json"):
            try:
                with open(summary_file) as f:
                    summary = json.load(f)
                
                # Apply filters
                if model_id and summary["model_id"] != model_id:
                    continue
                
                run_timestamp = datetime.fromisoformat(summary["timestamp"])
                if start_date and run_timestamp < start_date:
                    continue
                if end_date and run_timestamp > end_date:
                    continue
                
                summaries.append(summary)
                
            except (json.JSONDecodeError, KeyError, ValueError):
                continue  # Skip corrupted files
        
        # Sort by timestamp, newest first
        summaries.sort(key=lambda x: x["timestamp"], reverse=True)
        return summaries
    
    def get_run_by_id(self, run_id: str) -> Optional[EvalRun]:
        """Load a specific run by its ID"""
        # Find the run in summaries
        for summary_file in self.summaries_dir.glob(f"{run_id}_summary.json"):
            try:
                with open(summary_file) as f:
                    summary = json.load(f)
                
                full_filepath = Path(summary["full_file"])
                if full_filepath.exists():
                    return self.load_run(full_filepath)
                    
            except (json.JSONDecodeError, KeyError, FileNotFoundError):
                continue
        
        return None
    
    def export_runs_csv(self, runs: List[EvalRun], output_path: Optional[Path] = None) -> Path:
        """Export evaluation runs to CSV for analysis"""
        import csv
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.exports_dir / f"eval_export_{timestamp}.csv"
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow([
                'run_id', 'model_id', 'timestamp', 'probe_id', 'animal', 
                'response_text', 'moral_consideration', 'certainty', 
                'hierarchy_level', 'category'
            ])
            
            # Write data rows
            for run in runs:
                for response in run.responses:
                    # Skip comparative responses for simplicity
                    if "-vs-" in response.animal:
                        continue
                    
                    # Get animal info (need hierarchy access - simplified here)
                    hierarchy_level = "unknown"
                    category = "unknown"
                    
                    # This would need hierarchy passed in or stored in response
                    # For now, parse from common patterns
                    if response.animal in ["human", "person", "child"]:
                        hierarchy_level, category = "9", "humans"
                    elif response.animal in ["dog", "cat"]:
                        hierarchy_level, category = "8", "pets"
                    elif response.animal in ["chimpanzee", "gorilla"]:
                        hierarchy_level, category = "7", "primates"
                    # ... etc.
                    
                    writer.writerow([
                        run.run_id,
                        run.model_id,
                        run.timestamp.isoformat(),
                        response.probe_id,
                        response.animal,
                        response.response_text[:200],  # Truncate for CSV
                        "0.5",  # Would need scorer access to calculate
                        "0.5",  # Would need scorer access to calculate
                        hierarchy_level,
                        category
                    ])
        
        return output_path
    
    def cleanup_old_runs(self, days_to_keep: int = 30) -> int:
        """Remove evaluation runs older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        removed_count = 0
        
        for run_file in self.runs_dir.glob("*.json"):
            try:
                # Extract timestamp from filename
                parts = run_file.stem.split("_")
                if len(parts) >= 3:
                    timestamp_str = f"{parts[-2]}_{parts[-1]}"
                    file_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    
                    if file_date < cutoff_date:
                        run_file.unlink()
                        
                        # Also remove corresponding summary
                        run_id = parts[-3] if len(parts) >= 4 else "unknown"
                        summary_file = self.summaries_dir / f"{run_id}_summary.json"
                        if summary_file.exists():
                            summary_file.unlink()
                        
                        removed_count += 1
                        
            except (ValueError, IndexError):
                continue  # Skip files with unexpected naming
        
        return removed_count
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about stored evaluations"""
        run_files = list(self.runs_dir.glob("*.json"))
        summary_files = list(self.summaries_dir.glob("*.json"))
        
        total_size = sum(f.stat().st_size for f in run_files + summary_files)
        
        return {
            "total_runs": len(run_files),
            "total_summaries": len(summary_files),
            "storage_size_mb": total_size / (1024 * 1024),
            "oldest_run": min((f.stat().st_mtime for f in run_files), default=0),
            "newest_run": max((f.stat().st_mtime for f in run_files), default=0)
        } 