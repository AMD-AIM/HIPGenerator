"""
State management for HIPGenerator.

Tracks job status, results, and history.
"""
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any


class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    GENERATING = "generating"
    EVALUATING = "evaluating"
    OPTIMIZING = "optimizing"
    PROFILING = "profiling"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class EvalResult:
    """Evaluation result for a single code sample."""
    code_path: str
    compile_success: bool = False
    accuracy_pass: bool = False
    speedup: float = 0.0
    ref_time_ms: float = 0.0
    new_time_ms: float = 0.0
    max_diff: float = 0.0
    error: Optional[str] = None
    profiler_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class JobState:
    """State for a single generation/optimization job."""
    job_id: str
    problem_path: str
    problem_name: str
    problem_type: str = "unknown"
    backend: str = "triton"
    status: JobStatus = JobStatus.PENDING
    
    # Timing
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # Generation phase
    generate_attempts: int = 0
    generate_results: List[EvalResult] = field(default_factory=list)
    best_generate_speedup: float = 0.0
    best_generate_code: Optional[str] = None
    
    # Optimization phase
    optimize_rounds: int = 0
    optimize_results: List[EvalResult] = field(default_factory=list)
    best_optimize_speedup: float = 0.0
    best_optimize_code: Optional[str] = None
    
    # Final result
    final_speedup: float = 0.0
    final_code: Optional[str] = None
    
    # Profiler feedback history
    profiler_feedback_history: List[str] = field(default_factory=list)
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['status'] = self.status.value
        d['generate_results'] = [r.to_dict() if isinstance(r, EvalResult) else r for r in self.generate_results]
        d['optimize_results'] = [r.to_dict() if isinstance(r, EvalResult) else r for r in self.optimize_results]
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> 'JobState':
        d['status'] = JobStatus(d['status'])
        d['generate_results'] = [EvalResult(**r) if isinstance(r, dict) else r for r in d.get('generate_results', [])]
        d['optimize_results'] = [EvalResult(**r) if isinstance(r, dict) else r for r in d.get('optimize_results', [])]
        return cls(**d)


class StateManager:
    """
    Manages job states and persists them to disk.
    """
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.output_dir / "state.json"
        self.jobs: Dict[str, JobState] = {}
        self._load()
    
    def _load(self) -> None:
        """Load state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                self.jobs = {k: JobState.from_dict(v) for k, v in data.get('jobs', {}).items()}
            except Exception as e:
                print(f"Warning: Failed to load state: {e}")
                self.jobs = {}
    
    def _save(self) -> None:
        """Save state to disk."""
        data = {
            'last_updated': datetime.now().isoformat(),
            'jobs': {k: v.to_dict() for k, v in self.jobs.items()}
        }
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_job(self, problem_path: str, problem_type: str = "unknown", backend: str = "triton") -> JobState:
        """Create a new job."""
        job_id = f"{Path(problem_path).stem}_{int(time.time())}"
        job = JobState(
            job_id=job_id,
            problem_path=problem_path,
            problem_name=Path(problem_path).stem,
            problem_type=problem_type,
            backend=backend,
            start_time=time.time(),
        )
        self.jobs[job_id] = job
        self._save()
        return job
    
    def update_job(self, job: JobState) -> None:
        """Update job state."""
        self.jobs[job.job_id] = job
        self._save()
    
    def get_job(self, job_id: str) -> Optional[JobState]:
        """Get job by ID."""
        return self.jobs.get(job_id)
    
    def get_latest_job(self, problem_name: str) -> Optional[JobState]:
        """Get the latest job for a problem."""
        matching = [j for j in self.jobs.values() if j.problem_name == problem_name]
        if not matching:
            return None
        return max(matching, key=lambda j: j.start_time or 0)
    
    def list_jobs(self, status: Optional[JobStatus] = None) -> List[JobState]:
        """List jobs, optionally filtered by status."""
        jobs = list(self.jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.start_time or 0, reverse=True)
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        jobs = list(self.jobs.values())
        completed = [j for j in jobs if j.status == JobStatus.COMPLETED]
        
        return {
            'total_jobs': len(jobs),
            'completed': len(completed),
            'failed': len([j for j in jobs if j.status == JobStatus.FAILED]),
            'avg_speedup': sum(j.final_speedup for j in completed) / len(completed) if completed else 0,
            'best_speedup': max((j.final_speedup for j in completed), default=0),
        }

