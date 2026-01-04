"""
High-level optimizer that coordinates generation, evaluation, and optimization.
"""
import sys
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from utils.config import get_config
from utils.state import StateManager, JobState, JobStatus, EvalResult as StateEvalResult
from core.generator import Generator
from core.evaluator import Evaluator, EvalResult
from core.profiler import Profiler
from core.classifier import ProblemClassifier

logger = get_logger(__name__)


class Optimizer:
    """
    Orchestrates the generate-evaluate-optimize loop.
    """
    
    def __init__(self, config=None, state_manager: Optional[StateManager] = None):
        self.config = config or get_config()
        self.state = state_manager or StateManager(self.config.output_dir)
        self.generator = Generator(self.config)
        self.evaluator = Evaluator()
        self.profiler = Profiler(self.config.profiler_timeout)
        self.classifier = ProblemClassifier()
    
    def run(
        self,
        problem_path: str,
        max_optimize_rounds: int = 2,
        target_speedup: float = 1.0,
    ) -> JobState:
        """
        Run the full generate-evaluate-optimize pipeline.
        
        Args:
            problem_path: Path to the problem file
            max_optimize_rounds: Maximum optimization rounds
            target_speedup: Target speedup (stop early if achieved)
            
        Returns:
            JobState with results
        """
        # Classify problem
        problem_type, _ = self.classifier.classify(problem_path)
        problem_name = Path(problem_path).stem
        
        # Create job
        job = self.state.create_job(problem_path, problem_type, self.config.backend)
        job.status = JobStatus.GENERATING
        self.state.update_job(job)
        
        logger.info("=" * 60)
        logger.info(f"PROBLEM: {problem_name}")
        logger.info(f"TYPE: {problem_type}")
        logger.info("=" * 60)
        
        # Setup output directory
        output_dir = Path(self.config.output_dir) / problem_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Stage 1: Generate
            logger.info("\n[STAGE 1] GENERATE (torch2triton)")
            gen_result = self._run_generate(job, problem_path, output_dir)
            
            if not gen_result or gen_result.speedup <= 0:
                job.status = JobStatus.FAILED
                job.errors.append("Generation failed - no valid code produced")
                self.state.update_job(job)
                return job
            
            job.best_generate_speedup = gen_result.speedup
            logger.info(f"Generate result: {gen_result.speedup:.2f}x speedup")
            
            # Check if target achieved
            if gen_result.speedup >= target_speedup:
                logger.info(f"✓ Target speedup achieved in generation phase!")
                job.final_speedup = gen_result.speedup
                job.final_code = job.best_generate_code
                job.status = JobStatus.COMPLETED
                self.state.update_job(job)
                return job
            
            # Stage 2: Optimize
            logger.info(f"\n[STAGE 2] OPTIMIZE (triton2triton) - up to {max_optimize_rounds} rounds")
            job.status = JobStatus.OPTIMIZING
            self.state.update_job(job)
            
            current_speedup = gen_result.speedup
            current_code = job.best_generate_code
            
            for round_num in range(1, max_optimize_rounds + 1):
                logger.info(f"\n--- Optimize Round {round_num}/{max_optimize_rounds} ---")
                
                opt_result = self._run_optimize(
                    job, problem_path, current_code, current_speedup, output_dir, round_num
                )
                
                if opt_result and opt_result.accuracy_pass:
                    if opt_result.speedup > current_speedup:
                        improvement = (opt_result.speedup - current_speedup) / current_speedup * 100
                        logger.info(f"Speedup improved: {current_speedup:.2f}x → {opt_result.speedup:.2f}x (+{improvement:.1f}%)")
                        current_speedup = opt_result.speedup
                        current_code = str(output_dir / f"optimize_{round_num}.py")
                    else:
                        logger.info(f"No improvement: {opt_result.speedup:.2f}x <= {current_speedup:.2f}x")
                    
                    if current_speedup >= target_speedup:
                        logger.info(f"✓ Target speedup achieved!")
                        break
                else:
                    logger.warning(f"Optimize round {round_num} failed")
                    if opt_result:
                        job.errors.append(f"Optimize round {round_num}: {opt_result.error}")
            
            # Set final results
            job.best_optimize_speedup = current_speedup
            job.best_optimize_code = current_code
            job.final_speedup = current_speedup
            job.final_code = current_code
            job.status = JobStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            job.status = JobStatus.FAILED
            job.errors.append(str(e))
        
        self.state.update_job(job)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Problem: {problem_name}")
        logger.info(f"Generate Speedup: {job.best_generate_speedup:.2f}x")
        logger.info(f"Final Speedup: {job.final_speedup:.2f}x")
        logger.info(f"Status: {job.status.value}")
        if job.errors:
            logger.info(f"Errors: {len(job.errors)}")
        logger.info("=" * 60)
        
        return job
    
    def _run_generate(self, job: JobState, problem_path: str, output_dir: Path) -> Optional[EvalResult]:
        """Run generation phase."""
        job.generate_attempts += 1
        
        # Generate code
        gen_paths = self.generator.generate(
            problem_path=problem_path,
            output_path=str(output_dir / "generate_1.py"),
            num_samples=self.config.num_samples,
            temperature=self.config.temperature,
        )
        
        if not gen_paths:
            return None
        
        # Evaluate each sample
        best_result = None
        best_path = None
        
        for gen_path in gen_paths:
            logger.info(f"Evaluating: {Path(gen_path).name}")
            result = self.evaluator.evaluate(gen_path, problem_path)
            logger.info(f"  {result}")
            
            # Track in job state
            state_result = StateEvalResult(
                code_path=gen_path,
                compile_success=result.compile_success,
                accuracy_pass=result.accuracy_pass,
                speedup=result.speedup,
                ref_time_ms=result.ref_time_ms,
                new_time_ms=result.new_time_ms,
                max_diff=result.max_diff,
                error=result.error,
            )
            job.generate_results.append(state_result)
            
            # Track best
            if result.accuracy_pass and (best_result is None or result.speedup > best_result.speedup):
                best_result = result
                best_path = gen_path
        
        if best_path:
            job.best_generate_code = best_path
            # Save as best code
            import shutil
            shutil.copy(best_path, output_dir / "best_code.py")
        
        self.state.update_job(job)
        return best_result
    
    def _run_optimize(
        self,
        job: JobState,
        problem_path: str,
        prev_code: str,
        current_speedup: float,
        output_dir: Path,
        round_num: int,
    ) -> Optional[EvalResult]:
        """Run one optimization round."""
        job.optimize_rounds += 1
        
        # Profile current code for feedback
        profiler_feedback = ""
        if self.config.enable_profiler:
            job.status = JobStatus.PROFILING
            self.state.update_job(job)
            
            # Create a temporary script to profile
            profile_script = self._create_profile_script(prev_code, problem_path, output_dir)
            if profile_script:
                metrics = self.profiler.profile(profile_script)
                if metrics:
                    profiler_feedback = self.profiler.build_feedback(metrics, current_speedup)
                    job.profiler_feedback_history.append(profiler_feedback)
                    logger.info(f"Profiler: VGPR={metrics.vgpr_count}, LDS={metrics.lds_bytes/1024:.1f}KB, L2={metrics.l2_hit_rate:.1f}%")
        
        job.status = JobStatus.OPTIMIZING
        self.state.update_job(job)
        
        # Generate optimized code
        opt_path = str(output_dir / f"optimize_{round_num}.py")
        self.generator.optimize(
            problem_path=problem_path,
            prev_code_path=prev_code,
            output_path=opt_path,
            speedup=current_speedup,
            profiler_feedback=profiler_feedback,
        )
        
        if not Path(opt_path).exists():
            return None
        
        # Evaluate
        result = self.evaluator.evaluate(opt_path, problem_path)
        logger.info(f"Optimize result: {result}")
        
        # Track in job state
        state_result = StateEvalResult(
            code_path=opt_path,
            compile_success=result.compile_success,
            accuracy_pass=result.accuracy_pass,
            speedup=result.speedup,
            ref_time_ms=result.ref_time_ms,
            new_time_ms=result.new_time_ms,
            max_diff=result.max_diff,
            error=result.error,
        )
        job.optimize_results.append(state_result)
        self.state.update_job(job)
        
        return result
    
    def _create_profile_script(self, code_path: str, problem_path: str, output_dir: Path) -> Optional[str]:
        """Create a script for profiling."""
        script_path = output_dir / "profile_script.py"
        
        script = f'''
import torch
import sys
sys.path.insert(0, ".")

# Load problem
import importlib.util
spec = importlib.util.spec_from_file_location("problem", "{problem_path}")
problem = importlib.util.module_from_spec(spec)
spec.loader.exec_module(problem)

# Load generated code
spec = importlib.util.spec_from_file_location("generated", "{code_path}")
generated = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generated)

# Run
model = generated.ModelNew().cuda().eval()
inputs = problem.get_inputs()
if not isinstance(inputs, (list, tuple)):
    inputs = [inputs]

# Warmup
for _ in range(3):
    with torch.no_grad():
        _ = model(*inputs)
torch.cuda.synchronize()

# Profile run
with torch.no_grad():
    _ = model(*inputs)
torch.cuda.synchronize()
'''
        
        with open(script_path, 'w') as f:
            f.write(script)
        
        return str(script_path)

