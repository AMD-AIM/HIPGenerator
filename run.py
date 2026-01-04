#!/usr/bin/env python3
"""
HIPGenerator - Unified Entry Point

Usage:
    # Full pipeline (generate + optimize)
    python run.py --problem datasets/25_Swish.py
    
    # Generate only
    python run.py --problem datasets/25_Swish.py --mode generate
    
    # Optimize existing code
    python run.py --problem datasets/25_Swish.py --mode optimize --prev-code results/25_Swish/generate_1.py
    
    # Evaluate code
    python run.py --problem datasets/25_Swish.py --mode evaluate --code results/25_Swish/best_code.py
    
    # Profile code
    python run.py --problem datasets/25_Swish.py --mode profile --code results/25_Swish/best_code.py
    
    # Batch test all datasets
    python run.py --batch --datasets-dir datasets/
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logging, get_logger
from utils.config import Config, set_config
from utils.state import StateManager, JobStatus
from core.generator import Generator
from core.evaluator import Evaluator
from core.profiler import Profiler
from core.optimizer import Optimizer
from core.classifier import ProblemClassifier


def cmd_run(args):
    """Run full generate-optimize pipeline."""
    logger = get_logger()
    
    config = Config.from_env()
    config.output_dir = args.output
    config.num_samples = args.samples
    set_config(config)
    
    errors = config.validate()
    if errors:
        logger.error(f"Configuration errors: {errors}")
        return 1
    
    optimizer = Optimizer(config)
    job = optimizer.run(
        problem_path=args.problem,
        max_optimize_rounds=args.optimize_rounds,
        target_speedup=args.target_speedup,
    )
    
    # Output result
    if args.json:
        print(json.dumps(job.to_dict(), indent=2))
    
    return 0 if job.status == JobStatus.COMPLETED else 1


def cmd_generate(args):
    """Generate Triton kernel."""
    logger = get_logger()
    
    config = Config.from_env()
    set_config(config)
    
    generator = Generator(config)
    paths = generator.generate(
        problem_path=args.problem,
        output_path=args.output,
        num_samples=args.samples,
        temperature=args.temperature,
    )
    
    if paths:
        logger.info(f"Generated {len(paths)} sample(s)")
        for p in paths:
            print(p)
        return 0
    return 1


def cmd_optimize(args):
    """Optimize existing Triton kernel."""
    logger = get_logger()
    
    config = Config.from_env()
    set_config(config)
    
    # Get profiler feedback if requested
    profiler_feedback = ""
    if args.profile:
        profiler = Profiler()
        # Create temp profile script
        problem_path = args.problem
        prev_code = args.prev_code
        
        script = f'''
import torch
import sys
import importlib.util

spec = importlib.util.spec_from_file_location("problem", "{problem_path}")
problem = importlib.util.module_from_spec(spec)
spec.loader.exec_module(problem)

spec = importlib.util.spec_from_file_location("generated", "{prev_code}")
generated = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generated)

model = generated.ModelNew().cuda().eval()
inputs = problem.get_inputs()
if not isinstance(inputs, (list, tuple)):
    inputs = [inputs]

for _ in range(3):
    with torch.no_grad():
        _ = model(*inputs)
torch.cuda.synchronize()

with torch.no_grad():
    _ = model(*inputs)
torch.cuda.synchronize()
'''
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name
        
        metrics = profiler.profile(script_path)
        if metrics:
            profiler_feedback = profiler.build_feedback(metrics, args.speedup)
            logger.info(f"Profiler: VGPR={metrics.vgpr_count}, LDS={metrics.lds_bytes/1024:.1f}KB")
    
    generator = Generator(config)
    path = generator.optimize(
        problem_path=args.problem,
        prev_code_path=args.prev_code,
        output_path=args.output,
        speedup=args.speedup,
        profiler_feedback=profiler_feedback,
    )
    
    if path:
        print(path)
        return 0
    return 1


def cmd_evaluate(args):
    """Evaluate generated kernel."""
    logger = get_logger()
    
    evaluator = Evaluator()
    result = evaluator.evaluate(args.code, args.problem)
    
    print(f"Compile: {'✓' if result.compile_success else '✗'}")
    print(f"Accuracy: {'✓' if result.accuracy_pass else '✗'} (max_diff={result.max_diff:.6f})")
    print(f"Reference: {result.ref_time_ms:.3f} ms")
    print(f"Generated: {result.new_time_ms:.3f} ms")
    print(f"Speedup: {result.speedup:.2f}x")
    
    if result.error:
        print(f"Error: {result.error}")
    
    if args.json:
        print(json.dumps({
            'compile_success': result.compile_success,
            'accuracy_pass': result.accuracy_pass,
            'speedup': result.speedup,
            'ref_time_ms': result.ref_time_ms,
            'new_time_ms': result.new_time_ms,
            'max_diff': result.max_diff,
            'error': result.error,
        }, indent=2))
    
    return 0 if result.accuracy_pass else 1


def cmd_profile(args):
    """Profile generated kernel."""
    logger = get_logger()
    
    # Create profile script
    script = f'''
import torch
import sys
import importlib.util

spec = importlib.util.spec_from_file_location("problem", "{args.problem}")
problem = importlib.util.module_from_spec(spec)
spec.loader.exec_module(problem)

spec = importlib.util.spec_from_file_location("generated", "{args.code}")
generated = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generated)

model = generated.ModelNew().cuda().eval()
inputs = problem.get_inputs()
if not isinstance(inputs, (list, tuple)):
    inputs = [inputs]

for _ in range(3):
    with torch.no_grad():
        _ = model(*inputs)
torch.cuda.synchronize()

with torch.no_grad():
    _ = model(*inputs)
torch.cuda.synchronize()
'''
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name
    
    profiler = Profiler()
    metrics = profiler.profile(script_path)
    
    if metrics:
        print(f"Kernel: {metrics.kernel_name}")
        print(f"Time: {metrics.avg_time_us:.1f} μs")
        print(f"VGPR: {metrics.vgpr_count}")
        print(f"SGPR: {metrics.sgpr_count}")
        print(f"LDS: {metrics.lds_bytes / 1024:.1f} KB")
        print(f"Workgroup: {metrics.workgroup_size}")
        print(f"Grid: {metrics.grid_size}")
        print(f"L2 Hit Rate: {metrics.l2_hit_rate:.1f}%")
        print(f"LDS Conflicts: {metrics.lds_bank_conflicts:.0f}")
        print(f"VALU Insts: {metrics.valu_insts}")
        print(f"SALU Insts: {metrics.salu_insts}")
        
        if args.json:
            print(json.dumps(metrics.to_dict(), indent=2))
        return 0
    
    logger.error("Profiling failed")
    return 1


def cmd_classify(args):
    """Classify problem type."""
    classifier = ProblemClassifier()
    ptype, prompt = classifier.classify(args.problem)
    
    print(f"Problem: {Path(args.problem).stem}")
    print(f"Type: {ptype}")
    print(f"Prompt: {prompt}")
    print(f"Description: {classifier.get_type_description(ptype)}")
    return 0


def cmd_batch(args):
    """Run batch test on all datasets."""
    logger = get_logger()
    
    datasets_dir = Path(args.datasets_dir)
    problems = sorted(datasets_dir.glob("*.py"))
    
    logger.info(f"Found {len(problems)} problems in {datasets_dir}")
    
    config = Config.from_env()
    config.output_dir = args.output
    config.num_samples = args.samples
    set_config(config)
    
    results = []
    
    for problem_path in problems:
        if problem_path.name.startswith("_"):
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {problem_path.name}")
        logger.info(f"{'='*60}")
        
        optimizer = Optimizer(config)
        job = optimizer.run(
            problem_path=str(problem_path),
            max_optimize_rounds=args.optimize_rounds,
            target_speedup=args.target_speedup,
        )
        
        results.append({
            'problem': problem_path.stem,
            'type': job.problem_type,
            'status': job.status.value,
            'generate_speedup': job.best_generate_speedup,
            'final_speedup': job.final_speedup,
            'errors': job.errors,
        })
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("BATCH RESULTS")
    logger.info("=" * 60)
    
    passed = sum(1 for r in results if r['status'] == 'completed' and r['final_speedup'] >= 1.0)
    total = len(results)
    
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Average Speedup: {sum(r['final_speedup'] for r in results) / total:.2f}x")
    
    for r in sorted(results, key=lambda x: x['final_speedup'], reverse=True):
        status = "✓" if r['final_speedup'] >= 1.0 else "✗"
        logger.info(f"  {status} {r['problem']}: {r['final_speedup']:.2f}x ({r['type']})")
    
    # Save results
    results_file = Path(args.output) / "batch_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")
    
    return 0 if passed == total else 1


def cmd_status(args):
    """Show job status."""
    state = StateManager(args.output)
    
    summary = state.get_summary()
    print(f"Total Jobs: {summary['total_jobs']}")
    print(f"Completed: {summary['completed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Average Speedup: {summary['avg_speedup']:.2f}x")
    print(f"Best Speedup: {summary['best_speedup']:.2f}x")
    
    if args.list:
        print("\nRecent Jobs:")
        for job in state.list_jobs()[:10]:
            print(f"  {job.problem_name}: {job.final_speedup:.2f}x ({job.status.value})")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="HIPGenerator - Generate and optimize Triton kernels using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--output", "-o", default="results", help="Output directory")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run command (default)
    run_parser = subparsers.add_parser("run", help="Run full generate-optimize pipeline")
    run_parser.add_argument("--problem", "-p", required=True, help="Problem file path")
    run_parser.add_argument("--samples", "-n", type=int, default=3, help="Number of samples to generate")
    run_parser.add_argument("--optimize-rounds", type=int, default=2, help="Max optimization rounds")
    run_parser.add_argument("--target-speedup", type=float, default=1.0, help="Target speedup")
    run_parser.add_argument("--output", "-o", default="results", help="Output directory")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate Triton kernel")
    gen_parser.add_argument("--problem", "-p", required=True, help="Problem file path")
    gen_parser.add_argument("--output", "-o", default="results/generated.py", help="Output file")
    gen_parser.add_argument("--samples", "-n", type=int, default=1, help="Number of samples")
    gen_parser.add_argument("--temperature", "-t", type=float, default=0.1, help="LLM temperature")
    
    # Optimize command
    opt_parser = subparsers.add_parser("optimize", help="Optimize existing kernel")
    opt_parser.add_argument("--problem", "-p", required=True, help="Problem file path")
    opt_parser.add_argument("--prev-code", required=True, help="Previous code to optimize")
    opt_parser.add_argument("--output", "-o", default="results/optimized.py", help="Output file")
    opt_parser.add_argument("--speedup", type=float, default=0.5, help="Current speedup")
    opt_parser.add_argument("--profile", action="store_true", help="Run profiler for feedback")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate generated kernel")
    eval_parser.add_argument("--problem", "-p", required=True, help="Problem file path")
    eval_parser.add_argument("--code", "-c", required=True, help="Generated code path")
    
    # Profile command
    prof_parser = subparsers.add_parser("profile", help="Profile kernel")
    prof_parser.add_argument("--problem", "-p", required=True, help="Problem file path")
    prof_parser.add_argument("--code", "-c", required=True, help="Generated code path")
    
    # Classify command
    class_parser = subparsers.add_parser("classify", help="Classify problem type")
    class_parser.add_argument("--problem", "-p", required=True, help="Problem file path")
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch test all datasets")
    batch_parser.add_argument("--datasets-dir", "-d", default="datasets", help="Datasets directory")
    batch_parser.add_argument("--samples", "-n", type=int, default=3, help="Samples per problem")
    batch_parser.add_argument("--optimize-rounds", type=int, default=2, help="Optimize rounds")
    batch_parser.add_argument("--target-speedup", type=float, default=1.0, help="Target speedup")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show job status")
    status_parser.add_argument("--list", "-l", action="store_true", help="List recent jobs")
    status_parser.add_argument("--output", "-o", default="results", help="Output directory to check")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Handle commands
    if args.command == "run" or (args.command is None and hasattr(args, 'problem')):
        return cmd_run(args)
    elif args.command == "generate":
        return cmd_generate(args)
    elif args.command == "optimize":
        return cmd_optimize(args)
    elif args.command == "evaluate":
        return cmd_evaluate(args)
    elif args.command == "profile":
        return cmd_profile(args)
    elif args.command == "classify":
        return cmd_classify(args)
    elif args.command == "batch":
        return cmd_batch(args)
    elif args.command == "status":
        return cmd_status(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

