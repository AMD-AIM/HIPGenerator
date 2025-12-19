#!/usr/bin/env python3
"""
Generate a detailed report from batch test results.
Usage: python generate_report.py [results_dir]
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime


def generate_report(results_dir: str):
    """Generate detailed report from results directory."""
    summary_file = os.path.join(results_dir, "summary.json")
    
    if not os.path.exists(summary_file):
        print(f"Summary file not found: {summary_file}")
        return
    
    with open(summary_file) as f:
        summary = json.load(f)
    
    # Collect detailed results
    detailed_results = []
    
    for task in summary.get('tasks', []):
        problem_name = task['problem']
        task_dir = os.path.join(results_dir, problem_name)
        
        result = {
            'problem': problem_name,
            'status': task['status'],
            'best_speedup': task.get('best_speedup', 0),
            'total_attempts': task.get('total_attempts', 0),
            'best_attempt': task.get('best_attempt', 0),
            'max_diff': None,
            'ref_time_ms': None,
            'new_time_ms': None,
            'prompt_file': None,
            'code_file': None,
            'error': None
        }
        
        # Load best result if exists
        best_result_file = os.path.join(task_dir, "best_result.json")
        if os.path.exists(best_result_file):
            with open(best_result_file) as f:
                best = json.load(f)
                result['max_diff'] = best.get('max_diff')
                result['ref_time_ms'] = best.get('ref_time_ms')
                result['new_time_ms'] = best.get('new_time_ms')
                result['error'] = best.get('error')
            result['prompt_file'] = os.path.join(task_dir, "best_prompt.txt")
            result['code_file'] = os.path.join(task_dir, "best_code.py")
        else:
            # Find latest result
            for i in range(task.get('total_attempts', 0), 0, -1):
                result_file = os.path.join(task_dir, f"result_{i}.json")
                if os.path.exists(result_file):
                    with open(result_file) as f:
                        latest = json.load(f)
                        result['error'] = latest.get('error')
                    break
        
        detailed_results.append(result)
    
    # Generate report
    report = []
    report.append("=" * 100)
    report.append("HipKittens Kernel Generation Batch Test Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 100)
    report.append("")
    
    # Summary statistics
    success = sum(1 for r in detailed_results if r['status'] == 'success')
    partial = sum(1 for r in detailed_results if r['status'] == 'partial')
    failed = sum(1 for r in detailed_results if r['status'] == 'failed')
    total = len(detailed_results)
    
    report.append("SUMMARY STATISTICS")
    report.append("-" * 50)
    report.append(f"Total tasks tested: {total}")
    report.append(f"✓ Success (speedup >= 1.0x): {success} ({100*success/total:.1f}%)" if total else "")
    report.append(f"⚠ Partial (accuracy pass, speedup < 1.0x): {partial} ({100*partial/total:.1f}%)" if total else "")
    report.append(f"✗ Failed: {failed} ({100*failed/total:.1f}%)" if total else "")
    report.append("")
    
    # Detailed table
    report.append("DETAILED RESULTS")
    report.append("-" * 100)
    header = f"{'Problem':<35} {'Status':<10} {'Speedup':<10} {'Max Diff':<12} {'Ref(ms)':<10} {'New(ms)':<10}"
    report.append(header)
    report.append("-" * 100)
    
    for r in sorted(detailed_results, key=lambda x: x['problem']):
        status_icon = {'success': '✓', 'partial': '⚠', 'failed': '✗'}.get(r['status'], '?')
        speedup = f"{r['best_speedup']:.2f}x" if r['best_speedup'] else "N/A"
        max_diff = f"{r['max_diff']:.6f}" if r['max_diff'] is not None else "N/A"
        ref_time = f"{r['ref_time_ms']:.3f}" if r['ref_time_ms'] else "N/A"
        new_time = f"{r['new_time_ms']:.3f}" if r['new_time_ms'] else "N/A"
        
        row = f"{r['problem']:<35} {status_icon} {r['status']:<8} {speedup:<10} {max_diff:<12} {ref_time:<10} {new_time:<10}"
        report.append(row)
    
    report.append("-" * 100)
    report.append("")
    
    # Failed tasks details
    failed_tasks = [r for r in detailed_results if r['status'] == 'failed']
    if failed_tasks:
        report.append("FAILED TASKS - ERROR DETAILS")
        report.append("-" * 100)
        for r in failed_tasks:
            report.append(f"\n{r['problem']}:")
            error = r.get('error', 'Unknown error')
            if error:
                # Truncate long errors
                error_lines = error.split('\n')[:5]
                for line in error_lines:
                    report.append(f"  {line[:100]}")
        report.append("")
    
    # Best performing tasks
    best_tasks = sorted([r for r in detailed_results if r['best_speedup'] > 0], 
                        key=lambda x: x['best_speedup'], reverse=True)[:10]
    if best_tasks:
        report.append("TOP 10 BEST PERFORMING TASKS")
        report.append("-" * 50)
        for i, r in enumerate(best_tasks, 1):
            report.append(f"{i}. {r['problem']}: {r['best_speedup']:.2f}x speedup")
        report.append("")
    
    # Save report
    report_text = "\n".join(report)
    report_file = os.path.join(results_dir, "detailed_report.txt")
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nReport saved to: {report_file}")
    
    # Also save as JSON
    json_report = {
        'generated': datetime.now().isoformat(),
        'summary': {
            'total': total,
            'success': success,
            'partial': partial,
            'failed': failed
        },
        'tasks': detailed_results
    }
    
    json_file = os.path.join(results_dir, "detailed_report.json")
    with open(json_file, 'w') as f:
        json.dump(json_report, f, indent=2)
    
    print(f"JSON report saved to: {json_file}")


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)
    
    generate_report(results_dir)


if __name__ == "__main__":
    main()



