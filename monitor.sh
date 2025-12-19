#!/bin/bash
# Monitor batch test progress
# Usage: ./monitor.sh

RESULTS_DIR="${1:-results}"

echo "=== Batch Test Monitor ==="
echo ""

# Check if test is running
if pgrep -f "run_batch.sh" > /dev/null; then
    echo "Status: RUNNING"
else
    echo "Status: NOT RUNNING"
fi

echo ""
echo "=== Current Progress ==="

# Count completed tasks
if [[ -f "${RESULTS_DIR}/summary.json" ]]; then
    python3 << EOF
import json
import os

summary = json.load(open('${RESULTS_DIR}/summary.json'))
tasks = summary.get('tasks', [])

success = sum(1 for t in tasks if t['status'] == 'success')
partial = sum(1 for t in tasks if t['status'] == 'partial')
failed = sum(1 for t in tasks if t['status'] == 'failed')
total = len(tasks)

print(f"Completed: {total} tasks")
print(f"  ✓ Success: {success}")
print(f"  ⚠ Partial: {partial}")
print(f"  ✗ Failed: {failed}")
print()

# Show recent results
if tasks:
    print("Recent results:")
    for task in tasks[-5:]:
        icon = {'success': '✓', 'partial': '⚠', 'failed': '✗'}.get(task['status'], '?')
        print(f"  {icon} {task['problem']}: {task['best_speedup']:.2f}x")
EOF
else
    echo "No results yet"
fi

echo ""
echo "=== Recent Log ==="
tail -10 "${RESULTS_DIR}/batch_run.log" 2>/dev/null | head -10



