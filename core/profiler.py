"""
Profiler for analyzing kernel performance using rocprofv3.
"""
import os
import re
import sys
import subprocess
import tempfile
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProfilerMetrics:
    """Collected profiler metrics."""
    kernel_name: str = ""
    avg_time_us: float = 0.0
    vgpr_count: int = 0
    sgpr_count: int = 0
    lds_bytes: int = 0
    workgroup_size: int = 0
    grid_size: int = 0
    l2_hit_rate: float = 0.0
    lds_bank_conflicts: float = 0.0
    valu_insts: int = 0
    salu_insts: int = 0
    waves: int = 0
    
    def to_dict(self) -> dict:
        return {
            'kernel_name': self.kernel_name,
            'avg_time_us': self.avg_time_us,
            'vgpr_count': self.vgpr_count,
            'sgpr_count': self.sgpr_count,
            'lds_bytes': self.lds_bytes,
            'workgroup_size': self.workgroup_size,
            'grid_size': self.grid_size,
            'l2_hit_rate': self.l2_hit_rate,
            'lds_bank_conflicts': self.lds_bank_conflicts,
            'valu_insts': self.valu_insts,
            'salu_insts': self.salu_insts,
            'waves': self.waves,
        }


class Profiler:
    """
    Profiles Triton kernels using rocprofv3.
    """
    
    # PMC counters to collect
    PMC_COUNTERS = [
        "SQ_INSTS_SALU",
        "SQ_INSTS_VALU", 
        "SQ_LDS_BANK_CONFLICT",
        "SQ_WAVES",
        "TCC_HIT",
        "TCC_MISS",
    ]
    
    def __init__(self, timeout: int = 300):
        self.timeout = timeout
    
    def profile(self, script_path: str) -> Optional[ProfilerMetrics]:
        """
        Profile a Python script containing a Triton kernel.
        
        Args:
            script_path: Path to the script to profile
            
        Returns:
            ProfilerMetrics if successful, None otherwise
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "rocprof_output"
            output_dir.mkdir()
            
            # Build rocprofv3 command
            pmc_list = ",".join(self.PMC_COUNTERS)
            cmd = [
                "rocprofv3",
                "-d", str(output_dir),
                "--pmc", pmc_list,
                "python3", script_path
            ]
            
            logger.debug(f"Running: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env={**os.environ, "ROCPROF_TRUNCATE_KERNEL_PATH": "1"}
                )
            except subprocess.TimeoutExpired:
                logger.warning(f"Profiler timed out after {self.timeout}s")
                return None
            except Exception as e:
                logger.error(f"Profiler error: {e}")
                return None
            
            # Find output database
            db_files = list(output_dir.glob("*.db"))
            if not db_files:
                logger.warning("No rocprof database generated")
                return None
            
            return self._parse_database(db_files[0])
    
    def _parse_database(self, db_path: Path) -> ProfilerMetrics:
        """Parse rocprofv3 SQLite database."""
        metrics = ProfilerMetrics()
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Get table list
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Parse kernel dispatch info
            if 'rocpd_kernel_dispatch' in tables:
                cursor.execute("""
                    SELECT kernel_name, vgpr_count, sgpr_count, 
                           group_segment_size, workgroup_size, grid_size
                    FROM rocpd_kernel_dispatch 
                    WHERE kernel_name NOT LIKE '%__triton_launcher%'
                    LIMIT 1
                """)
                row = cursor.fetchone()
                if row:
                    metrics.kernel_name = row[0] or ""
                    metrics.vgpr_count = row[1] or 0
                    metrics.sgpr_count = row[2] or 0
                    metrics.lds_bytes = row[3] or 0
                    metrics.workgroup_size = row[4] or 0
                    metrics.grid_size = row[5] or 0
            
            # Parse timing info
            if 'rocpd_op' in tables:
                cursor.execute("""
                    SELECT AVG(end_ns - start_ns) / 1000.0 
                    FROM rocpd_op 
                    WHERE args LIKE '%kernel%'
                """)
                row = cursor.fetchone()
                if row and row[0]:
                    metrics.avg_time_us = row[0]
            
            # Parse PMC counters
            if 'rocpd_pmc_event' in tables:
                cursor.execute("SELECT event_name, value FROM rocpd_pmc_event")
                pmc_values = {row[0]: row[1] for row in cursor.fetchall()}
                
                # L2 hit rate
                tcc_hit = pmc_values.get('TCC_HIT', 0)
                tcc_miss = pmc_values.get('TCC_MISS', 0)
                if tcc_hit + tcc_miss > 0:
                    metrics.l2_hit_rate = 100.0 * tcc_hit / (tcc_hit + tcc_miss)
                
                # Other metrics
                metrics.lds_bank_conflicts = pmc_values.get('SQ_LDS_BANK_CONFLICT', 0)
                metrics.valu_insts = pmc_values.get('SQ_INSTS_VALU', 0)
                metrics.salu_insts = pmc_values.get('SQ_INSTS_SALU', 0)
                metrics.waves = pmc_values.get('SQ_WAVES', 0)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to parse profiler database: {e}")
        
        return metrics
    
    def build_feedback(self, metrics: ProfilerMetrics, current_speedup: float) -> str:
        """
        Build optimization feedback from profiler metrics.
        
        Args:
            metrics: Profiler metrics
            current_speedup: Current speedup value
            
        Returns:
            Feedback string for LLM
        """
        lines = [
            f"PERFORMANCE ANALYSIS (Current Speedup: {current_speedup:.2f}x)",
            "=" * 60,
            "",
            f"Kernel: {metrics.kernel_name}",
            f"Average Time: {metrics.avg_time_us:.1f} μs",
            "",
            "RESOURCE USAGE:",
            f"  - VGPRs: {metrics.vgpr_count}",
            f"  - SGPRs: {metrics.sgpr_count}",
            f"  - LDS: {metrics.lds_bytes / 1024:.1f} KB",
            f"  - Workgroup Size: {metrics.workgroup_size}",
            f"  - Grid Size: {metrics.grid_size}",
            "",
            "PERFORMANCE METRICS:",
            f"  - L2 Cache Hit Rate: {metrics.l2_hit_rate:.1f}%",
            f"  - LDS Bank Conflicts: {metrics.lds_bank_conflicts:.0f}/wave",
            f"  - VALU Instructions: {metrics.valu_insts}",
            f"  - SALU Instructions: {metrics.salu_insts}",
            "",
            "OPTIMIZATION RECOMMENDATIONS:",
        ]
        
        # Generate specific recommendations
        if metrics.vgpr_count > 128:
            lines.append(f"  ⚠ HIGH VGPR ({metrics.vgpr_count}): Limits occupancy. Reduce register pressure.")
        
        if metrics.lds_bytes > 65536:  # 64KB
            lines.append(f"  ⚠ HIGH LDS ({metrics.lds_bytes/1024:.0f}KB): Exceeds 64KB/CU. Reduce tile sizes.")
        
        if metrics.l2_hit_rate < 50:
            lines.append(f"  ⚠ LOW L2 HIT RATE ({metrics.l2_hit_rate:.1f}%): Add tile grouping for cache reuse.")
        
        if metrics.lds_bank_conflicts > 1000:
            lines.append(f"  ⚠ LDS CONFLICTS ({metrics.lds_bank_conflicts:.0f}): Optimize memory access patterns.")
        
        if metrics.workgroup_size < 128:
            lines.append(f"  ⚠ SMALL WORKGROUP ({metrics.workgroup_size}): Increase for better occupancy.")
        
        if current_speedup < 1.0:
            lines.append(f"  ⚠ SLOWER THAN REFERENCE: Focus on algorithmic improvements.")
        
        return "\n".join(lines)

