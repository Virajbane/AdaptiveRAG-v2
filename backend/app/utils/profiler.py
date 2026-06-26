import time
from contextlib import asynccontextmanager
from typing import Dict
import json

class PerformanceProfiler:
    """Track timing for each step"""
    
    def __init__(self):
        self.timings: Dict[str, list] = {}
        self.current_traces: Dict[str, float] = {}
    
    def start(self, name: str):
        """Start timing a step"""
        self.current_traces[name] = time.time()
    
    def end(self, name: str) -> float:
        """End timing and return elapsed ms"""
        if name not in self.current_traces:
            return 0.0
        
        elapsed_ms = (time.time() - self.current_traces[name]) * 1000
        
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(elapsed_ms)
        
        del self.current_traces[name]
        return elapsed_ms
    
    def get_stats(self) -> Dict:
        """Get timing statistics"""
        stats = {}
        for name, times in self.timings.items():
            stats[name] = {
                "count": len(times),
                "avg_ms": sum(times) / len(times),
                "min_ms": min(times),
                "max_ms": max(times),
                "total_ms": sum(times)
            }
        return stats
    
    def print_summary(self):
        """Print performance summary"""
        stats = self.get_stats()
        total = sum(s["total_ms"] for s in stats.values())
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        for name, stat in sorted(stats.items(), key=lambda x: x[1]["total_ms"], reverse=True):
            percent = (stat["total_ms"] / total * 100) if total > 0 else 0
            print(f"{name:30} {stat['avg_ms']:8.0f}ms ({percent:5.1f}%)")
        
        print("-"*60)
        print(f"{'TOTAL':30} {total:8.0f}ms")
        print("="*60 + "\n")
    
    def reset(self):
        """Reset all timings"""
        self.timings.clear()
        self.current_traces.clear()

# Global instance
profiler = PerformanceProfiler()