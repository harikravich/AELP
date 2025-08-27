#!/usr/bin/env python3
"""
Component Logger for GAELP
Tracks every decision made by every component with full traceability
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading
import queue

@dataclass
class ComponentDecision:
    """Record of a single component decision"""
    timestamp: datetime
    component_name: str
    decision_type: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    impact_metrics: Dict[str, Any]
    processing_time_ms: float
    trace_id: str  # Links related decisions across components

class ComponentLogger:
    """Centralized logger for all GAELP components"""
    
    def __init__(self, log_file: str = "gaelp_decisions.jsonl"):
        self.log_file = log_file
        self.decisions = []
        self.component_metrics = defaultdict(lambda: {
            'calls': 0,
            'total_time_ms': 0,
            'errors': 0,
            'impact_score': 0
        })
        self.trace_map = {}  # Maps trace_id to all related decisions
        self.queue = queue.Queue()
        self.running = True
        self._start_writer_thread()
    
    def _start_writer_thread(self):
        """Background thread to write logs without blocking"""
        def writer():
            while self.running:
                try:
                    decision = self.queue.get(timeout=1)
                    if decision:
                        with open(self.log_file, 'a') as f:
                            f.write(json.dumps(asdict(decision), default=str) + '\n')
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Logger error: {e}")
        
        self.writer_thread = threading.Thread(target=writer, daemon=True)
        self.writer_thread.start()
    
    def log_decision(self, 
                     component_name: str,
                     decision_type: str,
                     input_data: Dict[str, Any],
                     output_data: Dict[str, Any],
                     processing_time_ms: float,
                     trace_id: str,
                     impact_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Log a component decision with full context"""
        
        decision = ComponentDecision(
            timestamp=datetime.now(),
            component_name=component_name,
            decision_type=decision_type,
            input_data=input_data,
            output_data=output_data,
            impact_metrics=impact_metrics or {},
            processing_time_ms=processing_time_ms,
            trace_id=trace_id
        )
        
        # Store in memory for analysis
        self.decisions.append(decision)
        
        # Update component metrics
        self.component_metrics[component_name]['calls'] += 1
        self.component_metrics[component_name]['total_time_ms'] += processing_time_ms
        
        # Track trace
        if trace_id not in self.trace_map:
            self.trace_map[trace_id] = []
        self.trace_map[trace_id].append(decision)
        
        # Queue for background writing
        self.queue.put(decision)
        
        # Print summary for visibility
        print(f"[{component_name}] {decision_type}: {output_data.get('summary', output_data)}")
    
    def trace_conversion_path(self, conversion_id: str) -> List[ComponentDecision]:
        """Trace all decisions that led to a conversion"""
        if conversion_id in self.trace_map:
            path = self.trace_map[conversion_id]
            print(f"\n=== CONVERSION PATH for {conversion_id} ===")
            for decision in path:
                print(f"  {decision.timestamp.strftime('%H:%M:%S')} | {decision.component_name}: {decision.decision_type}")
                print(f"    Input: {decision.input_data}")
                print(f"    Output: {decision.output_data}")
                if decision.impact_metrics:
                    print(f"    Impact: {decision.impact_metrics}")
            return path
        return []
    
    def get_component_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all components"""
        summary = {}
        for component, metrics in self.component_metrics.items():
            summary[component] = {
                'total_calls': metrics['calls'],
                'avg_time_ms': metrics['total_time_ms'] / metrics['calls'] if metrics['calls'] > 0 else 0,
                'error_rate': metrics['errors'] / metrics['calls'] if metrics['calls'] > 0 else 0,
                'last_decision': None
            }
            
            # Add last decision
            for decision in reversed(self.decisions):
                if decision.component_name == component:
                    summary[component]['last_decision'] = {
                        'type': decision.decision_type,
                        'time': decision.timestamp.strftime('%H:%M:%S'),
                        'output': decision.output_data
                    }
                    break
        
        return summary
    
    def verify_component_active(self, component_name: str, min_calls: int = 1) -> bool:
        """Verify a component is actually being used"""
        metrics = self.component_metrics.get(component_name, {})
        is_active = metrics.get('calls', 0) >= min_calls
        
        if is_active:
            print(f"✅ {component_name}: ACTIVE ({metrics['calls']} calls)")
        else:
            print(f"❌ {component_name}: INACTIVE (0 calls)")
        
        return is_active
    
    def shutdown(self):
        """Clean shutdown of logger"""
        self.running = False
        self.writer_thread.join(timeout=5)
        print(f"Logger shutdown. Logged {len(self.decisions)} decisions.")

# Global logger instance
LOGGER = ComponentLogger()

def log_component_decision(component_name: str):
    """Decorator to automatically log component decisions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            trace_id = kwargs.get('trace_id', f"trace_{int(time.time()*1000)}")
            
            # Capture input
            input_data = {
                'args': str(args)[:200],  # Truncate for readability
                'kwargs': str(kwargs)[:200]
            }
            
            try:
                # Execute component function
                result = func(*args, **kwargs)
                
                # Calculate processing time
                processing_time_ms = (time.time() - start_time) * 1000
                
                # Log the decision
                LOGGER.log_decision(
                    component_name=component_name,
                    decision_type=func.__name__,
                    input_data=input_data,
                    output_data={'result': str(result)[:500]},
                    processing_time_ms=processing_time_ms,
                    trace_id=trace_id
                )
                
                return result
                
            except Exception as e:
                LOGGER.component_metrics[component_name]['errors'] += 1
                raise e
        
        return wrapper
    return decorator