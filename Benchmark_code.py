#!/usr/bin/env python3
"""
Ollama Performance Monitoring Tool
Measures inference time, CPU usage, and power consumption
"""

import ollama
import time
import psutil
import json
import csv
from datetime import datetime
from pathlib import Path

class OllamaMonitor:
    def __init__(self, model="phi3:mini", output_dir="./results"):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
        # Try to import power monitoring (Pi-specific)
        self.power_available = self._check_power_monitoring()
        
    def _check_power_monitoring(self):
        """Check if power monitoring is available"""
        try:
            # Check if vcgencmd is available (Raspberry Pi)
            import subprocess
            subprocess.run(['vcgencmd', 'measure_volts'], 
                         capture_output=True, check=True)
            return True
        except:
            print("⚠️  Power monitoring not available (vcgencmd not found)")
            return False
    
    def _get_power_metrics(self):
        """Get power-related metrics from Raspberry Pi"""
        if not self.power_available:
            return {}
        
        try:
            import subprocess
            
            # Get voltage
            voltage_cmd = subprocess.run(
                ['vcgencmd', 'measure_volts'],
                capture_output=True, text=True
            )
            voltage_str = voltage_cmd.stdout.strip()
            voltage = float(voltage_str.split('=')[1].replace('V', ''))
            
            # Get CPU temperature
            temp_cmd = subprocess.run(
                ['vcgencmd', 'measure_temp'],
                capture_output=True, text=True
            )
            temp_str = temp_cmd.stdout.strip()
            temperature = float(temp_str.split('=')[1].replace("'C", ''))
            
            # Get throttling status
            throttle_cmd = subprocess.run(
                ['vcgencmd', 'get_throttled'],
                capture_output=True, text=True
            )
            throttled = throttle_cmd.stdout.strip()
            
            return {
                'voltage_v': voltage,
                'temperature_c': temperature,
                'throttled': throttled
            }
        except Exception as e:
            print(f"Error getting power metrics: {e}")
            return {}
    
    def _get_cpu_metrics(self):
        """Get CPU usage and frequency"""
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_freq = psutil.cpu_freq()
        
        return {
            'cpu_percent_avg': sum(cpu_percent) / len(cpu_percent),
            'cpu_percent_per_core': cpu_percent,
            'cpu_freq_current_mhz': cpu_freq.current if cpu_freq else None,
            'cpu_freq_max_mhz': cpu_freq.max if cpu_freq else None,
        }
    
    def _get_memory_metrics(self):
        """Get memory usage"""
        mem = psutil.virtual_memory()
        return {
            'memory_used_mb': mem.used / (1024 * 1024),
            'memory_percent': mem.percent,
            'memory_available_mb': mem.available / (1024 * 1024),
        }
    
    def ask_question(self, question, stream=False):
        """Ask a question and measure performance"""
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        
        # Get baseline metrics
        baseline_cpu = psutil.cpu_percent(interval=0.5)
        baseline_mem = psutil.virtual_memory().used / (1024 * 1024)
        
        # Start timing
        start_time = time.time()
        start_timestamp = datetime.now()
        
        # CPU and memory tracking during inference
        cpu_samples = []
        mem_samples = []
        
        response_text = ""
        token_count = 0
        first_token_time = None
        
        try:
            if stream:
                # Streaming response
                stream_response = ollama.chat(
                    model=self.model,
                    messages=[{'role': 'user', 'content': question}],
                    stream=True
                )
                
                for chunk in stream_response:
                    # Record first token time
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                    
                    content = chunk['message']['content']
                    response_text += content
                    token_count += 1
                    
                    # Sample metrics periodically
                    if token_count % 5 == 0:
                        cpu_samples.append(psutil.cpu_percent())
                        mem_samples.append(psutil.virtual_memory().used / (1024 * 1024))
                    
                    print(content, end='', flush=True)
                
                print()  # New line after response
            else:
                # Non-streaming response
                response = ollama.chat(
                    model=self.model,
                    messages=[{'role': 'user', 'content': question}],
                    stream=False
                )
                response_text = response['message']['content']
                
                # Estimate tokens (rough approximation)
                token_count = len(response_text.split())
                
                print(f"\nResponse: {response_text}")
        
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            return None
        
        # End timing
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Get final metrics
        final_cpu = self._get_cpu_metrics()
        final_mem = self._get_memory_metrics()
        final_power = self._get_power_metrics()
        
        # Calculate statistics
        avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else final_cpu['cpu_percent_avg']
        peak_mem = max(mem_samples) if mem_samples else final_mem['memory_used_mb']
        mem_increase = peak_mem - baseline_mem
        
        tokens_per_second = token_count / inference_time if inference_time > 0 else 0
        
        # Compile results
        result = {
            'timestamp': start_timestamp.isoformat(),
            'model': self.model,
            'question': question,
            'response': response_text,
            'response_length_chars': len(response_text),
            'estimated_tokens': token_count,
            
            # Timing metrics
            'inference_time_s': round(inference_time, 3),
            'time_to_first_token_s': round(first_token_time, 3) if first_token_time else None,
            'tokens_per_second': round(tokens_per_second, 2),
            
            # CPU metrics
            'cpu_baseline_percent': round(baseline_cpu, 2),
            'cpu_average_percent': round(avg_cpu, 2),
            'cpu_per_core': [round(x, 2) for x in final_cpu['cpu_percent_per_core']],
            'cpu_freq_mhz': final_cpu['cpu_freq_current_mhz'],
            
            # Memory metrics
            'memory_baseline_mb': round(baseline_mem, 2),
            'memory_peak_mb': round(peak_mem, 2),
            'memory_increase_mb': round(mem_increase, 2),
            'memory_percent': round(final_mem['memory_percent'], 2),
        }
        
        # Add power metrics if available
        if final_power:
            result.update(final_power)
        
        # Print summary
        self._print_summary(result)
        
        # Save result
        self.results.append(result)
        
        return result
    
    def _print_summary(self, result):
        """Print formatted summary of metrics"""
        print(f"\n{'─'*60}")
        print("📊 PERFORMANCE METRICS")
        print(f"{'─'*60}")
        print(f"⏱️  Inference Time: {result['inference_time_s']:.3f}s")
        if result['time_to_first_token_s']:
            print(f"⚡ Time to First Token: {result['time_to_first_token_s']:.3f}s")
        print(f"🚀 Tokens/Second: {result['tokens_per_second']:.2f}")
        print(f"📝 Response Length: {result['response_length_chars']} chars (~{result['estimated_tokens']} tokens)")
        print(f"\n💻 CPU Usage: {result['cpu_average_percent']:.1f}% (baseline: {result['cpu_baseline_percent']:.1f}%)")
        print(f"🧠 Memory: {result['memory_peak_mb']:.0f}MB (Δ {result['memory_increase_mb']:.0f}MB)")
        
        if 'temperature_c' in result:
            print(f"🌡️  Temperature: {result['temperature_c']:.1f}°C")
            print(f"⚡ Voltage: {result['voltage_v']:.2f}V")
        
        print(f"{'─'*60}\n")
    
    def run_benchmark(self, questions, stream=False):
        """Run a benchmark with multiple questions"""
        print(f"\n🔬 Starting Benchmark")
        print(f"Model: {self.model}")
        print(f"Questions: {len(questions)}")
        print(f"Streaming: {stream}\n")
        
        for i, question in enumerate(questions, 1):
            print(f"\n[Question {i}/{len(questions)}]")
            self.ask_question(question, stream=stream)
            time.sleep(1)  # Brief pause between questions
        
        # Print aggregate statistics
        self._print_aggregate_stats()
    
    def _print_aggregate_stats(self):
        """Print aggregate statistics from all results"""
        if not self.results:
            return
        
        avg_inference = sum(r['inference_time_s'] for r in self.results) / len(self.results)
        avg_tokens_per_sec = sum(r['tokens_per_second'] for r in self.results) / len(self.results)
        avg_cpu = sum(r['cpu_average_percent'] for r in self.results) / len(self.results)
        avg_mem = sum(r['memory_peak_mb'] for r in self.results) / len(self.results)
        
        print(f"\n{'='*60}")
        print("📈 AGGREGATE STATISTICS")
        print(f"{'='*60}")
        print(f"Total Questions: {len(self.results)}")
        print(f"Average Inference Time: {avg_inference:.3f}s")
        print(f"Average Tokens/Second: {avg_tokens_per_sec:.2f}")
        print(f"Average CPU Usage: {avg_cpu:.1f}%")
        print(f"Average Memory Usage: {avg_mem:.0f}MB")
        
        if self.results and 'temperature_c' in self.results[0]:
            avg_temp = sum(r.get('temperature_c', 0) for r in self.results) / len(self.results)
            print(f"Average Temperature: {avg_temp:.1f}°C")
        
        print(f"{'='*60}\n")
    
    def save_results(self, format='json'):
        """Save results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'json':
            filename = self.output_dir / f"benchmark_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"✅ Results saved to: {filename}")
        
        elif format == 'csv':
            filename = self.output_dir / f"benchmark_{timestamp}.csv"
            if self.results:
                keys = self.results[0].keys()
                with open(filename, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(self.results)
            print(f"✅ Results saved to: {filename}")
        
        return filename


# Example usage
if __name__ == "__main__":
    # Initialize monitor
    monitor = OllamaMonitor(model="phi3:mini")
    
    # Sample questions
    test_questions = [
        "What is 15 + 27?",
        "Explain photosynthesis in simple terms.",
        "What are the three laws of motion?",
        "How do computers store information?",
        "What is the capital of France?",
    ]
    
    print("="*60)
    print("OLLAMA PERFORMANCE MONITORING TOOL")
    print("="*60)
    
    # Run benchmark
    monitor.run_benchmark(test_questions, stream=True)
    
    # Save results
    monitor.save_results(format='json')
    monitor.save_results(format='csv')
    
    print("\n✨ Benchmark complete!")