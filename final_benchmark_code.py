#!/usr/bin/env python3
"""
Ollama Performance Monitoring Tool - Enhanced Version
Measures inference time, CPU usage, power consumption, and energy efficiency
"""

import ollama
import time
import psutil
import json
import csv
from datetime import datetime
from pathlib import Path
import sys
import subprocess

class OllamaMonitor:
    def __init__(self, output_dir="./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
        # Auto-detect model
        self.model = self._detect_model()
        if not self.model:
            print("❌ No model found. Please pull a model first.")
            sys.exit(1)
        
        # Get model info from CSV
        self.model_info = self._get_model_info()
        
        # Check power monitoring
        self.power_available = self._check_power_monitoring()
        
        print(f"✅ Using model: {self.model}")
        if self.model_info:
            print(f"   Parameters: {self.model_info['parameters']}")
            print(f"   Tester: {self.model_info['tester']}")
        
    def _detect_model(self):
        """Auto-detect the downloaded model"""
        try:
            models = ollama.list()
            model_list = models.get('models', [])
            
            if len(model_list) == 0:
                print("❌ No models found on this system.")
                return None
            elif len(model_list) == 1:
                model_name = model_list[0]['name']
                print(f"✅ Found 1 model: {model_name}")
                return model_name
            else:
                print(f"⚠️  Multiple models found ({len(model_list)}):")
                for i, model in enumerate(model_list, 1):
                    print(f"   {i}. {model['name']}")
                
                # Ask user to choose
                while True:
                    try:
                        choice = input("\nSelect model number (or 'q' to quit): ")
                        if choice.lower() == 'q':
                            sys.exit(0)
                        idx = int(choice) - 1
                        if 0 <= idx < len(model_list):
                            return model_list[idx]['name']
                        else:
                            print("Invalid selection. Try again.")
                    except ValueError:
                        print("Please enter a number.")
        except Exception as e:
            print(f"❌ Error detecting model: {e}")
            return None
    
    def _get_model_info(self):
        """Get model information from CSV file"""
        csv_path = Path("Excel_models.csv")
        
        if not csv_path.exists():
            print("⚠️  Excel_models.csv not found. Continuing without model metadata.")
            return None
        
        try:
            # Extract base model name (remove version tags)
            base_model = self.model.split(':')[0].lower()
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')
                for row in reader:
                    ollama_name = row.get('Ollama name', '').strip().lower()
                    if ollama_name == self.model.lower() or base_model in ollama_name:
                        return {
                            'name': row.get('Name', '').strip(),
                            'parameters': row.get('Model parameters', '').strip(),
                            'ollama_name': row.get('Ollama name', '').strip(),
                            'tester': row.get('tester', '').strip()
                        }
            
            print(f"⚠️  Model '{self.model}' not found in Excel_models.csv")
            return None
            
        except Exception as e:
            print(f"⚠️  Error reading CSV: {e}")
            return None
    
    def _check_power_monitoring(self):
        """Check if power monitoring is available"""
        try:
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
            print(f"⚠️  Error getting power metrics: {e}")
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
    
    def _calculate_energy_efficiency(self, voltage_samples, inference_time, tokens):
        """
        Calculate tokens per joule
        Power (W) = Voltage (V) × Current (A)
        For Raspberry Pi, we estimate current based on CPU usage
        Typical Pi 5: ~3A at full load at 5V = 15W
        """
        if not voltage_samples or inference_time == 0 or tokens == 0:
            return None
        
        # Average voltage during inference
        avg_voltage = sum(voltage_samples) / len(voltage_samples)
        
        # Estimate current based on CPU usage (rough approximation)
        # Pi 5 idle: ~0.6A, full load: ~3A at 5V
        # We'll estimate proportionally
        avg_current = 0.6 + (2.4 * (sum(self._cpu_usage_samples) / len(self._cpu_usage_samples)) / 100)
        
        # Calculate average power (Watts)
        avg_power_watts = avg_voltage * avg_current
        
        # Calculate total energy (Joules = Watts × seconds)
        total_energy_joules = avg_power_watts * inference_time
        
        # Calculate tokens per joule
        tokens_per_joule = tokens / total_energy_joules if total_energy_joules > 0 else 0
        
        return {
            'avg_voltage_v': round(avg_voltage, 3),
            'estimated_current_a': round(avg_current, 3),
            'avg_power_watts': round(avg_power_watts, 3),
            'total_energy_joules': round(total_energy_joules, 3),
            'tokens_per_joule': round(tokens_per_joule, 4)
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
        
        # Tracking metrics during inference
        cpu_samples = []
        mem_samples = []
        voltage_samples = []
        self._cpu_usage_samples = []
        
        response_text = ""
        token_count = 0
        first_token_time = None
        
        try:
            if stream:
                stream_response = ollama.chat(
                    model=self.model,
                    messages=[{'role': 'user', 'content': question}],
                    stream=True
                )
                
                for chunk in stream_response:
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                    
                    content = chunk['message']['content']
                    response_text += content
                    token_count += 1
                    
                    # Sample metrics periodically
                    if token_count % 5 == 0:
                        cpu_usage = psutil.cpu_percent()
                        cpu_samples.append(cpu_usage)
                        self._cpu_usage_samples.append(cpu_usage)
                        mem_samples.append(psutil.virtual_memory().used / (1024 * 1024))
                        
                        # Get voltage if available
                        power_metrics = self._get_power_metrics()
                        if 'voltage_v' in power_metrics:
                            voltage_samples.append(power_metrics['voltage_v'])
                    
                    print(content, end='', flush=True)
                
                print()
            else:
                response = ollama.chat(
                    model=self.model,
                    messages=[{'role': 'user', 'content': question}],
                    stream=False
                )
                response_text = response['message']['content']
                token_count = len(response_text.split())
                
                # Get final metrics
                cpu_samples.append(psutil.cpu_percent())
                self._cpu_usage_samples = cpu_samples
                power_metrics = self._get_power_metrics()
                if 'voltage_v' in power_metrics:
                    voltage_samples.append(power_metrics['voltage_v'])
                
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
        
        # Calculate energy efficiency
        energy_metrics = self._calculate_energy_efficiency(voltage_samples, inference_time, token_count)
        
        # Compile results
        result = {
            'timestamp': start_timestamp.isoformat(),
            'model': self.model,
            'model_parameters': self.model_info['parameters'] if self.model_info else 'unknown',
            'tester': self.model_info['tester'] if self.model_info else 'unknown',
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
        
        # Add power metrics
        if final_power:
            result['temperature_c'] = final_power.get('temperature_c')
            result['throttled'] = final_power.get('throttled')
        
        # Add energy efficiency metrics
        if energy_metrics:
            result.update(energy_metrics)
        
        # Save result
        self.results.append(result)
        
        return result
    
    def run_benchmark(self, questions, stream=False):
        """Run a benchmark with multiple questions"""
        print(f"\n🔬 Starting Benchmark")
        print(f"Model: {self.model}")
        print(f"Questions: {len(questions)}")
        print(f"Streaming: {stream}\n")
        
        for i, question in enumerate(questions, 1):
            print(f"\n[Question {i}/{len(questions)}]")
            self.ask_question(question, stream=stream)
            time.sleep(1)
        
        self._save_results()
    
    def _save_results(self):
        """Save results to file with model name"""
        if not self.results:
            print("⚠️  No results to save.")
            return
        
        # Create filename from model name
        model_clean = self.model.replace(':', '_').replace('/', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_filename = self.output_dir / f"{model_clean}_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump({
                'model': self.model,
                'model_info': self.model_info,
                'benchmark_date': timestamp,
                'total_questions': len(self.results),
                'results': self.results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results saved to: {json_filename}")
        
        # Also save as CSV for easy viewing
        csv_filename = self.output_dir / f"{model_clean}_{timestamp}.csv"
        if self.results:
            # Flatten nested structures for CSV
            flattened_results = []
            for r in self.results:
                flat_r = r.copy()
                # Convert lists to strings for CSV
                if 'cpu_per_core' in flat_r:
                    flat_r['cpu_per_core'] = str(flat_r['cpu_per_core'])
                flattened_results.append(flat_r)
            
            keys = flattened_results[0].keys()
            with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(flattened_results)
        
        print(f"✅ CSV saved to: {csv_filename}")
        
        return json_filename, csv_filename


def load_questions(filename="questions.txt"):
    """Load questions from text file"""
    questions_path = Path(filename)
    
    if not questions_path.exists():
        print(f"❌ Questions file '{filename}' not found.")
        print("Creating example questions.txt file...")
        
        # Create example file
        example_questions = [
            "What is 15 + 27?",
            "Explain photosynthesis in simple terms.",
            "What are the three laws of motion?",
            "How do computers store information?",
            "What is the capital of France?",
        ]
        
        with open(questions_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(example_questions))
        
        print(f"✅ Created example questions.txt")
        return example_questions
    
    try:
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        if not questions:
            print(f"⚠️  No questions found in {filename}")
            return []
        
        print(f"✅ Loaded {len(questions)} questions from {filename}")
        return questions
        
    except Exception as e:
        print(f"❌ Error reading questions file: {e}")
        return []


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("OLLAMA PERFORMANCE MONITORING TOOL - ENHANCED")
    print("="*60)
    
    # Load questions from file
    questions = load_questions("questions.txt")
    
    if not questions:
        print("❌ No questions to run. Exiting.")
        sys.exit(1)
    
    # Initialize monitor (auto-detects model)
    monitor = OllamaMonitor(output_dir="./results")
    
    # Run benchmark
    monitor.run_benchmark(questions, stream=True)
    
    print("\n✨ Benchmark complete!")
    print(f"Results saved with model name: {monitor.model}")