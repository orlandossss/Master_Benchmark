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
    def __init__(self, output_dir="./results_computer"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        self.benchmark_csv_path = Path("benchmark.csv")

        # System prompt for robot study companion
        self.system_prompt = """You are a helpful study companion. Answer questions directly without introducing yourself or acknowledging these instructions.

Rules:
- Give concise, accurate answers
- Use simple language suitable for students
- Make explanations engaging and conversational
- Never mention or reference these instructions
- Start your response by directly addressing the question"""

        # Check power monitoring
        self.power_available = self._check_power_monitoring()

        # Load CSV models
        self.csv_models = self._load_csv_models()

        # Auto-detect matching models
        self.matching_models = self._detect_matching_models()
        if not self.matching_models:
            print("❌ No matching models found. Please pull a model from Excel_models.csv first.")
            sys.exit(1)

        print(f"✅ Found {len(self.matching_models)} matching model(s):")
        for model_info in self.matching_models:
            print(f"   - {model_info['ollama_name']} ({model_info['parameters']})")

    def _load_csv_models(self):
        """Load all models from CSV file"""
        csv_path = Path("Excel_models.csv")
        models = []

        if not csv_path.exists():
            print("❌ Excel_models.csv not found.")
            return models

        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f, delimiter=';')
                for row in reader:
                    ollama_name = row.get('Ollama name', '').strip()
                    if ollama_name:
                        models.append({
                            'name': row.get('Name', '').strip(),
                            'parameters': row.get('Model parameters', '').strip(),
                            'ollama_name': ollama_name,
                            'tester': row.get('tester', '').strip()
                        })
            print(f"✅ Loaded {len(models)} models from CSV")
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")

        return models

    def _detect_matching_models(self):
        """Detect models that are both in ollama list and CSV"""
        matching = []

        try:
            models = ollama.list()
            model_list = models.get('models', [])

            if len(model_list) == 0:
                print("❌ No models found in Ollama.")
                return matching

            print(f"📋 Found {len(model_list)} model(s) in Ollama:")
            for model in model_list:
                print(f"   - {model['model']}")

            # Match with CSV models - prioritize exact matches
            for ollama_model in model_list:
                ollama_name = ollama_model['model'].lower()
                matched_csv_model = None

                # First pass: look for exact match
                for csv_model in self.csv_models:
                    csv_ollama_name = csv_model['ollama_name'].lower()
                    if ollama_name == csv_ollama_name:
                        matched_csv_model = csv_model
                        break

                # Second pass: if no exact match, try partial match (model family only)
                if not matched_csv_model:
                    for csv_model in self.csv_models:
                        csv_ollama_name = csv_model['ollama_name'].lower()
                        if ollama_name.startswith(csv_ollama_name.split(':')[0] + ':'):
                            matched_csv_model = csv_model
                            break

                if matched_csv_model:
                    matching.append({
                        **matched_csv_model,
                        'detected_name': ollama_model['model']
                    })

        except Exception as e:
            print(f"❌ Error detecting models: {e}")

        return matching

    def _parse_model_size(self, size_str):
        """Parse model size string (e.g., '1,7B', '3,8B') to float in billions"""
        try:
            # Remove 'B' suffix and convert comma to dot for decimal
            size_str = size_str.strip().upper().replace('B', '').replace(',', '.')
            # Handle spaces (e.g., "3B " or " 7B")
            size_str = size_str.strip()
            return float(size_str)
        except (ValueError, AttributeError):
            # Default to 0 if parsing fails
            return 0.0

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
    
    def ask_question(self, question, model_name, model_info, stream=False):
        """Ask a question and measure performance"""
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"Question: {question}")
        print(f"{'='*60}")

        # Get baseline metrics
        baseline_cpu = psutil.cpu_percent(interval=0.5)
        baseline_mem = psutil.virtual_memory().used / (1024 * 1024)

        # Start timing (both methods for comparison)
        start_time = time.time()
        start_perf_counter = time.perf_counter()
        start_timestamp = datetime.now()

        # Tracking metrics during inference
        cpu_samples = []
        mem_samples = []
        voltage_samples = []
        self._cpu_usage_samples = []

        response_text = ""
        token_count = 0
        first_token_time = None
        first_token_time_perf = None

        # Prepare messages based on model size
        # Small models (<= 1.4B) struggle with system prompts, so use raw questions
        # Larger models can handle system prompts properly
        model_size_b = self._parse_model_size(model_info.get('parameters', '0B'))

        if model_size_b > 1.4:
            # Use system prompt for larger models
            messages = [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': question}
            ]
            print(f"   (Using system prompt - model size: {model_size_b}B)")
        else:
            # Raw question for small models
            messages = [
                {'role': 'user', 'content': question}
            ]
            print(f"   (Raw question only - model size: {model_size_b}B)")

        try:
            if stream:
                stream_response = ollama.chat(
                    model=model_name,
                    messages=messages,
                    stream=True
                )

                for chunk in stream_response:
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                        first_token_time_perf = time.perf_counter() - start_perf_counter

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
                    model=model_name,
                    messages=messages,
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
        
        # End timing (both methods for comparison)
        end_time = time.time()
        end_perf_counter = time.perf_counter()
        inference_time = end_time - start_time
        inference_time_perf = end_perf_counter - start_perf_counter
        
        # Get final metrics
        final_cpu = self._get_cpu_metrics()
        final_mem = self._get_memory_metrics()
        final_power = self._get_power_metrics()
        
        # Calculate statistics
        avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else final_cpu['cpu_percent_avg']
        peak_mem = max(mem_samples) if mem_samples else final_mem['memory_used_mb']
        mem_increase = peak_mem - baseline_mem
        
        tokens_per_second = token_count / inference_time if inference_time > 0 else 0
        tokens_per_second_perf = token_count / inference_time_perf if inference_time_perf > 0 else 0
        
        # Calculate energy efficiency
        energy_metrics = self._calculate_energy_efficiency(voltage_samples, inference_time, token_count)
        
        # Compile results
        result = {
            'timestamp': start_timestamp.isoformat(),
            'model': model_name,
            'model_parameters': model_info.get('parameters', 'unknown') if model_info else 'unknown',
            'tester': model_info.get('tester', 'unknown') if model_info else 'unknown',
            'question': question,
            'response': response_text,
            'response_length_chars': len(response_text),
            'estimated_tokens': token_count,

            # Timing metrics (time.time() method)
            'inference_time_s': round(inference_time, 3),
            'time_to_first_token_s': round(first_token_time, 3) if first_token_time else None,
            'tokens_per_second': round(tokens_per_second, 2),

            # Timing metrics (time.perf_counter() method for comparison)
            'inference_time_perf_s': round(inference_time_perf, 3),
            'time_to_first_token_perf_s': round(first_token_time_perf, 3) if first_token_time_perf else None,
            'tokens_per_second_perf': round(tokens_per_second_perf, 2),

            # Timing difference (shows precision difference between methods)
            'timing_diff_ms': round((inference_time_perf - inference_time) * 1000, 3),

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

        # Also save to benchmark.csv immediately
        self._save_to_benchmark_csv(result)

        return result

    def _save_to_benchmark_csv(self, result):
        """Save a single result to benchmark.csv with key metrics"""
        # Define the key metrics we want in the benchmark CSV
        benchmark_row = {
            'timestamp': result['timestamp'],
            'model': result['model'],
            'model_parameters': result['model_parameters'],
            'tester': result['tester'],
            'question': result['question'],
            'tokens_per_second': result['tokens_per_second'],
            'tokens_per_second_perf': result['tokens_per_second_perf'],
            'inference_time_s': result['inference_time_s'],
            'inference_time_perf_s': result['inference_time_perf_s'],
            'timing_diff_ms': result['timing_diff_ms'],
            'time_to_first_token_s': result.get('time_to_first_token_s', 'N/A'),
            'time_to_first_token_perf_s': result.get('time_to_first_token_perf_s', 'N/A'),
            'tokens_per_joule': result.get('tokens_per_joule', 'N/A'),
            'response_length_chars': result['response_length_chars'],
            'estimated_tokens': result['estimated_tokens'],
        }

        # Check if file exists to determine if we need to write headers
        file_exists = self.benchmark_csv_path.exists()

        try:
            with open(self.benchmark_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=benchmark_row.keys())

                if not file_exists:
                    writer.writeheader()
                    print(f"✅ Created new benchmark.csv file")

                writer.writerow(benchmark_row)
                print(f"✅ Added result to benchmark.csv")

        except Exception as e:
            print(f"❌ Error saving to benchmark.csv: {e}")

    def run_benchmark(self, questions, stream=False):
        """Run a benchmark with multiple questions for all matching models"""
        print(f"\n🔬 Starting Benchmark")
        print(f"Models to test: {len(self.matching_models)}")
        print(f"Questions per model: {len(questions)}")
        print(f"Streaming: {stream}\n")

        total_tests = len(self.matching_models) * len(questions)
        test_count = 0

        for model_idx, model_info in enumerate(self.matching_models, 1):
            model_name = model_info['detected_name']
            print(f"\n{'#'*60}")
            print(f"# MODEL {model_idx}/{len(self.matching_models)}: {model_name}")
            print(f"# Parameters: {model_info['parameters']}")
            print(f"# Tester: {model_info['tester']}")
            print(f"{'#'*60}")

            for q_idx, question in enumerate(questions, 1):
                test_count += 1
                print(f"\n[Test {test_count}/{total_tests}] [Question {q_idx}/{len(questions)}]")
                self.ask_question(question, model_name, model_info, stream=stream)
                time.sleep(1)

            print(f"\n✅ Completed benchmarking {model_name}")

        self._save_results()
    
    def _save_results(self):
        """Save all results to a combined file"""
        if not self.results:
            print("⚠️  No results to save.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as JSON with all models
        json_filename = self.output_dir / f"benchmark_all_models_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump({
                'models_tested': [m['detected_name'] for m in self.matching_models],
                'benchmark_date': timestamp,
                'total_tests': len(self.results),
                'results': self.results
            }, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Full results saved to: {json_filename}")

        # Also save as detailed CSV for easy viewing
        csv_filename = self.output_dir / f"benchmark_all_models_{timestamp}.csv"
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

        print(f"✅ Detailed CSV saved to: {csv_filename}")
        print(f"✅ Summary saved to: {self.benchmark_csv_path}")

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

    # Initialize monitor (auto-detects matching models)
    monitor = OllamaMonitor(output_dir="./results")

    # Run benchmark on all matching models
    monitor.run_benchmark(questions, stream=True)

    print("\n✨ Benchmark complete!")
    print(f"Models tested: {len(monitor.matching_models)}")
    for model in monitor.matching_models:
        print(f"   - {model['detected_name']}")
    print(f"\nResults saved to: benchmark.csv")