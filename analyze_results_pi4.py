#!/usr/bin/env python3
"""
Benchmark Results Analyzer for Pi4 - Enhanced with I/O Metrics
Aggregates results, generates visualizations, and rates teaching effectiveness using OpenAI API
"""

import json
import os
from pathlib import Path
from datetime import datetime
import statistics
from collections import defaultdict

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    print("Warning: python-dotenv not installed. Run: pip install python-dotenv")
    print("         Environment variables must be set manually.")

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not installed. Run: pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("Warning: pandas not installed. Run: pip install pandas")
    PANDAS_AVAILABLE = False

# OpenAI API for LLM rating
try:
    import openai
    OPENAI_AVAILABLE = True
    # Check if new API (v1.0.0+) or old API
    OPENAI_NEW_API = hasattr(openai, 'OpenAI')
except ImportError:
    print("Warning: openai not installed. Run: pip install openai")
    OPENAI_AVAILABLE = False
    OPENAI_NEW_API = False


class BenchmarkAnalyzer:
    def __init__(self, results_dir="./results_pi4"):
        self.results_dir = Path(results_dir)
        self.all_results = []
        self.models_data = defaultdict(list)
        self.ratings = {}

        # Load all results
        self._load_all_results()

    def _parse_model_size(self, param_str):
        """Parse model size from parameter string (e.g., '1,7B' -> 1.7, '500M' -> 0.5)"""
        try:
            if not param_str or param_str == 'unknown':
                return 0.0

            # Convert to uppercase for consistent parsing
            param_upper = str(param_str).upper().strip()

            # Handle millions (M) - convert to billions
            if 'M' in param_upper:
                size_str = param_upper.replace('M', '').replace(',', '.').strip()
                return float(size_str) / 1000.0  # Convert millions to billions

            # Handle billions (B)
            if 'B' in param_upper:
                size_str = param_upper.replace('B', '').replace(',', '.').strip()
                return float(size_str)

            # If no unit, assume it's already a number in billions
            size_str = param_upper.replace(',', '.').strip()
            return float(size_str)

        except (ValueError, AttributeError, TypeError):
            # Return 0 for unparseable values
            return 0.0

    def _get_model_category_override(self, model_name):
        """Manual override for specific models that need special categorization"""
        # Models that should be in big category regardless of parameter count
        big_model_overrides = [
            'granite4:tiny-h',
            'granite3-dense:2b',
            'granite4:3b-h',
        ]

        model_lower = model_name.lower()
        for override in big_model_overrides:
            if override.lower() in model_lower:
                return 'big'

        return None  # No override, use normal categorization

    def _categorize_models(self, summary):
        """Categorize models into small (<2B) and big (≥2B) based on parameters"""
        small_models = {}
        big_models = {}

        for model, stats in summary.items():
            # Check for manual override first
            override = self._get_model_category_override(model)

            if override == 'big':
                big_models[model] = stats
            else:
                # Use automatic categorization based on parameter size
                model_size = self._parse_model_size(stats.get('model_parameters', '0B'))
                if model_size < 2.0:
                    small_models[model] = stats
                else:
                    big_models[model] = stats

        return small_models, big_models

    def _load_all_results(self):
        """Load all JSON result files from the results directory"""
        json_files = list(self.results_dir.glob("*.json"))

        if not json_files:
            print(f"No JSON files found in {self.results_dir}")
            return

        print(f"Found {len(json_files)} result file(s)")

        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                results = data.get('results', [])
                print(f"  - {json_file.name}: {len(results)} test(s)")

                for result in results:
                    self.all_results.append(result)
                    model_name = result.get('model', 'unknown')
                    self.models_data[model_name].append(result)

            except Exception as e:
                print(f"  Error loading {json_file.name}: {e}")

        print(f"\nTotal results loaded: {len(self.all_results)}")
        print(f"Unique models: {len(self.models_data)}")
        for model, results in self.models_data.items():
            print(f"  - {model}: {len(results)} test(s)")

    def get_summary_statistics(self):
        """Calculate summary statistics for each model"""
        summary = {}

        for model, results in self.models_data.items():
            if not results:
                continue

            # Extract metrics
            tokens_per_second = [r.get('tokens_per_second', 0) for r in results]
            inference_times = [r.get('inference_time_s', 0) for r in results]
            tokens_per_joule = [r.get('tokens_per_joule', 0) for r in results if r.get('tokens_per_joule')]
            response_lengths = [r.get('response_length_chars', 0) for r in results]
            cpu_usage = [r.get('cpu_average_percent', 0) for r in results]
            memory_increase = [r.get('memory_increase_mb', 0) for r in results]
            temperatures = [r.get('temperature_c', 0) for r in results if r.get('temperature_c')]
            ttft = [r.get('time_to_first_token_s', 0) for r in results if r.get('time_to_first_token_s')]

            # Extract I/O metrics
            io_iops = [r.get('io_iops', 0) for r in results if r.get('io_iops')]
            # Separate cold start (Q1) from warm operation (Q2-10)
            io_iops_cold = [results[0].get('io_iops', 0)] if results and results[0].get('io_iops') else []
            io_iops_warm = [r.get('io_iops', 0) for r in results[1:] if r.get('io_iops')]

            io_throughput = [r.get('io_throughput_mb_s', 0) for r in results if r.get('io_throughput_mb_s')]
            io_read_count = [r.get('io_read_count', 0) for r in results if r.get('io_read_count')]
            io_write_count = [r.get('io_write_count', 0) for r in results if r.get('io_write_count')]
            io_read_bytes = [r.get('io_read_bytes', 0) for r in results if r.get('io_read_bytes')]
            io_write_bytes = [r.get('io_write_bytes', 0) for r in results if r.get('io_write_bytes')]
            io_read_latency = [r.get('io_avg_read_latency_ms', 0) for r in results if r.get('io_avg_read_latency_ms')]
            io_write_latency = [r.get('io_avg_write_latency_ms', 0) for r in results if r.get('io_avg_write_latency_ms')]

            summary[model] = {
                'model_parameters': results[0].get('model_parameters', 'unknown'),
                'tester': results[0].get('tester', 'unknown'),
                'num_tests': len(results),

                # Performance metrics
                'avg_tokens_per_second': round(statistics.mean(tokens_per_second), 2),
                'std_tokens_per_second': round(statistics.stdev(tokens_per_second), 2) if len(tokens_per_second) > 1 else 0,
                'min_tokens_per_second': round(min(tokens_per_second), 2),
                'max_tokens_per_second': round(max(tokens_per_second), 2),

                # Timing
                'avg_inference_time_s': round(statistics.mean(inference_times), 2),
                'total_inference_time_s': round(sum(inference_times), 2),
                'avg_ttft_s': round(statistics.mean(ttft), 2) if ttft else 0,

                # Energy efficiency
                'avg_tokens_per_joule': round(statistics.mean(tokens_per_joule), 4) if tokens_per_joule else 0,
                'total_energy_joules': round(sum([r.get('total_energy_joules', 0) for r in results]), 2),

                # Response quality metrics
                'avg_response_length': round(statistics.mean(response_lengths), 0),
                'total_tokens_generated': sum([r.get('estimated_tokens', 0) for r in results]),

                # Resource usage
                'avg_cpu_percent': round(statistics.mean(cpu_usage), 2),
                'avg_memory_increase_mb': round(statistics.mean(memory_increase), 2),
                'avg_temperature_c': round(statistics.mean(temperatures), 1) if temperatures else 0,

                # I/O metrics
                'avg_io_iops': round(statistics.mean(io_iops), 2) if io_iops else 0,
                'avg_io_iops_cold_start': round(statistics.mean(io_iops_cold), 2) if io_iops_cold else 0,
                'avg_io_iops_warm': round(statistics.mean(io_iops_warm), 2) if io_iops_warm else 0,
                'avg_io_throughput_mb_s': round(statistics.mean(io_throughput), 3) if io_throughput else 0,
                'total_io_read_count': sum(io_read_count) if io_read_count else 0,
                'total_io_write_count': sum(io_write_count) if io_write_count else 0,
                'total_io_read_mb': round(sum(io_read_bytes) / (1024 * 1024), 2) if io_read_bytes else 0,
                'total_io_write_mb': round(sum(io_write_bytes) / (1024 * 1024), 2) if io_write_bytes else 0,
                'avg_io_read_latency_ms': round(statistics.mean(io_read_latency), 2) if io_read_latency else 0,
                'avg_io_write_latency_ms': round(statistics.mean(io_write_latency), 2) if io_write_latency else 0,
            }

        return summary

    def print_summary(self):
        """Print a formatted summary of all results"""
        summary = self.get_summary_statistics()

        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY (PI4 ENHANCED)")
        print("="*80)

        for model, stats in summary.items():
            print(f"\n{'='*60}")
            print(f"MODEL: {model}")
            print(f"Parameters: {stats['model_parameters']} | Tester: {stats['tester']}")
            print(f"Tests Run: {stats['num_tests']}")
            print(f"{'='*60}")

            print("\nPERFORMANCE:")
            print(f"  Tokens/Second: {stats['avg_tokens_per_second']} avg "
                  f"(min: {stats['min_tokens_per_second']}, max: {stats['max_tokens_per_second']})")
            print(f"  Avg Inference Time: {stats['avg_inference_time_s']}s")
            print(f"  Avg Time to First Token: {stats['avg_ttft_s']}s")

            print("\nENERGY EFFICIENCY:")
            print(f"  Tokens/Joule: {stats['avg_tokens_per_joule']}")
            print(f"  Total Energy Used: {stats['total_energy_joules']} J")

            print("\nRESPONSE QUALITY:")
            print(f"  Avg Response Length: {stats['avg_response_length']} chars")
            print(f"  Total Tokens Generated: {stats['total_tokens_generated']}")

            print("\nRESOURCE USAGE:")
            print(f"  Avg CPU: {stats['avg_cpu_percent']}%")
            print(f"  Avg Memory Increase: {stats['avg_memory_increase_mb']} MB")
            print(f"  Avg Temperature: {stats['avg_temperature_c']}C")

            print("\nDISK I/O METRICS:")
            print(f"  Avg IOPS (All Questions): {stats['avg_io_iops']}")
            print(f"  Cold Start IOPS (Q1): {stats['avg_io_iops_cold_start']}")
            print(f"  Warm IOPS (Q2-10): {stats['avg_io_iops_warm']}")
            print(f"  Avg Throughput: {stats['avg_io_throughput_mb_s']} MB/s")
            print(f"  Total Read Operations: {stats['total_io_read_count']}")
            print(f"  Total Write Operations: {stats['total_io_write_count']}")
            print(f"  Total Data Read: {stats['total_io_read_mb']} MB")
            print(f"  Total Data Written: {stats['total_io_write_mb']} MB")
            print(f"  Avg Read Latency: {stats['avg_io_read_latency_ms']} ms")
            print(f"  Avg Write Latency: {stats['avg_io_write_latency_ms']} ms")

    def generate_graphs(self, output_dir="./final_result"):
        """Generate visualization graphs for the benchmark results"""
        if not MATPLOTLIB_AVAILABLE:
            print("Cannot generate graphs: matplotlib not installed")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        summary = self.get_summary_statistics()

        if not summary:
            print("No data to visualize")
            return

        # Categorize models by size
        small_models, big_models = self._categorize_models(summary)

        # Set style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

        print("\n📊 Generating graphs for model categories:")
        print(f"   Small models (<2B): {len(small_models)}")
        print(f"   Big models (≥2B): {len(big_models)}")

        # Generate graphs for SMALL MODELS
        if small_models:
            print("\n🔹 Creating graphs for SMALL models (<2B)...")

            # Debug: print model sizes before sorting
            print("\n  Model sizes (for sorting):")
            for model, stats in small_models.items():
                param_str = stats.get('model_parameters', '0B')
                parsed_size = self._parse_model_size(param_str)
                print(f"    {model}: '{param_str}' -> {parsed_size}B")

            # Sort by model size (ascending - smallest to biggest)
            small_models_list = sorted(small_models.keys(),
                                      key=lambda x: self._parse_model_size(small_models[x].get('model_parameters', '0B')))

            print(f"\n  Sorted order: {small_models_list}\n")
            self._plot_tokens_per_second(small_models, small_models_list, output_path, suffix="_small")
            self._plot_energy_efficiency(small_models, small_models_list, output_path, suffix="_small")
            self._plot_inference_times_by_category(small_models, output_path, suffix="_small")
            self._plot_response_analysis_by_category(small_models, output_path, suffix="_small")
            self._plot_response_length(small_models, small_models_list, output_path, suffix="_small")
            self._plot_resource_usage(small_models, small_models_list, output_path, suffix="_small")
            self._plot_ttft_first_vs_average(small_models, small_models_list, output_path, suffix="_small")
            self._plot_inference_time_first_vs_average(small_models, small_models_list, output_path, suffix="_small")
            self._plot_iops_first_vs_average(small_models, small_models_list, output_path, suffix="_small")
            self._plot_radar_chart(small_models, small_models_list, output_path, suffix="_small")
            self._plot_io_metrics(small_models, small_models_list, output_path, suffix="_small")

        # Generate graphs for BIG MODELS
        if big_models:
            print("\n🔸 Creating graphs for BIG models (≥2B)...")

            # Debug: print model sizes before sorting
            print("\n  Model sizes (for sorting):")
            for model, stats in big_models.items():
                param_str = stats.get('model_parameters', '0B')
                parsed_size = self._parse_model_size(param_str)
                print(f"    {model}: '{param_str}' -> {parsed_size}B")

            # Sort by model size (ascending - smallest to biggest)
            big_models_list = sorted(big_models.keys(),
                                    key=lambda x: self._parse_model_size(big_models[x].get('model_parameters', '0B')))

            print(f"\n  Sorted order: {big_models_list}\n")
            self._plot_tokens_per_second(big_models, big_models_list, output_path, suffix="_big")
            self._plot_energy_efficiency(big_models, big_models_list, output_path, suffix="_big")
            self._plot_inference_times_by_category(big_models, output_path, suffix="_big")
            self._plot_response_analysis_by_category(big_models, output_path, suffix="_big")
            self._plot_response_length(big_models, big_models_list, output_path, suffix="_big")
            self._plot_resource_usage(big_models, big_models_list, output_path, suffix="_big")
            self._plot_ttft_first_vs_average(big_models, big_models_list, output_path, suffix="_big")
            self._plot_inference_time_first_vs_average(big_models, big_models_list, output_path, suffix="_big")
            self._plot_iops_first_vs_average(big_models, big_models_list, output_path, suffix="_big")
            self._plot_radar_chart(big_models, big_models_list, output_path, suffix="_big")
            self._plot_io_metrics(big_models, big_models_list, output_path, suffix="_big")

        # Generate combined graphs for ALL MODELS (optional)
        print("\n🔷 Creating combined graphs for ALL models...")
        # Sort by model size (ascending - smallest to biggest)
        all_models = sorted(summary.keys(),
                           key=lambda x: self._parse_model_size(summary[x].get('model_parameters', '0B')))
        self._plot_tokens_per_second(summary, all_models, output_path, suffix="_all")
        self._plot_energy_efficiency(summary, all_models, output_path, suffix="_all")
        self._plot_inference_times_by_category(summary, output_path, suffix="_all")
        self._plot_response_analysis_by_category(summary, output_path, suffix="_all")
        self._plot_response_length(summary, all_models, output_path, suffix="_all")
        self._plot_resource_usage(summary, all_models, output_path, suffix="_all")
        self._plot_ttft_first_vs_average(summary, all_models, output_path, suffix="_all")
        self._plot_inference_time_first_vs_average(summary, all_models, output_path, suffix="_all")
        self._plot_iops_first_vs_average(summary, all_models, output_path, suffix="_all")
        self._plot_radar_chart(summary, all_models, output_path, suffix="_all")
        self._plot_io_metrics(summary, all_models, output_path, suffix="_all")

        print(f"\n✅ Graphs saved to: {output_path}")

    def _plot_tokens_per_second(self, summary, models, output_path, suffix=""):
        """Plot tokens per second comparison"""
        if not models:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        x_pos = range(len(models))
        avgs = [summary[m]['avg_tokens_per_second'] for m in models]
        stds = [summary[m]['std_tokens_per_second'] for m in models]

        bars = ax.bar(x_pos, avgs, yerr=stds, capsize=5, color='steelblue', alpha=0.8)

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Tokens per Second', fontsize=12)
        category_label = " - Small Models (<2B)" if suffix == "_small" else " - Big Models (≥2B)" if suffix == "_big" else ""
        ax.set_title(f'Model Performance: Tokens per Second{category_label}', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace(':', '\n') for m in models], rotation=0)

        # Add value labels on bars
        for bar, avg in zip(bars, avgs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{avg:.2f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        filename = f'tokens_per_second{suffix}.png'
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"  Created: {filename}")

    def _plot_energy_efficiency(self, summary, models, output_path, suffix=""):
        """Plot energy efficiency comparison - split into separate graphs"""
        if not models:
            return

        category_label = " - Small Models (<2B)" if suffix == "_small" else " - Big Models (≥2B)" if suffix == "_big" else ""

        # Graph 1: Tokens per Joule
        fig, ax = plt.subplots(figsize=(12, 6))
        tpj = [summary[m]['avg_tokens_per_joule'] for m in models]
        colors = plt.cm.RdYlGn([v/max(tpj) if max(tpj) > 0 else 0 for v in tpj])

        bars = ax.bar(range(len(models)), tpj, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Tokens per Joule', fontsize=12)
        ax.set_title(f'Energy Efficiency: Tokens per Joule{category_label}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace(':', '\n') for m in models])
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, tpj):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(tpj)*0.02 if max(tpj) > 0 else 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        filename = f'tokens_per_joule{suffix}.png'
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"  Created: {filename}")

        # Graph 2: Total Energy Consumption
        fig, ax = plt.subplots(figsize=(12, 6))
        energy = [summary[m]['total_energy_joules'] for m in models]
        bars = ax.bar(range(len(models)), energy, color='coral', alpha=0.8, edgecolor='darkred', linewidth=1.2)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Total Energy (Joules)', fontsize=12)
        ax.set_title(f'Total Energy Consumption{category_label}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace(':', '\n') for m in models])
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, energy):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(energy)*0.02 if max(energy) > 0 else 10,
                    f'{val:.0f}J', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        filename = f'total_energy_consumption{suffix}.png'
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"  Created: {filename}")

    def _plot_inference_times_by_category(self, summary, output_path, suffix=""):
        """Plot inference time distribution per model (using summary dict)"""
        if not summary:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        data = []
        labels = []
        for model, stats in summary.items():
            # Get results for this specific model from self.models_data
            if model in self.models_data:
                times = [r.get('inference_time_s', 0) for r in self.models_data[model]]
                data.append(times)
                labels.append(model.replace(':', '\n'))

        if not data:
            plt.close()
            return

        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        colors = plt.cm.Set3(range(len(data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Inference Time (seconds)', fontsize=12)
        category_label = " - Small Models (<2B)" if suffix == "_small" else " - Big Models (≥2B)" if suffix == "_big" else ""
        ax.set_title(f'Inference Time Distribution{category_label}', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        filename = f'inference_time_distribution{suffix}.png'
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"  Created: {filename}")

    def _plot_response_analysis_by_category(self, summary, output_path, suffix=""):
        """Plot response length vs performance"""
        if not summary:
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # Filter models in this category
        models_in_category = list(summary.keys())
        colors = plt.cm.tab10(range(len(models_in_category)))

        for idx, model in enumerate(models_in_category):
            if model in self.models_data:
                results = self.models_data[model]
                lengths = [r.get('response_length_chars', 0) for r in results]
                tps = [r.get('tokens_per_second', 0) for r in results]

                ax.scatter(lengths, tps, label=model, color=colors[idx],
                          alpha=0.7, s=100, edgecolors='black', linewidths=0.5)

        ax.set_xlabel('Response Length (characters)', fontsize=12)
        ax.set_ylabel('Tokens per Second', fontsize=12)
        category_label = " - Small Models (<2B)" if suffix == "_small" else " - Big Models (≥2B)" if suffix == "_big" else ""
        ax.set_title(f'Response Length vs Performance{category_label}', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        filename = f'response_vs_performance{suffix}.png'
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"  Created: {filename}")

    def _plot_response_length(self, summary, models, output_path, suffix=""):
        """Plot average response length comparison"""
        if not models:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        x_pos = range(len(models))
        lengths = [summary[m]['avg_response_length'] for m in models]

        # Color code by length (higher = more green)
        colors = plt.cm.Blues([l/max(lengths) if max(lengths) > 0 else 0 for l in lengths])
        bars = ax.bar(x_pos, lengths, color=colors, alpha=0.8, edgecolor='navy', linewidth=1.2)

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Average Response Length (characters)', fontsize=12)
        category_label = " - Small Models (<2B)" if suffix == "_small" else " - Big Models (≥2B)" if suffix == "_big" else ""
        ax.set_title(f'Average Response Length{category_label}', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace(':', '\n') for m in models], rotation=0)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, length in zip(bars, lengths):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(lengths)*0.01,
                   f'{int(length)}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        plt.tight_layout()
        filename = f'response_length{suffix}.png'
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"  Created: {filename}")

    def _plot_resource_usage(self, summary, models, output_path, suffix=""):
        """Plot resource usage comparison - split into separate graphs"""
        if not models:
            return

        category_label = " - Small Models (<2B)" if suffix == "_small" else " - Big Models (≥2B)" if suffix == "_big" else ""

        # Graph 1: CPU Usage
        fig, ax = plt.subplots(figsize=(12, 6))
        cpu = [summary[m]['avg_cpu_percent'] for m in models]
        bars = ax.bar(range(len(models)), cpu, color='indianred', alpha=0.8, edgecolor='darkred', linewidth=1.2)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('CPU Usage (%)', fontsize=12)
        ax.set_title(f'Average CPU Usage{category_label}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace(':', '\n') for m in models])
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, cpu):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cpu)*0.02 if max(cpu) > 0 else 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        filename = f'cpu_usage{suffix}.png'
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"  Created: {filename}")

        # Graph 2: Memory Usage
        fig, ax = plt.subplots(figsize=(12, 6))
        mem = [summary[m]['avg_memory_increase_mb'] for m in models]
        bars = ax.bar(range(len(models)), mem, color='mediumseagreen', alpha=0.8, edgecolor='darkgreen', linewidth=1.2)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Memory Increase (MB)', fontsize=12)
        ax.set_title(f'Average Memory Increase{category_label}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace(':', '\n') for m in models])
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, mem):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mem)*0.02 if max(mem) > 0 else 10,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        filename = f'memory_usage{suffix}.png'
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"  Created: {filename}")

        # Graph 3: Temperature
        fig, ax = plt.subplots(figsize=(12, 6))
        temp = [summary[m]['avg_temperature_c'] for m in models]
        bars = ax.bar(range(len(models)), temp, color='orange', alpha=0.8, edgecolor='darkorange', linewidth=1.2)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.set_title(f'Average Temperature{category_label}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace(':', '\n') for m in models])
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, temp):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(temp)*0.02 if max(temp) > 0 else 1,
                    f'{val:.1f}°C', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        filename = f'temperature{suffix}.png'
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"  Created: {filename}")

        # Graph 4: Time to First Token
        fig, ax = plt.subplots(figsize=(12, 6))
        ttft = [summary[m]['avg_ttft_s'] for m in models]
        bars = ax.bar(range(len(models)), ttft, color='mediumpurple', alpha=0.8, edgecolor='purple', linewidth=1.2)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Time to First Token (seconds)', fontsize=12)
        ax.set_title(f'Average Time to First Token{category_label}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace(':', '\n') for m in models])
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, ttft):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ttft)*0.02 if max(ttft) > 0 else 0.1,
                    f'{val:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        filename = f'time_to_first_token{suffix}.png'
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"  Created: {filename}")

    def _plot_ttft_first_vs_average(self, summary, models, output_path, suffix=""):
        """Plot TTFT comparison: first question vs average of questions 2-10"""
        if not models:
            return

        # Note: summary parameter kept for consistency with other plotting methods
        category_label = " - Small Models (<2B)" if suffix == "_small" else " - Big Models (≥2B)" if suffix == "_big" else ""

        # Extract TTFT for first question and average of questions 2-10
        first_question_ttft = []
        avg_other_questions_ttft = []

        for model in models:
            if model in self.models_data:
                results = self.models_data[model]
                ttft_values = [r.get('time_to_first_token_s', 0) for r in results if r.get('time_to_first_token_s')]

                if ttft_values:
                    first_question_ttft.append(ttft_values[0] if len(ttft_values) > 0 else 0)
                    avg_other_questions_ttft.append(
                        sum(ttft_values[1:]) / len(ttft_values[1:]) if len(ttft_values) > 1 else 0
                    )
                else:
                    first_question_ttft.append(0)
                    avg_other_questions_ttft.append(0)
            else:
                first_question_ttft.append(0)
                avg_other_questions_ttft.append(0)

        # Graph 1: First Question TTFT
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(models)), first_question_ttft, color='mediumpurple', alpha=0.8, edgecolor='purple', linewidth=1.2)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Time to First Token (seconds)', fontsize=12)
        ax.set_title(f'TTFT - First Question{category_label}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace(':', '\n') for m in models])
        ax.grid(axis='y', alpha=0.3)

        max_val = max(first_question_ttft) if first_question_ttft else 1
        for bar, val in zip(bars, first_question_ttft):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val*0.02,
                        f'{val:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        filename = f'ttft_first_question{suffix}.png'
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"  Created: {filename}")

        # Graph 2: Average of Questions 2-10 TTFT
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(models)), avg_other_questions_ttft, color='mediumorchid', alpha=0.8, edgecolor='purple', linewidth=1.2)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Time to First Token (seconds)', fontsize=12)
        ax.set_title(f'TTFT - Average of Questions 2-10{category_label}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace(':', '\n') for m in models])
        ax.grid(axis='y', alpha=0.3)

        max_val = max(avg_other_questions_ttft) if avg_other_questions_ttft else 1
        for bar, val in zip(bars, avg_other_questions_ttft):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val*0.02,
                        f'{val:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        filename = f'ttft_avg_questions_2_10{suffix}.png'
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"  Created: {filename}")

    def _plot_inference_time_first_vs_average(self, summary, models, output_path, suffix=""):
        """Plot inference time comparison: first question vs average of questions 2-10"""
        if not models:
            return

        # Note: summary parameter kept for consistency with other plotting methods
        category_label = " - Small Models (<2B)" if suffix == "_small" else " - Big Models (≥2B)" if suffix == "_big" else ""

        # Extract inference time for first question and average of questions 2-10
        first_question_time = []
        avg_other_questions_time = []

        for model in models:
            if model in self.models_data:
                results = self.models_data[model]
                inference_times = [r.get('inference_time_s', 0) for r in results if r.get('inference_time_s')]

                if inference_times:
                    first_question_time.append(inference_times[0] if len(inference_times) > 0 else 0)
                    avg_other_questions_time.append(
                        sum(inference_times[1:]) / len(inference_times[1:]) if len(inference_times) > 1 else 0
                    )
                else:
                    first_question_time.append(0)
                    avg_other_questions_time.append(0)
            else:
                first_question_time.append(0)
                avg_other_questions_time.append(0)

        # Graph 1: First Question Inference Time
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(models)), first_question_time, color='steelblue', alpha=0.8, edgecolor='navy', linewidth=1.2)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Inference Time (seconds)', fontsize=12)
        ax.set_title(f'Inference Time - First Question{category_label}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace(':', '\n') for m in models])
        ax.grid(axis='y', alpha=0.3)

        max_val = max(first_question_time) if first_question_time else 1
        for bar, val in zip(bars, first_question_time):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val*0.02,
                        f'{val:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        filename = f'inference_time_first_question{suffix}.png'
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"  Created: {filename}")

        # Graph 2: Average of Questions 2-10 Inference Time
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(models)), avg_other_questions_time, color='cornflowerblue', alpha=0.8, edgecolor='navy', linewidth=1.2)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Inference Time (seconds)', fontsize=12)
        ax.set_title(f'Inference Time - Average of Questions 2-10{category_label}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace(':', '\n') for m in models])
        ax.grid(axis='y', alpha=0.3)

        max_val = max(avg_other_questions_time) if avg_other_questions_time else 1
        for bar, val in zip(bars, avg_other_questions_time):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val*0.02,
                        f'{val:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        filename = f'inference_time_avg_questions_2_10{suffix}.png'
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"  Created: {filename}")

    def _plot_iops_first_vs_average(self, summary, models, output_path, suffix=""):
        """Plot IOPS comparison: first question (cold start) vs average of questions 2-10 (warm)"""
        if not models:
            return

        # Note: summary parameter kept for consistency with other plotting methods
        category_label = " - Small Models (<2B)" if suffix == "_small" else " - Big Models (≥2B)" if suffix == "_big" else ""

        # Extract IOPS for first question and average of questions 2-10
        first_question_iops = []
        avg_other_questions_iops = []

        for model in models:
            if model in self.models_data:
                results = self.models_data[model]
                iops_values = [r.get('io_iops', 0) for r in results if r.get('io_iops')]

                if iops_values:
                    first_question_iops.append(iops_values[0] if len(iops_values) > 0 else 0)
                    avg_other_questions_iops.append(
                        sum(iops_values[1:]) / len(iops_values[1:]) if len(iops_values) > 1 else 0
                    )
                else:
                    first_question_iops.append(0)
                    avg_other_questions_iops.append(0)
            else:
                first_question_iops.append(0)
                avg_other_questions_iops.append(0)

        # Graph 1: First Question IOPS (Cold Start)
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(models)), first_question_iops, color='darkorange', alpha=0.8, edgecolor='darkred', linewidth=1.2)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('IOPS (Operations/Second)', fontsize=12)
        ax.set_title(f'Cold Start IOPS - First Question{category_label}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace(':', '\n') for m in models])
        ax.grid(axis='y', alpha=0.3)

        max_val = max(first_question_iops) if first_question_iops else 1
        for bar, val in zip(bars, first_question_iops):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val*0.02,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        filename = f'iops_cold_start_first_question{suffix}.png'
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"  Created: {filename}")

        # Graph 2: Average of Questions 2-10 IOPS (Warm Operation)
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(models)), avg_other_questions_iops, color='limegreen', alpha=0.8, edgecolor='darkgreen', linewidth=1.2)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('IOPS (Operations/Second)', fontsize=12)
        ax.set_title(f'Warm IOPS - Average of Questions 2-10{category_label}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace(':', '\n') for m in models])
        ax.grid(axis='y', alpha=0.3)

        max_val = max(avg_other_questions_iops) if avg_other_questions_iops else 1
        for bar, val in zip(bars, avg_other_questions_iops):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val*0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        filename = f'iops_warm_avg_questions_2_10{suffix}.png'
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"  Created: {filename}")

    def _plot_io_metrics(self, summary, models, output_path, suffix=""):
        """Plot I/O performance metrics - split into separate graphs"""
        if not models:
            return

        category_label = " - Small Models (<2B)" if suffix == "_small" else " - Big Models (≥2B)" if suffix == "_big" else ""

        # Graph 1: IOPS
        fig, ax = plt.subplots(figsize=(12, 6))
        iops = [summary[m]['avg_io_iops'] for m in models]
        bars = ax.bar(range(len(models)), iops, color='dodgerblue', alpha=0.8, edgecolor='darkblue', linewidth=1.2)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Operations/Second', fontsize=12)
        ax.set_title(f'Average IOPS{category_label}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace(':', '\n') for m in models])
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, iops):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(iops)*0.02 if max(iops) > 0 else 1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        filename = f'iops{suffix}.png'
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"  Created: {filename}")

        # Graph 2: Throughput
        fig, ax = plt.subplots(figsize=(12, 6))
        throughput = [summary[m]['avg_io_throughput_mb_s'] for m in models]
        bars = ax.bar(range(len(models)), throughput, color='limegreen', alpha=0.8, edgecolor='darkgreen', linewidth=1.2)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Throughput (MB/s)', fontsize=12)
        ax.set_title(f'Average I/O Throughput{category_label}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace(':', '\n') for m in models])
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, throughput):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(throughput)*0.02 if max(throughput) > 0 else 0.1,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        filename = f'io_throughput{suffix}.png'
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"  Created: {filename}")

        # Graph 3: Read vs Write Operations
        fig, ax = plt.subplots(figsize=(12, 6))
        read_ops = [summary[m]['total_io_read_count'] for m in models]
        write_ops = [summary[m]['total_io_write_count'] for m in models]
        x_pos = range(len(models))
        width = 0.35
        bars1 = ax.bar([x - width/2 for x in x_pos], read_ops, width, label='Read', color='skyblue', alpha=0.8, edgecolor='darkblue', linewidth=1.2)
        bars2 = ax.bar([x + width/2 for x in x_pos], write_ops, width, label='Write', color='salmon', alpha=0.8, edgecolor='darkred', linewidth=1.2)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Operations Count', fontsize=12)
        ax.set_title(f'Total I/O Operations (Read vs Write){category_label}', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace(':', '\n') for m in models])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        filename = f'io_operations{suffix}.png'
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"  Created: {filename}")

        # Graph 4: Latency
        fig, ax = plt.subplots(figsize=(12, 6))
        read_latency = [summary[m]['avg_io_read_latency_ms'] for m in models]
        write_latency = [summary[m]['avg_io_write_latency_ms'] for m in models]
        bars1 = ax.bar([x - width/2 for x in x_pos], read_latency, width, label='Read', color='skyblue', alpha=0.8, edgecolor='darkblue', linewidth=1.2)
        bars2 = ax.bar([x + width/2 for x in x_pos], write_latency, width, label='Write', color='salmon', alpha=0.8, edgecolor='darkred', linewidth=1.2)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title(f'Average I/O Latency (Read vs Write){category_label}', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace(':', '\n') for m in models])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        filename = f'io_latency{suffix}.png'
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"  Created: {filename}")

    def _plot_radar_chart(self, summary, models, output_path, suffix=""):
        """Create a radar chart comparing models across multiple dimensions"""
        if len(models) < 1:
            return

        # Metrics to compare (normalized 0-1, higher is better)
        metrics = ['Speed', 'Energy Eff.', 'Response Len', 'Low Memory', 'Low Temp', 'Fast TTFT', 'I/O Perf']

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        angles = [n / float(len(metrics)) * 2 * 3.14159 for n in range(len(metrics))]
        angles += angles[:1]  # Complete the circle

        colors = plt.cm.Set2(range(len(models)))

        for idx, model in enumerate(models):
            stats = summary[model]

            # Normalize metrics (0-1 scale, higher = better)
            max_tps = max([summary[m]['avg_tokens_per_second'] for m in models])
            max_tpj = max([summary[m]['avg_tokens_per_joule'] for m in models]) or 1
            max_len = max([summary[m]['avg_response_length'] for m in models])
            max_mem = max([summary[m]['avg_memory_increase_mb'] for m in models]) or 1
            max_temp = max([summary[m]['avg_temperature_c'] for m in models]) or 1
            max_ttft = max([summary[m]['avg_ttft_s'] for m in models]) or 1
            max_iops = max([summary[m]['avg_io_iops'] for m in models]) or 1

            values = [
                stats['avg_tokens_per_second'] / max_tps if max_tps > 0 else 0,
                stats['avg_tokens_per_joule'] / max_tpj if max_tpj > 0 else 0,
                stats['avg_response_length'] / max_len if max_len > 0 else 0,
                1 - (stats['avg_memory_increase_mb'] / max_mem),  # Lower is better
                1 - (stats['avg_temperature_c'] / max_temp),  # Lower is better
                1 - (stats['avg_ttft_s'] / max_ttft),  # Lower is better
                stats['avg_io_iops'] / max_iops if max_iops > 0 else 0,  # Higher is better
            ]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, 1)
        category_label = " - Small Models (<2B)" if suffix == "_small" else " - Big Models (≥2B)" if suffix == "_big" else ""
        ax.set_title(f'Model Comparison Radar Chart{category_label}', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        plt.tight_layout()
        filename = f'model_radar_chart{suffix}.png'
        plt.savefig(output_path / filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Created: {filename}")

    def rate_teaching_effectiveness(self, api_key=None, model="gpt-4o-mini"):
        """Use OpenAI API to rate the teaching effectiveness of each response"""
        if not OPENAI_AVAILABLE:
            print("Cannot rate responses: openai package not installed")
            print("Install with: pip install openai")
            return {}

        if not api_key:
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                print("No API key provided. Set OPENAI_API_KEY environment variable or pass api_key parameter")
                return {}

        # Handle both old and new OpenAI API versions
        if OPENAI_NEW_API:
            client = openai.OpenAI(api_key=api_key)
        else:
            # Old API (< v1.0.0)
            openai.api_key = api_key
            client = None

        print("\nRating teaching effectiveness of responses...")
        print(f"Using model: {model}")
        print("This may take a while depending on the number of responses.\n")

        ratings = []

        for idx, result in enumerate(self.all_results):
            question = result.get('question', '')
            response = result.get('response', '')
            model_name = result.get('model', 'unknown')

            print(f"Rating {idx+1}/{len(self.all_results)}: {model_name} - {question[:50]}...")

            try:
                rating_prompt = f"""You are an expert educator evaluating the teaching effectiveness of AI responses.

Rate the following response on a scale of 1-10 based on its ability to teach effectively, try to be sure that your rating is strict enough.

Consider these criteria:
1. Clarity: Is the explanation clear and easy to understand?
2. Accuracy: Is the information correct?
3. Engagement: Does it engage the learner?
4. Structure: Is it well-organized?
5. Completeness: Does it adequately address the question?
6. Actionable: is it suitable for a talking conversation with a student ?

QUESTION: {question}

RESPONSE TO EVALUATE:
{response}

Provide your rating in this exact JSON format:
{{
    "score": <number 1-10>,
    "strengths": ["<strength 1>", "<strength 2>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"],
    "brief_justification": "<1-2 sentence explanation>"
}}

Only output the JSON, nothing else."""

                # Call API based on version
                if OPENAI_NEW_API:
                    completion = client.chat.completions.create(
                        model=model,
                        max_tokens=500,
                        messages=[
                            {"role": "user", "content": rating_prompt}
                        ]
                    )
                    rating_text = completion.choices[0].message.content.strip()
                else:
                    # Old API (< v1.0.0)
                    completion = openai.ChatCompletion.create(
                        model=model,
                        max_tokens=500,
                        messages=[
                            {"role": "user", "content": rating_prompt}
                        ]
                    )
                    rating_text = completion['choices'][0]['message']['content'].strip()

                # Parse the response
                rating_data = json.loads(rating_text)

                rating_entry = {
                    'model': model_name,
                    'model_parameters': result.get('model_parameters', 'unknown'),
                    'question': question,
                    'response_preview': response[:200] + '...' if len(response) > 200 else response,
                    'score': rating_data.get('score', 0),
                    'strengths': rating_data.get('strengths', []),
                    'weaknesses': rating_data.get('weaknesses', []),
                    'justification': rating_data.get('brief_justification', ''),
                    'tokens_per_second': result.get('tokens_per_second', 0),
                    'inference_time_s': result.get('inference_time_s', 0),
                }

                ratings.append(rating_entry)
                print(f"  Score: {rating_data.get('score', 'N/A')}/10")

            except json.JSONDecodeError as e:
                print(f"  Error parsing rating response: {e}")
                ratings.append({
                    'model': model_name,
                    'question': question,
                    'score': 0,
                    'error': 'Failed to parse rating'
                })
            except Exception as e:
                print(f"  Error rating response: {e}")
                ratings.append({
                    'model': model_name,
                    'question': question,
                    'score': 0,
                    'error': str(e)
                })

        self.ratings = ratings
        return ratings

    def save_ratings_report(self, output_file="teaching_effectiveness_ratings_pi4.json"):
        """Save the teaching effectiveness ratings to a file"""
        if not self.ratings:
            print("No ratings to save. Run rate_teaching_effectiveness() first.")
            return

        # Calculate summary statistics
        model_scores = defaultdict(list)
        for rating in self.ratings:
            if rating.get('score', 0) > 0:
                model_scores[rating['model']].append(rating['score'])

        summary = {}
        for model, scores in model_scores.items():
            summary[model] = {
                'average_score': round(statistics.mean(scores), 2),
                'min_score': min(scores),
                'max_score': max(scores),
                'num_rated': len(scores)
            }

        report = {
            'generated_at': datetime.now().isoformat(),
            'total_responses_rated': len(self.ratings),
            'model_summary': summary,
            'detailed_ratings': self.ratings
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\nRatings report saved to: {output_file}")

        # Print summary
        print("\n" + "="*60)
        print("TEACHING EFFECTIVENESS SUMMARY")
        print("="*60)

        for model, stats in summary.items():
            print(f"\n{model}:")
            print(f"  Average Score: {stats['average_score']}/10")
            print(f"  Score Range: {stats['min_score']} - {stats['max_score']}")
            print(f"  Responses Rated: {stats['num_rated']}")

    def plot_teaching_scores(self, output_dir="./final_result"):
        """Generate visualizations for teaching effectiveness scores"""
        if not MATPLOTLIB_AVAILABLE:
            print("Cannot generate graphs: matplotlib not installed")
            return

        if not self.ratings:
            print("No ratings available. Run rate_teaching_effectiveness() first.")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Group scores by model
        model_scores = defaultdict(list)
        for rating in self.ratings:
            if rating.get('score', 0) > 0:
                model_scores[rating['model']].append(rating['score'])

        # Categorize models by size
        summary = self.get_summary_statistics()
        small_models_set = set()
        big_models_set = set()

        for model, stats in summary.items():
            model_size = self._parse_model_size(stats.get('model_parameters', '0B'))
            if model_size < 2.0:
                small_models_set.add(model)
            else:
                big_models_set.add(model)

        # Filter models that have ratings and sort by size (ascending - smallest to biggest)
        small_models = sorted([m for m in model_scores.keys() if m in small_models_set],
                             key=lambda x: self._parse_model_size(summary.get(x, {}).get('model_parameters', '0B')))
        big_models = sorted([m for m in model_scores.keys() if m in big_models_set],
                           key=lambda x: self._parse_model_size(summary.get(x, {}).get('model_parameters', '0B')))
        all_models = sorted(model_scores.keys(),
                           key=lambda x: self._parse_model_size(summary.get(x, {}).get('model_parameters', '0B')))

        print("\n📊 Generating teaching effectiveness graphs:")
        print(f"   Small models (<2B): {len(small_models)}")
        print(f"   Big models (≥2B): {len(big_models)}")

        # Generate graphs for SMALL MODELS
        if small_models:
            print("\n🔹 Creating teaching graphs for SMALL models (<2B)...")
            self._plot_teaching_bar_chart(small_models, model_scores, output_path, suffix="_small")
            self._plot_performance_vs_teaching(small_models, output_path, suffix="_small")

        # Generate graphs for BIG MODELS
        if big_models:
            print("\n🔸 Creating teaching graphs for BIG models (≥2B)...")
            self._plot_teaching_bar_chart(big_models, model_scores, output_path, suffix="_big")
            self._plot_performance_vs_teaching(big_models, output_path, suffix="_big")

        # Generate combined graphs for ALL MODELS
        print("\n🔷 Creating teaching graphs for ALL models...")
        self._plot_teaching_bar_chart(all_models, model_scores, output_path, suffix="_all")
        self._plot_performance_vs_teaching(all_models, output_path, suffix="_all")

    def _plot_teaching_bar_chart(self, models, model_scores, output_path, suffix=""):
        """Generate bar chart of teaching effectiveness scores"""
        if not models:
            return

        avg_scores = [statistics.mean(model_scores[m]) for m in models]

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.RdYlGn([s/10 for s in avg_scores])
        bars = ax.bar(range(len(models)), avg_scores, color=colors, alpha=0.8, edgecolor='black')

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Teaching Effectiveness Score', fontsize=12)
        category_label = " - Small Models (<2B)" if suffix == "_small" else " - Big Models (≥2B)" if suffix == "_big" else ""
        ax.set_title(f'Average Teaching Effectiveness by Model{category_label}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace(':', '\n') for m in models])
        ax.set_ylim(0, 10)
        ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='Average (5)')
        ax.axhline(y=7, color='green', linestyle='--', alpha=0.5, label='Good (7)')

        for bar, score in zip(bars, avg_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{score:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

        ax.legend()
        plt.tight_layout()
        filename = f'teaching_effectiveness_scores{suffix}.png'
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"  Created: {filename}")

    def _plot_performance_vs_teaching(self, models, output_path, suffix=""):
        """Generate scatter plot of performance vs teaching quality"""
        if not models:
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.tab10(range(len(models)))

        for idx, model in enumerate(models):
            model_data = [r for r in self.ratings if r.get('model') == model and r.get('score', 0) > 0]
            scores = [r['score'] for r in model_data]
            tps = [r.get('tokens_per_second', 0) for r in model_data]

            ax.scatter(tps, scores, label=model, color=colors[idx], s=100, alpha=0.7, edgecolors='black')

        ax.set_xlabel('Tokens per Second (Performance)', fontsize=12)
        ax.set_ylabel('Teaching Effectiveness Score', fontsize=12)
        category_label = " - Small Models (<2B)" if suffix == "_small" else " - Big Models (≥2B)" if suffix == "_big" else ""
        ax.set_title(f'Performance vs Teaching Quality{category_label}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 10)

        plt.tight_layout()
        filename = f'performance_vs_teaching{suffix}.png'
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"  Created: {filename}")

    def export_to_csv(self, output_file="analysis_summary_pi4.csv"):
        """Export summary statistics to CSV"""
        if not PANDAS_AVAILABLE:
            print("Cannot export to CSV: pandas not installed")
            return

        summary = self.get_summary_statistics()
        df = pd.DataFrame.from_dict(summary, orient='index')
        df.index.name = 'model'
        df.to_csv(output_file)
        print(f"Summary exported to: {output_file}")


def main():
    print("="*60)
    print("BENCHMARK RESULTS ANALYZER - PI4 ENHANCED")
    print("="*60)

    # Initialize analyzer
    analyzer = BenchmarkAnalyzer(results_dir="./results_pi4")

    if not analyzer.all_results:
        print("No results to analyze. Exiting.")
        return

    # Print summary statistics
    analyzer.print_summary()

    # Generate graphs
    print("\nGenerating visualization graphs...")
    analyzer.generate_graphs()

    # Export to CSV
    analyzer.export_to_csv()

    # Rate teaching effectiveness (optional - requires API key)
    print("\n" + "="*60)
    print("TEACHING EFFECTIVENESS RATING")
    print("="*60)

    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        print("OpenAI API key found. Starting rating process...")
        # You can change the model here: gpt-4o-mini, gpt-4o, gpt-4-turbo, gpt-3.5-turbo
        ratings = analyzer.rate_teaching_effectiveness(api_key, model="gpt-4o-mini")

        if ratings:
            analyzer.save_ratings_report()
            analyzer.plot_teaching_scores()
    else:
        print("No OPENAI_API_KEY environment variable found.")
        print("To rate teaching effectiveness, set your API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'  (Linux/Mac)")
        print("  set OPENAI_API_KEY=your-api-key-here  (Windows CMD)")
        print("  $env:OPENAI_API_KEY='your-api-key-here'  (Windows PowerShell)")
        print("\nOr pass it programmatically:")
        print("  analyzer.rate_teaching_effectiveness(api_key='your-key', model='gpt-4o-mini')")
        print("\nAvailable models: gpt-4o-mini (cheapest), gpt-4o, gpt-4-turbo, gpt-3.5-turbo")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
