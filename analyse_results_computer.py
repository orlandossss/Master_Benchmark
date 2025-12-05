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
    def __init__(self, results_dir="./results_computer"):
        self.results_dir = Path(results_dir)
        self.all_results = []
        self.models_data = defaultdict(list)
        self.ratings = {}

        # Load all results
        self._load_all_results()

    def _parse_model_size(self, param_str):
        """Parse model size from parameter string (e.g., '1,7B' -> 1.7)"""
        try:
            # Remove 'B' and replace comma with dot
            size_str = param_str.upper().replace('B', '').replace(',', '.').strip()
            return float(size_str)
        except (ValueError, AttributeError):
            return 0.0

    def _categorize_models(self, summary):
        """Categorize models into small (<2B) and big (≥2B) based on parameters"""
        small_models = {}
        big_models = {}

        for model, stats in summary.items():
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
            print(f"  Avg IOPS: {stats['avg_io_iops']}")
            print(f"  Avg Throughput: {stats['avg_io_throughput_mb_s']} MB/s")
            print(f"  Total Read Operations: {stats['total_io_read_count']}")
            print(f"  Total Write Operations: {stats['total_io_write_count']}")
            print(f"  Total Data Read: {stats['total_io_read_mb']} MB")
            print(f"  Total Data Written: {stats['total_io_write_mb']} MB")
            print(f"  Avg Read Latency: {stats['avg_io_read_latency_ms']} ms")
            print(f"  Avg Write Latency: {stats['avg_io_write_latency_ms']} ms")

    def generate_graphs(self, output_dir="./analysis_graphs_computer"):
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
            # Sort by model size (ascending - smallest to biggest)
            small_models_list = sorted(small_models.keys(),
                                      key=lambda x: self._parse_model_size(small_models[x].get('model_parameters', '0B')))
            self._plot_tokens_per_second(small_models, small_models_list, output_path, suffix="_small")
            self._plot_energy_efficiency(small_models, small_models_list, output_path, suffix="_small")
            self._plot_inference_times_by_category(small_models, output_path, suffix="_small")
            self._plot_response_analysis_by_category(small_models, output_path, suffix="_small")
            self._plot_resource_usage(small_models, small_models_list, output_path, suffix="_small")
            self._plot_radar_chart(small_models, small_models_list, output_path, suffix="_small")
            self._plot_io_metrics(small_models, small_models_list, output_path, suffix="_small")

        # Generate graphs for BIG MODELS
        if big_models:
            print("\n🔸 Creating graphs for BIG models (≥2B)...")
            # Sort by model size (ascending - smallest to biggest)
            big_models_list = sorted(big_models.keys(),
                                    key=lambda x: self._parse_model_size(big_models[x].get('model_parameters', '0B')))
            self._plot_tokens_per_second(big_models, big_models_list, output_path, suffix="_big")
            self._plot_energy_efficiency(big_models, big_models_list, output_path, suffix="_big")
            self._plot_inference_times_by_category(big_models, output_path, suffix="_big")
            self._plot_response_analysis_by_category(big_models, output_path, suffix="_big")
            self._plot_resource_usage(big_models, big_models_list, output_path, suffix="_big")
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
        self._plot_resource_usage(summary, all_models, output_path, suffix="_all")
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
        """Plot energy efficiency comparison"""
        if not models:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Tokens per Joule
        tpj = [summary[m]['avg_tokens_per_joule'] for m in models]
        colors = plt.cm.RdYlGn([v/max(tpj) if max(tpj) > 0 else 0 for v in tpj])

        bars1 = ax1.bar(range(len(models)), tpj, color=colors, alpha=0.8)
        ax1.set_xlabel('Model', fontsize=11)
        ax1.set_ylabel('Tokens per Joule', fontsize=11)
        category_label = " - Small Models (<2B)" if suffix == "_small" else " - Big Models (≥2B)" if suffix == "_big" else ""
        ax1.set_title(f'Energy Efficiency: Tokens per Joule{category_label}', fontsize=13, fontweight='bold')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels([m.replace(':', '\n') for m in models])

        for bar, val in zip(bars1, tpj):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        # Total Energy Consumption
        energy = [summary[m]['total_energy_joules'] for m in models]
        bars2 = ax2.bar(range(len(models)), energy, color='coral', alpha=0.8)
        ax2.set_xlabel('Model', fontsize=11)
        ax2.set_ylabel('Total Energy (Joules)', fontsize=11)
        ax2.set_title(f'Total Energy Consumption{category_label}', fontsize=13, fontweight='bold')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels([m.replace(':', '\n') for m in models])

        for bar, val in zip(bars2, energy):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{val:.0f}J', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        filename = f'energy_efficiency{suffix}.png'
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

    def _plot_resource_usage(self, summary, models, output_path, suffix=""):
        """Plot resource usage comparison"""
        if not models:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # CPU Usage
        cpu = [summary[m]['avg_cpu_percent'] for m in models]
        axes[0, 0].bar(range(len(models)), cpu, color='indianred', alpha=0.8)
        axes[0, 0].set_title('Average CPU Usage (%)', fontweight='bold')
        axes[0, 0].set_xticks(range(len(models)))
        axes[0, 0].set_xticklabels([m.replace(':', '\n') for m in models])

        # Memory Usage
        mem = [summary[m]['avg_memory_increase_mb'] for m in models]
        axes[0, 1].bar(range(len(models)), mem, color='mediumseagreen', alpha=0.8)
        axes[0, 1].set_title('Average Memory Increase (MB)', fontweight='bold')
        axes[0, 1].set_xticks(range(len(models)))
        axes[0, 1].set_xticklabels([m.replace(':', '\n') for m in models])

        # Temperature
        temp = [summary[m]['avg_temperature_c'] for m in models]
        axes[1, 0].bar(range(len(models)), temp, color='orange', alpha=0.8)
        axes[1, 0].set_title('Average Temperature (C)', fontweight='bold')
        axes[1, 0].set_xticks(range(len(models)))
        axes[1, 0].set_xticklabels([m.replace(':', '\n') for m in models])

        # Time to First Token
        ttft = [summary[m]['avg_ttft_s'] for m in models]
        axes[1, 1].bar(range(len(models)), ttft, color='mediumpurple', alpha=0.8)
        axes[1, 1].set_title('Average Time to First Token (s)', fontweight='bold')
        axes[1, 1].set_xticks(range(len(models)))
        axes[1, 1].set_xticklabels([m.replace(':', '\n') for m in models])

        category_label = " - Small Models (<2B)" if suffix == "_small" else " - Big Models (≥2B)" if suffix == "_big" else ""
        plt.suptitle(f'Resource Usage Comparison{category_label}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        filename = f'resource_usage{suffix}.png'
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"  Created: {filename}")

    def _plot_io_metrics(self, summary, models, output_path, suffix=""):
        """Plot I/O performance metrics (NEW)"""
        if not models:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # IOPS
        iops = [summary[m]['avg_io_iops'] for m in models]
        axes[0, 0].bar(range(len(models)), iops, color='dodgerblue', alpha=0.8)
        axes[0, 0].set_title('Average IOPS', fontweight='bold')
        axes[0, 0].set_ylabel('Operations/Second')
        axes[0, 0].set_xticks(range(len(models)))
        axes[0, 0].set_xticklabels([m.replace(':', '\n') for m in models])
        axes[0, 0].grid(axis='y', alpha=0.3)

        # Throughput
        throughput = [summary[m]['avg_io_throughput_mb_s'] for m in models]
        axes[0, 1].bar(range(len(models)), throughput, color='limegreen', alpha=0.8)
        axes[0, 1].set_title('Average I/O Throughput', fontweight='bold')
        axes[0, 1].set_ylabel('MB/s')
        axes[0, 1].set_xticks(range(len(models)))
        axes[0, 1].set_xticklabels([m.replace(':', '\n') for m in models])
        axes[0, 1].grid(axis='y', alpha=0.3)

        # Read vs Write Operations
        read_ops = [summary[m]['total_io_read_count'] for m in models]
        write_ops = [summary[m]['total_io_write_count'] for m in models]
        x_pos = range(len(models))
        width = 0.35
        axes[1, 0].bar([x - width/2 for x in x_pos], read_ops, width, label='Read', color='skyblue', alpha=0.8)
        axes[1, 0].bar([x + width/2 for x in x_pos], write_ops, width, label='Write', color='salmon', alpha=0.8)
        axes[1, 0].set_title('Total I/O Operations', fontweight='bold')
        axes[1, 0].set_ylabel('Operations Count')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels([m.replace(':', '\n') for m in models])
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)

        # Latency
        read_latency = [summary[m]['avg_io_read_latency_ms'] for m in models]
        write_latency = [summary[m]['avg_io_write_latency_ms'] for m in models]
        axes[1, 1].bar([x - width/2 for x in x_pos], read_latency, width, label='Read', color='skyblue', alpha=0.8)
        axes[1, 1].bar([x + width/2 for x in x_pos], write_latency, width, label='Write', color='salmon', alpha=0.8)
        axes[1, 1].set_title('Average I/O Latency', fontweight='bold')
        axes[1, 1].set_ylabel('Latency (ms)')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels([m.replace(':', '\n') for m in models])
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)

        category_label = " - Small Models (<2B)" if suffix == "_small" else " - Big Models (≥2B)" if suffix == "_big" else ""
        plt.suptitle(f'Disk I/O Performance Metrics{category_label}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        filename = f'io_performance{suffix}.png'
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
                    print(rating_text)
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

    def save_ratings_report(self, output_file="teaching_effectiveness_ratings_computer.json"):
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

    def plot_teaching_scores(self, output_dir="./analysis_graphs_computer"):
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
    analyzer = BenchmarkAnalyzer(results_dir="./results_computer")

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
