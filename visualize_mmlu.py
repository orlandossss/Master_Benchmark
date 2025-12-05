import os
import json
import matplotlib.pyplot as plt
import re

def load_mmlu_results(mmlu_dir="./MMLU"):
    """Load all MMLU results from JSON files"""
    results = []

    if not os.path.exists(mmlu_dir):
        print(f"[ERROR] Directory {mmlu_dir} does not exist!")
        return results

    # Find all JSON files
    json_files = [f for f in os.listdir(mmlu_dir) if f.endswith('_MMLU.json')]

    if not json_files:
        print(f"[ERROR] No MMLU JSON files found in {mmlu_dir}")
        return results

    print(f"[OK] Found {len(json_files)} result files\n")

    # Load each JSON file
    for json_file in json_files:
        file_path = os.path.join(mmlu_dir, json_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                model_name = data.get('model_name', 'Unknown')
                status = data.get('status', 'Unknown')
                overall_score = data.get('overall_score', 0.0)

                # Check if we have task_scores (new format with multiple tasks)
                if 'task_scores' in data and isinstance(data['task_scores'], dict):
                    # Create a separate result entry for each task
                    task_scores = data['task_scores']
                    print(f"[+] Loaded: {model_name} - {len(task_scores)} tasks - Overall: {overall_score:.4f}")
                    for task_name, task_score in task_scores.items():
                        results.append({
                            'model_name': model_name,
                            'overall_score': task_score,
                            'task': task_name,
                            'status': status
                        })
                        print(f"   - {task_name}: {task_score:.4f}")
                else:
                    # Old format with single task
                    results.append({
                        'model_name': model_name,
                        'overall_score': overall_score,
                        'task': data.get('task', 'Unknown'),
                        'status': status
                    })
                    print(f"[+] Loaded: {model_name} - Task: {data.get('task', 'Unknown')} - Score: {overall_score:.4f}")
        except Exception as e:
            print(f"[WARNING] Error loading {json_file}: {e}")

    return results


def _parse_model_size(model_name):
    """Extract model size in billions from model name"""
    # Look for patterns like "1.7b", "3B", "0.6b", etc.
    match = re.search(r'(\d+\.?\d*)[bB]', model_name)
    if match:
        return float(match.group(1))
    return 0.0


def categorize_models(results):
    """Categorize models into small (<2B) and big (≥2B)"""
    small_models = []
    big_models = []

    for result in results:
        model_size = _parse_model_size(result['model_name'])
        if model_size < 2.0:
            small_models.append(result)
        else:
            big_models.append(result)

    return small_models, big_models


def get_tasks_from_results(results):
    """Extract unique tasks from results"""
    tasks = set()
    for result in results:
        if result['status'] == 'success' or result['status'] == 'Unknown':
            tasks.add(result['task'])
    return sorted(list(tasks))


def plot_overall_scores(results, suffix="", category_label=""):
    """Create a bar chart of overall MMLU scores across all tasks"""

    if not results:
        print(f"[ERROR] No results to plot for {category_label}!")
        return

    # Group by model and calculate average score across all tasks
    model_scores = {}
    for result in results:
        model_name = result['model_name']
        if model_name not in model_scores:
            model_scores[model_name] = []
        model_scores[model_name].append(result['overall_score'])

    # Calculate average scores
    model_avg_scores = {model: sum(scores)/len(scores) for model, scores in model_scores.items()}

    # Sort by model size (ascending - smallest to biggest)
    sorted_models = sorted(model_avg_scores.items(), key=lambda x: _parse_model_size(x[0]))
    model_names = [m[0] for m in sorted_models]
    scores = [m[1] for m in sorted_models]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 7))

    # Create bar chart
    bars = ax.bar(range(len(model_names)), scores, color='steelblue', edgecolor='navy', linewidth=1.2)

    # Customize the plot
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average MMLU Score', fontsize=12, fontweight='bold')
    title = f'MMLU Benchmark - Overall Scores{category_label}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on top of bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add a horizontal line for random guessing (25% for 4 choices)
    ax.axhline(y=0.25, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Random Guessing (25%)')
    ax.legend()

    # Tight layout
    plt.tight_layout()

    # Save the plot
    output_path = f'./MMLU/mmlu_overall_scores{suffix}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] Chart saved to: {output_path}")
    plt.close()


def plot_task_specific_scores(results, suffix="", category_label=""):
    """Create individual bar charts for each task comparing model performance"""

    if not results:
        print(f"[ERROR] No results to plot for {category_label}!")
        return

    # Get all unique tasks
    tasks = get_tasks_from_results(results)

    if not tasks:
        print(f"[ERROR] No tasks found in results!")
        return

    # Organize data by model and task
    model_task_scores = {}
    for result in results:
        model_name = result['model_name']
        task = result['task']
        score = result['overall_score']

        if model_name not in model_task_scores:
            model_task_scores[model_name] = {}
        model_task_scores[model_name][task] = score

    # Sort by model size (ascending - smallest to biggest)
    model_names = sorted(model_task_scores.keys(), key=lambda x: _parse_model_size(x))

    # Create a separate graph for each task
    for task in tasks:
        # Get scores for this specific task
        task_scores = [model_task_scores.get(model, {}).get(task, 0) for model in model_names]

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(14, 7))

        # Create bar chart for this task
        bars = ax.bar(range(len(model_names)), task_scores, color='steelblue', edgecolor='navy', linewidth=1.2)

        # Customize the plot
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('MMLU Score', fontsize=12, fontweight='bold')
        title = f'MMLU Benchmark - {task}{category_label}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels on top of bars
        for i, (bar, score) in enumerate(zip(bars, task_scores)):
            if score > 0:  # Only show label if there's a score
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{score:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Add a horizontal line for random guessing
        ax.axhline(y=0.25, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Random Guessing (25%)')
        ax.legend()

        plt.tight_layout()

        # Save the plot with task name in filename
        safe_task_name = task.replace(':', '_').replace('/', '_').replace(' ', '_')
        output_path = f'./MMLU/mmlu_task_{safe_task_name}{suffix}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Chart saved to: {output_path}")
        plt.close()


def plot_individual_model_radar(results, output_dir='./MMLU'):
    """Create radar charts for each model showing strengths and weaknesses across tasks"""

    if not results:
        print("[ERROR] No results to plot!")
        return

    # Get all unique tasks
    tasks = get_tasks_from_results(results)

    if not tasks or len(tasks) < 3:
        print(f"[WARNING] Need at least 3 tasks for radar chart. Found: {len(tasks)}")
        return

    # Organize data by model
    model_task_scores = {}
    for result in results:
        model_name = result['model_name']
        task = result['task']
        score = result['overall_score']

        if model_name not in model_task_scores:
            model_task_scores[model_name] = {}
        model_task_scores[model_name][task] = score

    # Create individual radar chart for each model
    num_tasks = len(tasks)
    angles = [n / float(num_tasks) * 2 * 3.14159 for n in range(num_tasks)]
    angles += angles[:1]  # Close the circle

    for model_name, task_scores in model_task_scores.items():
        # Prepare data
        values = [task_scores.get(task, 0) for task in tasks]
        values += values[:1]  # Close the circle

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        # Plot data
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color='steelblue')
        ax.fill(angles, values, alpha=0.25, color='steelblue')

        # Customize
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tasks, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'])
        ax.grid(True)

        # Add title with average score
        avg_score = sum(task_scores.values()) / len(task_scores)
        ax.set_title(f'{model_name}\nAverage Score: {avg_score:.3f}',
                     fontsize=14, fontweight='bold', pad=20)

        # Add reference circle for random guessing
        random_line = [0.25] * len(angles)
        ax.plot(angles, random_line, 'r--', linewidth=1.5, alpha=0.7, label='Random (25%)')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        plt.tight_layout()

        # Save the plot
        safe_model_name = model_name.replace(':', '_').replace('/', '_')
        output_path = f'{output_dir}/radar_{safe_model_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Radar chart saved: {output_path}")
        plt.close()


def print_summary(results):
    """Print a summary table of results"""

    if not results:
        return

    print("\n" + "="*80)
    print("MMLU RESULTS SUMMARY")
    print("="*80)

    # Group by model and calculate average across tasks
    model_scores = {}
    model_tasks = {}
    for result in results:
        model_name = result['model_name']
        if model_name not in model_scores:
            model_scores[model_name] = []
            model_tasks[model_name] = []
        model_scores[model_name].append(result['overall_score'])
        model_tasks[model_name].append(result['task'])

    # Calculate averages
    model_avg_scores = {model: sum(scores)/len(scores) for model, scores in model_scores.items()}

    # Sort by model size (ascending - smallest to biggest)
    sorted_models = sorted(model_avg_scores.items(), key=lambda x: _parse_model_size(x[0]))

    print("\nModel Rankings (Sorted by Size - Smallest to Biggest):")
    print("-"*80)
    for idx, (model, avg_score) in enumerate(sorted_models, 1):
        tasks_tested = len(model_scores[model])
        model_size = _parse_model_size(model)
        size_str = f"{model_size:.1f}B" if model_size > 0 else "N/A"
        print(f"{idx:2d}. {model:30s} ({size_str:>6s}) - {avg_score:.4f} ({avg_score*100:.2f}%) - {tasks_tested} task(s)")

    # Statistics
    if len(model_avg_scores) > 0:
        avg_scores_list = list(model_avg_scores.values())
        print("\n[STATISTICS]")
        print("-"*80)
        print(f"Number of unique models tested: {len(model_avg_scores)}")
        print(f"Total test results: {len(results)}")
        print(f"Best average score:  {max(avg_scores_list):.4f} ({max(avg_scores_list)*100:.2f}%)")
        print(f"Worst average score: {min(avg_scores_list):.4f} ({min(avg_scores_list)*100:.2f}%)")
        print(f"Mean score:  {sum(avg_scores_list)/len(avg_scores_list):.4f} ({sum(avg_scores_list)/len(avg_scores_list)*100:.2f}%)")

        # Tasks breakdown
        tasks = get_tasks_from_results(results)
        print(f"\nTasks tested: {', '.join(tasks)}")

    print("="*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MMLU RESULTS VISUALIZATION")
    print("="*80 + "\n")

    # Load results
    results = load_mmlu_results()

    if results:
        # Print summary
        print_summary(results)

        # Categorize models
        small_models, big_models = categorize_models(results)
        print(f"\n[MODEL CATEGORIES]")
        print(f"   Small models (<2B): {len(set(r['model_name'] for r in small_models))}")
        print(f"   Big models (>=2B):   {len(set(r['model_name'] for r in big_models))}")

        # Create visualizations
        print("\n[CREATING VISUALIZATIONS]")

        # 1. Overall scores - 3 graphs (all, small, big)
        print("\n   Generating overall score comparisons...")
        plot_overall_scores(results, suffix="_all", category_label="")
        if small_models:
            plot_overall_scores(small_models, suffix="_small", category_label=" - Small Models (<2B)")
        if big_models:
            plot_overall_scores(big_models, suffix="_big", category_label=" - Big Models (≥2B)")

        # 2. Task-specific scores - 3 graphs (all, small, big)
        print("\n   Generating task-specific performance comparisons...")
        plot_task_specific_scores(results, suffix="_all", category_label="")
        if small_models:
            plot_task_specific_scores(small_models, suffix="_small", category_label=" - Small Models (<2B)")
        if big_models:
            plot_task_specific_scores(big_models, suffix="_big", category_label=" - Big Models (≥2B)")

        # 3. Individual model radar charts
        print("\n   Generating individual model radar charts...")
        plot_individual_model_radar(results)

        print("\n[COMPLETE] All visualizations completed!")
        print(f"   Check the ./MMLU/ directory for all generated graphs.")

    else:
        print("\n[ERROR] No results found to visualize!")
        print("Make sure you have run test_single_model_mmlu.py first to generate results.")
