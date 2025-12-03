import os
import json
import matplotlib.pyplot as plt
import pandas as pd

def load_mmlu_results(mmlu_dir="./MMLU"):
    """Load all MMLU results from JSON files"""
    results = []

    if not os.path.exists(mmlu_dir):
        print(f"❌ Directory {mmlu_dir} does not exist!")
        return results

    # Find all JSON files
    json_files = [f for f in os.listdir(mmlu_dir) if f.endswith('_MMLU.json')]

    if not json_files:
        print(f"❌ No MMLU JSON files found in {mmlu_dir}")
        return results

    print(f"✅ Found {len(json_files)} result files\n")

    # Load each JSON file
    for json_file in json_files:
        file_path = os.path.join(mmlu_dir, json_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append({
                    'model_name': data.get('model_name', 'Unknown'),
                    'overall_score': data.get('overall_score', 0.0),
                    'task': data.get('task', 'Unknown'),
                    'status': data.get('status', 'Unknown')
                })
                print(f"✓ Loaded: {data.get('model_name')} - Score: {data.get('overall_score', 0.0):.4f}")
        except Exception as e:
            print(f"⚠️  Error loading {json_file}: {e}")

    return results


def plot_mmlu_results(results):
    """Create a bar chart of MMLU results"""

    if not results:
        print("❌ No results to plot!")
        return

    # Sort by score (descending)
    results_sorted = sorted(results, key=lambda x: x['overall_score'], reverse=True)

    # Extract data for plotting
    model_names = [r['model_name'] for r in results_sorted]
    scores = [r['overall_score'] for r in results_sorted]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bar chart
    bars = ax.bar(range(len(model_names)), scores, color='steelblue', edgecolor='navy', linewidth=1.2)

    # Customize the plot
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('MMLU Score', fontsize=12, fontweight='bold')
    ax.set_title('MMLU Benchmark Results - Formal Logic', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on top of bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add a horizontal line for random guessing (25% for 4 choices)
    ax.axhline(y=0.25, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Random Guessing (25%)')
    ax.legend()

    # Tight layout
    plt.tight_layout()

    # Save the plot
    output_path = './MMLU/mmlu_results_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Chart saved to: {output_path}")

    # Show the plot
    plt.show()


def print_summary(results):
    """Print a summary table of results"""

    if not results:
        return

    print("\n" + "="*60)
    print("MMLU RESULTS SUMMARY")
    print("="*60)

    df = pd.DataFrame(results)
    df = df.sort_values('overall_score', ascending=False)

    print("\nModel Rankings:")
    print("-"*60)
    for idx, row in df.iterrows():
        print(f"{df.index.get_loc(idx)+1:2d}. {row['model_name']:25s} - {row['overall_score']:.4f} ({row['overall_score']*100:.2f}%)")

    # Statistics
    if len(results) > 0:
        print("\n📊 Statistics:")
        print("-"*60)
        print(f"Number of models tested: {len(results)}")
        print(f"Best score:  {df['overall_score'].max():.4f} ({df['overall_score'].max()*100:.2f}%)")
        print(f"Worst score: {df['overall_score'].min():.4f} ({df['overall_score'].min()*100:.2f}%)")
        print(f"Mean score:  {df['overall_score'].mean():.4f} ({df['overall_score'].mean()*100:.2f}%)")
        print(f"Median score: {df['overall_score'].median():.4f} ({df['overall_score'].median()*100:.2f}%)")

    print("="*60)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MMLU RESULTS VISUALIZATION")
    print("="*60 + "\n")

    # Load results
    results = load_mmlu_results()

    if results:
        # Print summary
        print_summary(results)

        # Create visualization
        print("\n📊 Creating visualization...")
        plot_mmlu_results(results)
    else:
        print("\n❌ No results found to visualize!")
        print("Make sure you have run test_single_model_mmlu.py first to generate results.")
