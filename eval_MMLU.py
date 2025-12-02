import ollama
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
import pandas as pd
import json
import os
from datetime import datetime


class OllamaModel:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = ollama.generate(model=self.model_name, prompt=prompt)
        return response["response"]


def get_available_models():
    """Get list of all available Ollama models"""
    models_list = ollama.list()
    # Extract model names and filter out embedding models
    model_names = []
    for model in models_list['models']:
        name = model['model']
        # Skip embedding models
        if 'embed' not in name.lower():
            model_names.append(name)
    return sorted(model_names)


def run_mmlu_benchmark():
    """Run MMLU benchmark on all available Ollama models"""

    # Get all available models
    models = get_available_models()
    print(f"\n{'='*60}")
    print(f"MMLU BENCHMARK - MULTI-MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"Found {len(models)} models to test")
    print(f"Models: {', '.join(models)}\n")

    # Configure benchmark
    tasks = [MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY]
    n_shots = 3

    # Store results
    results = []

    # Test each model
    for idx, model_name in enumerate(models, 1):
        print(f"\n[{idx}/{len(models)}] Testing model: {model_name}")
        print(f"{'-'*60}")

        try:
            # Create model instance
            model = OllamaModel(model_name)

            # Create new benchmark instance for this model
            benchmark = MMLU(tasks=tasks, n_shots=n_shots)

            # Run evaluation
            benchmark.evaluate(model=model)

            # Store results
            result = {
                'model_name': model_name,
                'overall_score': benchmark.overall_score,
                'timestamp': datetime.now().isoformat(),
                'tasks': [task.value for task in tasks],
                'n_shots': n_shots,
                'status': 'success'
            }

            print(f"✅ Overall Score: {benchmark.overall_score:.4f}")

        except Exception as e:
            print(f"❌ Error testing {model_name}: {str(e)}")
            result = {
                'model_name': model_name,
                'overall_score': None,
                'timestamp': datetime.now().isoformat(),
                'tasks': [task.value for task in tasks],
                'n_shots': n_shots,
                'status': 'failed',
                'error': str(e)
            }

        results.append(result)

    # Save results
    save_results(results)

    # Print summary
    print_summary(results)

    return results


def save_results(results):
    """Save results to JSON and CSV files"""

    # Create output directory if it doesn't exist
    output_dir = './mmlu_results'
    os.makedirs(output_dir, exist_ok=True)

    # Save to JSON
    json_path = os.path.join(output_dir, 'mmlu_benchmark_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to JSON: {json_path}")

    # Save to CSV
    csv_path = os.path.join(output_dir, 'mmlu_benchmark_results.csv')
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"💾 Results saved to CSV: {csv_path}")


def print_summary(results):
    """Print summary of all results"""

    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")

    # Filter successful results
    successful_results = [r for r in results if r['status'] == 'success']

    if not successful_results:
        print("❌ No models completed successfully")
        return

    # Sort by score
    successful_results.sort(key=lambda x: x['overall_score'], reverse=True)

    print(f"\nTotal models tested: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(results) - len(successful_results)}")

    # Top performers
    print(f"\n🏆 TOP 10 MODELS:")
    print(f"{'Rank':<6} {'Model':<30} {'Score':<10}")
    print(f"{'-'*60}")
    for i, result in enumerate(successful_results[:10], 1):
        print(f"{i:<6} {result['model_name']:<30} {result['overall_score']:.4f}")

    # Statistics
    scores = [r['overall_score'] for r in successful_results]
    print(f"\n📊 STATISTICS:")
    print(f"Mean Score: {sum(scores)/len(scores):.4f}")
    print(f"Median Score: {sorted(scores)[len(scores)//2]:.4f}")
    print(f"Best Score: {max(scores):.4f}")
    print(f"Worst Score: {min(scores):.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_mmlu_benchmark()