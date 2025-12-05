import ollama
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
import pandas as pd
import re
import json
import os
from datetime import datetime

class OllamaModel:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        

        response = ollama.generate(model=self.model_name, prompt=prompt)
        answer = response["response"].strip()

        # Extract only the letter choice (A, B, C, or D)
        # Look for first occurrence of A, B, C, or D
        match = re.search(r'\b[A-D]\b', answer, re.IGNORECASE)
        if match:
            return match.group(0).upper()

        # Fallback: check if answer starts with a letter
        if answer and answer[0].upper() in ['A', 'B', 'C', 'D']:
            return answer[0].upper()

        # Last resort: return the full answer (will likely be wrong)
        return answer


def run_mmlu_single_model(model_name):
    """Run MMLU benchmark for a single model"""

    # Create output directory
    output_dir = "./MMLU"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("MMLU BENCHMARK - SINGLE MODEL")
    print("="*60)
    print(f"\nTesting model: {model_name}")

    # Define tasks
    tasks = [
        MMLUTask.FORMAL_LOGIC,
        MMLUTask.GLOBAL_FACTS,
        MMLUTask.COLLEGE_COMPUTER_SCIENCE,
        MMLUTask.COLLEGE_MATHEMATICS,
        MMLUTask.MARKETING,
        MMLUTask.HIGH_SCHOOL_MACROECONOMICS
    ]

    print(f"Tasks: {len(tasks)} MMLU categories")
    for task in tasks:
        task_name = task.value if hasattr(task, 'value') else str(task)
        print(f"   - {task_name}")
    print("-"*60)

    try:
        # Initialize model
        model = OllamaModel(model_name)

        benchmark = MMLU(
            tasks=tasks,
            n_shots=3
        )

        benchmark.evaluate(model=model)

        # Store results
        task_names = [task.value if hasattr(task, 'value') else str(task) for task in tasks]

        # Calculate per-task scores from predictions
        task_scores = {}
        if hasattr(benchmark, 'predictions') and benchmark.predictions is not None:
            predictions_df = benchmark.predictions if isinstance(benchmark.predictions, pd.DataFrame) else pd.DataFrame(benchmark.predictions)

            for task_name in task_names:
                task_preds = predictions_df[predictions_df['Task'] == task_name]
                if len(task_preds) > 0:
                    task_scores[task_name] = task_preds['Correct'].mean()

        result = {
            'model_name': model_name,
            'overall_score': benchmark.overall_score,
            'tasks': task_names,
            'task_scores': task_scores,
            'num_tasks': len(tasks),
            'n_shots': 3,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'success',
            'predictions': benchmark.predictions
        }

        # Print per-task scores
        if task_scores:
            print(f"\n📊 Per-Task Scores:")
            for task_name, score in task_scores.items():
                print(f"   {task_name:35s}: {score:.4f} ({score*100:.2f}%)")

        print(f"\n✅ Overall Score: {benchmark.overall_score:.4f} ({benchmark.overall_score*100:.2f}%)")

        # Save results
        save_results(result, output_dir, model_name)

        return result

    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        tasks = [
            MMLUTask.FORMAL_LOGIC,
            MMLUTask.GLOBAL_FACTS,
            MMLUTask.COLLEGE_COMPUTER_SCIENCE,
            MMLUTask.COLLEGE_MATHEMATICS,
            MMLUTask.MARKETING,
            MMLUTask.HIGH_SCHOOL_MACROECONOMICS
        ]
        task_names = [task.value if hasattr(task, 'value') else str(task) for task in tasks]
        result = {
            'model_name': model_name,
            'overall_score': 0.0,
            'tasks': task_names,
            'task_scores': {},
            'num_tasks': len(tasks),
            'n_shots': 3,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'failed',
            'error': str(e),
            'predictions': []
        }
        save_results(result, output_dir, model_name)
        return result


def save_results(result, output_dir, model_name):
    """Save results to JSON and CSV files"""

    # Clean model name for filename (replace special characters)
    clean_model_name = model_name.replace(':', '_').replace('/', '_')

    # Create a copy without predictions (only keep scores)
    result_copy = {
        'model_name': result['model_name'],
        'overall_score': result['overall_score'],
        'tasks': result['tasks'],
        'task_scores': result.get('task_scores', {}),
        'num_tasks': result.get('num_tasks', len(result['tasks'])),
        'n_shots': result['n_shots'],
        'timestamp': result['timestamp'],
        'status': result['status']
    }

    # Include error if present
    if 'error' in result:
        result_copy['error'] = result['error']

    # Save to JSON (without detailed predictions, only scores)
    json_path = os.path.join(output_dir, f'{clean_model_name}_MMLU.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_copy, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to: {json_path}")

    # Prepare data for CSV (without detailed predictions)
    csv_row = {
        'model_name': result['model_name'],
        'overall_score': result['overall_score'],
        'tasks': ', '.join(result['tasks']) if isinstance(result['tasks'], list) else result['tasks'],
        'num_tasks': result.get('num_tasks', len(result['tasks']) if isinstance(result['tasks'], list) else 1),
        'n_shots': result['n_shots'],
        'timestamp': result['timestamp'],
        'status': result['status']
    }

    # Add per-task scores as separate columns
    task_scores = result.get('task_scores', {})
    for task_name, score in task_scores.items():
        csv_row[f'score_{task_name}'] = score

    if 'error' in result:
        csv_row['error'] = result['error']

    # Save to CSV
    df = pd.DataFrame([csv_row])
    csv_path = os.path.join(output_dir, f'{clean_model_name}_MMLU.csv')
    df.to_csv(csv_path, index=False)
    print(f"💾 Results saved to: {csv_path}")


def get_available_models():
    """Get all available models from Ollama"""
    try:
        models_list = ollama.list()
        # Extract model names, excluding embedding models
        model_names = [model['model'] for model in models_list['models']
                      if 'embed' not in model['model'].lower()]
        return sorted(model_names)
    except Exception as e:
        print(f"❌ Error getting model list: {e}")
        return []


def run_mmlu_for_all_models():
    """Run MMLU benchmark for all available Ollama models"""

    print("\n" + "="*60)
    print("MMLU BENCHMARK - ALL MODELS")
    print("="*60)

    # Show tasks being tested
    tasks = [
        MMLUTask.FORMAL_LOGIC,
        MMLUTask.GLOBAL_FACTS,
        MMLUTask.COLLEGE_COMPUTER_SCIENCE,
        MMLUTask.COLLEGE_MATHEMATICS,
        MMLUTask.MARKETING,
        MMLUTask.HIGH_SCHOOL_MACROECONOMICS
    ]

    print(f"\n📚 Testing {len(tasks)} MMLU task categories:")
    for task in tasks:
        task_name = task.value if hasattr(task, 'value') else str(task)
        print(f"   - {task_name}")

    # Get all available models
    models = get_available_models()

    if not models:
        print("❌ No models found in Ollama!")
        return []

    print(f"\n✅ Found {len(models)} model(s) to test:")
    for model in models:
        print(f"   - {model}")
    print()

    # Store all results for summary
    all_results = []

    # Test each model
    for idx, model_name in enumerate(models, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(models)}] Testing: {model_name}")
        print('='*60)

        result = run_mmlu_single_model(model_name)
        all_results.append(result)

        # Print immediate result
        if result['status'] == 'success':
            print(f"✅ {model_name}: Score = {result['overall_score']:.4f}")
        else:
            print(f"❌ {model_name}: Failed - {result.get('error', 'Unknown error')}")

    # Print final summary
    print_summary(all_results)

    return all_results


def print_summary(results):
    """Print summary of all benchmark results"""

    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    # Filter successful results
    successful_results = [r for r in results if r['status'] == 'success']
    failed_results = [r for r in results if r['status'] == 'failed']

    print(f"\n📊 Total models tested: {len(results)}")
    print(f"   ✅ Successful: {len(successful_results)}")
    print(f"   ❌ Failed: {len(failed_results)}")

    if successful_results:
        # Sort by score
        sorted_results = sorted(successful_results, key=lambda x: x['overall_score'], reverse=True)

        print("\n🏆 RANKINGS:")
        print("-"*60)
        for i, result in enumerate(sorted_results, 1):
            score_pct = result['overall_score'] * 100
            print(f"{i:2d}. {result['model_name']:30s} - {result['overall_score']:.4f} ({score_pct:.2f}%)")

        # Statistics
        scores = [r['overall_score'] for r in successful_results]
        print("\n📈 STATISTICS:")
        print("-"*60)
        print(f"Mean score:   {sum(scores)/len(scores):.4f}")
        print(f"Median score: {sorted(scores)[len(scores)//2]:.4f}")
        print(f"Best score:   {max(scores):.4f}")
        print(f"Worst score:  {min(scores):.4f}")

    if failed_results:
        print("\n⚠️  FAILED MODELS:")
        print("-"*60)
        for result in failed_results:
            print(f"   - {result['model_name']}: {result.get('error', 'Unknown error')}")

    print("\n" + "="*60)


if __name__ == "__main__":
    # Run benchmark for all models
    print('starting')
    run_mmlu_for_all_models()
