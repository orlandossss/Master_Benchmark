"""
MMLU Baseline Test - Always Answer A
This script runs MMLU benchmark with a dummy model that always answers "A"
to establish a baseline score for comparison.
"""

from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
import pandas as pd
import json
import os
from datetime import datetime


class DummyModelAlwaysA:
    """Dummy model that always answers 'A' for baseline testing"""

    def __init__(self, model_name: str = "baseline_always_A"):
        self.model_name = model_name
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        """Always returns 'A' regardless of prompt"""
        self.call_count += 1

        # Show first 3 prompts for debugging
        if self.call_count <= 3:
            print(f"\n{'='*60}")
            print(f"PROMPT #{self.call_count}")
            print(f"{'='*60}")
            print(prompt[:300])
            if len(prompt) > 300:
                print("... (truncated)")
            print(f"\n-> ALWAYS ANSWERING: A")
            print(f"{'='*60}\n")

        return "A"


def run_baseline_mmlu_test():
    """Run MMLU benchmark with always-A baseline model"""

    model_name = "baseline_always_A"
    output_dir = "./MMLU"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("MMLU BASELINE TEST - ALWAYS ANSWER A")
    print("="*60)
    print(f"\nModel: {model_name}")
    print("Strategy: Always answer 'A' regardless of question")

    # Define tasks (same as main test)
    tasks = [
        MMLUTask.FORMAL_LOGIC,
        MMLUTask.GLOBAL_FACTS,
        MMLUTask.COLLEGE_COMPUTER_SCIENCE,
        MMLUTask.COLLEGE_MATHEMATICS,
        MMLUTask.MARKETING,
        MMLUTask.HIGH_SCHOOL_MACROECONOMICS
    ]

    print(f"\nTesting {len(tasks)} MMLU categories:")
    for task in tasks:
        task_name = task.value if hasattr(task, 'value') else str(task)
        print(f"   - {task_name}")
    print("-"*60)

    try:
        # Initialize dummy model
        print(f"\nInitializing baseline model...")
        model = DummyModelAlwaysA(model_name)

        # Create benchmark
        benchmark = MMLU(
            tasks=tasks,
            n_shots=3
        )

        # Run evaluation
        print("\nRunning baseline evaluation (showing first 3 questions)...")
        benchmark.evaluate(model=model)

        # Store results
        task_names = [task.value if hasattr(task, 'value') else str(task) for task in tasks]

        # Calculate per-task scores from predictions
        task_scores = {}
        if hasattr(benchmark, 'predictions') and benchmark.predictions is not None:
            predictions_df = benchmark.predictions if isinstance(benchmark.predictions, pd.DataFrame) else pd.DataFrame(benchmark.predictions)

            # Show first 3 predictions
            print(f"\n{'='*60}")
            print("FIRST 3 QUESTIONS - COMPARISON")
            print(f"{'='*60}")
            for idx in range(min(3, len(predictions_df))):
                row = predictions_df.iloc[idx]
                print(f"\nQuestion #{idx + 1}:")
                print(f"  Task: {row.get('Task', 'N/A')}")
                print(f"  Expected: {row.get('Expected Output', 'N/A')}")
                print(f"  Got: {row.get('Actual Output', 'N/A')}")
                print(f"  Correct: {'YES' if row.get('Correct', False) else 'NO'}")
                print("-" * 60)

            # Calculate scores per task
            for task_name in task_names:
                task_preds = predictions_df[predictions_df['Task'] == task_name]
                if len(task_preds) > 0:
                    score = task_preds['Correct'].mean()
                    task_scores[task_name] = score

        # Prepare result
        result = {
            'model_name': model_name,
            'overall_score': benchmark.overall_score,
            'tasks': task_names,
            'task_scores': task_scores,
            'num_tasks': len(tasks),
            'n_shots': 3,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'success',
            'strategy': 'always_answer_A',
            'total_questions': model.call_count
        }

        # Print per-task scores
        if task_scores:
            print(f"\n{'='*60}")
            print("PER-TASK SCORES")
            print(f"{'='*60}")
            for task_name, score in task_scores.items():
                print(f"{task_name:35s}: {score:.4f} ({score*100:.2f}%)")

        print(f"\n{'='*60}")
        print(f"OVERALL BASELINE SCORE: {benchmark.overall_score:.4f} ({benchmark.overall_score*100:.2f}%)")
        print(f"Total questions answered: {model.call_count}")
        print(f"{'='*60}")

        # Save results
        save_baseline_results(result, output_dir)

        # Print analysis
        print("\n" + "="*60)
        print("BASELINE ANALYSIS")
        print("="*60)
        print(f"\nExpected score for random 4-choice guessing: 0.2500 (25.00%)")
        print(f"Actual baseline score (always A): {benchmark.overall_score:.4f} ({benchmark.overall_score*100:.2f}%)")

        if benchmark.overall_score > 0.25:
            diff = (benchmark.overall_score - 0.25) * 100
            print(f"Baseline is {diff:.2f}% BETTER than random (A is more common answer)")
        elif benchmark.overall_score < 0.25:
            diff = (0.25 - benchmark.overall_score) * 100
            print(f"Baseline is {diff:.2f}% WORSE than random (A is less common answer)")
        else:
            print("Baseline equals random guessing (A appears 25% of the time)")

        print("\nThis baseline helps interpret real model scores:")
        print("- Any model scoring below this baseline is worse than always guessing A")
        print("- Good models should score significantly above this baseline")
        print("="*60)

        return result

    except Exception as e:
        print(f"\n[ERROR] Baseline test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_baseline_results(result, output_dir):
    """Save baseline results to JSON and CSV files"""

    # Save to JSON
    json_path = os.path.join(output_dir, 'baseline_always_A_MMLU.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] Results saved to: {json_path}")

    # Save to CSV
    csv_row = {
        'model_name': result['model_name'],
        'overall_score': result['overall_score'],
        'tasks': ', '.join(result['tasks']),
        'num_tasks': result['num_tasks'],
        'n_shots': result['n_shots'],
        'timestamp': result['timestamp'],
        'status': result['status'],
        'strategy': result.get('strategy', 'N/A'),
        'total_questions': result.get('total_questions', 0)
    }

    # Add per-task scores
    task_scores = result.get('task_scores', {})
    for task_name, score in task_scores.items():
        csv_row[f'score_{task_name}'] = score

    import pandas as pd
    df = pd.DataFrame([csv_row])
    csv_path = os.path.join(output_dir, 'baseline_always_A_MMLU.csv')
    df.to_csv(csv_path, index=False)
    print(f"[SAVED] Results saved to: {csv_path}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("STARTING MMLU BASELINE TEST")
    print("="*60)
    print("\nThis test establishes a baseline by always answering 'A'")
    print("It helps determine if answer distribution is balanced.\n")

    result = run_baseline_mmlu_test()

    if result:
        print("\n[COMPLETE] Baseline test completed successfully!")
        print(f"Check ./MMLU/baseline_always_A_MMLU.json for full results")
    else:
        print("\n[FAILED] Baseline test failed!")
