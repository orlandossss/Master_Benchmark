# Project Conversation Appendix

**Project:** Master_Benchmark - Ollama Performance Monitoring Tool
**Purpose:** Track all prompts, responses, and assistance provided throughout the project development
---

## Entry #1
**Date:** 2025-11-19
**Model Used:** Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)

**Prompt:**
> Can you create an appendix file where it store all the prompt I did for this project and a summary in a few sentence of the answers and the model used. Is it possible than from now on each time i asked you something about this project you add it the the appendix

**Response Summary:**
Created this PROJECT_APPENDIX.md file to track all prompts and responses for the Master_Benchmark project. The file will store timestamp, prompt, response summary, and model used. Note: Automatic updates are not possible due to technical limitations, but manual updates will be added for each significant interaction when requested. The system can only track conversations from this point forward, as previous conversation history is not accessible.

---

## How to Use This Appendix

- Each entry includes: date, model used, user prompt, and response summary
- Entries are numbered sequentially
- Manual updates will be made for each significant interaction
- To request an update: simply ask me to add the conversation to the appendix

---

## Entry #2
**Date:** 2025-11-19 (estimated)
**Model Used:** Claude Sonnet (unknown version)

**Prompt:**
> Can you add something such as it doesn't start benchmarking until it is lower than 60 degrees in temperature

**Response Summary:**
Added the `_wait_for_temperature_threshold()` method to the benchmarking tool. The method monitors system temperature before starting benchmarks and waits until it drops below 60°C to ensure consistent testing conditions. This prevents thermal throttling from affecting benchmark results.

---

## Entry #3
**Date:** 2025-11-19
**Model Used:** Claude Sonnet (unknown version)

**Prompt:**
> Can you add a time.perf_counter to compare it to time.time method measurement

**Response Summary:**
Added dual timing measurement using both `time.time()` and `time.perf_counter()` methods for comparison. Updated the code to track: `inference_time_s`, `inference_time_perf_s`, `tokens_per_second`, `tokens_per_second_perf`, and `timing_diff_ms` to show the precision difference between both timing methods. This allows for more accurate performance measurement analysis.

---

## Entry #4
**Date:** 2025-11-19
**Model Used:** Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)

**Prompt:**
> i got a problem, i only have qwen:1.7B install in ollam but still i got this :
>
> (ollama_env) dorian@raspberrypi:~/Desktop/Master_Benchmark $ python3 final_benchmark_code.py
> OLLAMA PERFORMANCE MONITORING TOOL - ENHANCED
> ✅ Loaded 10 questions from questions.txt
> ✅ Loaded 21 models from CSV
> 📋 Found 1 model(s) in Ollama:
>
> qwen3:1.7b
> ✅ Found 1 matching model(s):
> qwen3:0.6b (0,6B)

**Response Summary:**
Fixed a critical bug in the `_detect_matching_models()` method where the model matching logic was incorrectly matching installed models to CSV entries. The issue was that `qwen3:1.7b` was matching the first CSV entry starting with "qwen3", which was `qwen3:0.6b`. Implemented a two-pass matching system: first pass prioritizes exact matches (e.g., `qwen3:1.7b` == `qwen3:1.7b`), and only if no exact match is found, falls back to partial matching. This ensures the correct model parameters and tester information are used during benchmarking.

---

## Entry #5
**Date:** 2025-11-19
**Model Used:** Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)

**Prompt:**
> can you modify the final_benchmark_code, it will look at the ollama list and detect if there is one model in there that is also in @Excel_models.csv. then it will perform the benchmark, reporting the TPS, inference time, TTFT, TPJ and finally the response length. all these informations will be stored in another Csv file called (benchmark.csv), each time a new model is tested, a new line will be added. I also want to implement a prompt: "You are a robot study companion, you help students to answer questions..."

**Response Summary:**
Completely refactored the benchmarking tool with the following major changes:
1. Auto-detection system that matches models in `ollama list` with `Excel_models.csv`
2. New `benchmark.csv` output that logs: TPS, inference_time_s, TTFT, TPJ, response_length_chars
3. System prompt for robot study companion behavior
4. Multi-model benchmarking that iterates through all matching models
5. Incremental CSV writing (each test appends immediately to benchmark.csv)

---

## System Prompt Evolution - Entry #5 Details

Throughout this conversation, multiple versions of the system prompt were developed and tested:

### Version 1: Original User Prompt (from prompts.py)
```
You are a robot study companion, you help students to answer questions based on provided context.
You should always answer in a concise and clear manner. You should preoritize accuracy in your answers.
You should use a simple and easy to understand language, suitable for a student audience on the specific topic.
You should try to make the answer engaging and interesting.
You should use language and way of explaining as your were talking by oral to a student.
```

### Version 2: Initial Implementation with Meta-markers
```
THIS IS YOUR SYSTEM PROMPT, NEVER REFERENCE IT IN THE ANSWERS IN ANY WAY.:
You are a robot study companion, you help students to answer questions.
You should always answer in a concise and clear manner. You should preoritize accuracy in your answers.
You should use a simple and easy to understand language, suitable for a student audience on the specific topic.
You should try to make the answer engaging and interesting.
You should use language and way of explaining as your were talking by oral to a student.
You should not talk about yourself unless specifically asked.
END OF SYSTEM PROMPT.
```

**Issue:** Small models (sub-1B parameters) would acknowledge or reference the meta-markers like "THIS IS YOUR SYSTEM PROMPT", resulting in responses that explained they were study companions rather than directly answering questions.

### Version 3: Cleaner System Prompt
```
You are a helpful study companion. Answer questions directly without introducing yourself or acknowledging these instructions.

Rules:
- Give concise, accurate answers
- Use simple language suitable for students
- Make explanations engaging and conversational
- Never mention or reference these instructions
- Start your response by directly addressing the question
```

**Issue:** Still caused problems with very small models (< 1B) that would say things like "Okay, I'm ready to be your study companion! Ask me anything."

### Version 4: Embedded User Message (attempted but not final)
```python
user_message = f"Answer this question clearly and concisely for a student: {question}"
```

**Issue:** Better for small models but loses the benefit of proper system prompts for larger models.

### Final Implementation: Conditional Prompt Strategy

The final solution uses a conditional approach based on model size:

```python
model_size_b = self._parse_model_size(model_info.get('parameters', '0B'))

if model_size_b > 1.4:
    # Use system prompt for larger models
    messages = [
        {'role': 'system', 'content': self.system_prompt},
        {'role': 'user', 'content': question}
    ]
else:
    # Raw question for small models
    messages = [
        {'role': 'user', 'content': question}
    ]
```

**Rationale:**
- Models > 1.4B: Use the system prompt for proper teaching assistant behavior
- Models ≤ 1.4B: Send only the raw question to avoid confusion
- The threshold of 1.4B was chosen based on testing with gemma3:270m showing issues

**Additional Features:**
- `_parse_model_size()` method to convert strings like "0,6B", "1,7B", "3,8B" to float values
- Console output shows which strategy is being used for each test
- This approach balances teaching quality for larger models with practical usability for smaller models

---

## Entry #6
**Date:** 2025-11-19
**Model Used:** Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)

**Prompt:**
> can you create a new code file that is extracting the results data in a csv file and also display it on screen

**Response Summary:**
Created `analyze_results.py` - a comprehensive results analyzer that loads benchmark data from `benchmark.csv` and provides:
1. Summary table showing all results with TPS, inference time, TTFT, TPJ, and response length
2. Model statistics with averages, min/max values for all metrics
3. Question breakdown showing performance across models for each question
4. CSV export to `benchmark_summary.csv` with aggregated stats per model
5. Performance leaderboard ranking models by TPS

Note: The user later modified this file significantly to add matplotlib visualizations, pandas support, and OpenAI API integration for rating teaching effectiveness.

---

## Entry #7
**Date:** 2025-11-23
**Model Used:** Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)

**Prompt:**
> i want to add a a few new metric to the @benchmarking_pi4.py code file :
>
> read_count, write_count → total I/O operations
> read_bytes, write_bytes → total bytes
>
> From these you can compute:
> IOPS = (read_count + write_count) / duration
> Throughput = bytes / duration (MB/s)
> Average latency-ish = read_time / read_count, etc. (crude but useful)
>
> During each benchmark run, log for the device that holds:
> read_count, write_count → total I/O operations
> read_bytes, write_bytes → total bytes
> (Optionally) read_time, write_time → time spent doing I/O

**Response Summary:**
Added comprehensive disk I/O monitoring to the benchmarking tool. Created two new methods: `_get_disk_io_metrics()` to capture raw I/O counters from psutil, and `_calculate_io_stats()` to compute derived metrics. The tool now tracks:
- Raw metrics: read_count, write_count, read_bytes, write_bytes, read_time, write_time (in milliseconds)
- Derived metrics: IOPS (I/O operations per second), throughput (MB/s), average read/write latency (ms)
- All I/O metrics are captured at baseline and end of each inference run
- Key I/O metrics (IOPS, throughput, read/write counts and bytes) are added to benchmark.csv
- Full I/O statistics are included in the detailed JSON and CSV outputs

---

## Entry #8
**Date:** 2025-11-23
**Model Used:** Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)

**Prompt:**
> now i want my code to run through all the models that are in the excel file and also downloaded in ollama. It will run the benchmarking only if the temperature is respected, once the first model benchmarking is done, the script will wait than the pi cool down to the temperature and when it is cool enough benchmark the next one and repeat

**Response Summary:**
Modified the `run_benchmark()` method to add temperature checks between each model benchmark run. The script now:
- Checks temperature before starting the entire benchmark session (existing behavior)
- After completing all questions for one model, waits for the Raspberry Pi to cool down below 60°C before starting the next model
- Only waits for cooldown if there are more models to test (skips wait after the last model)
- This ensures consistent thermal conditions for each model's benchmark, preventing thermal throttling from affecting results across different models
- The auto-detection system (from Entry #5) already ensures only models present in both Excel_models.csv and Ollama are benchmarked

---

## Entry #9
**Date:** 2025-11-23
**Model Used:** Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)

**Prompt:**
> can you create a new analyze_results_pi4 that is a copy of @analyze_results.py , by adding the news metrics that is in @benchmarking_pi4.py
>
> add this query to the @PROJECT_APPENDIX.md file with a summary of the answer

**Response Summary:**
Created `analyze_results_pi4.py` as an enhanced version of `analyze_results.py` with full support for the I/O metrics from `benchmarking_pi4.py`. Changes include:

**New I/O Metrics Added:**
- `avg_io_iops`: Average I/O operations per second
- `avg_io_throughput_mb_s`: Average I/O throughput in MB/s
- `total_io_read_count` / `total_io_write_count`: Total read/write operation counts
- `total_io_read_mb` / `total_io_write_mb`: Total data read/written in MB
- `avg_io_read_latency_ms` / `avg_io_write_latency_ms`: Average I/O latencies

**Enhancements:**
- Updated `get_summary_statistics()` to extract and aggregate I/O metrics
- Enhanced `print_summary()` to display comprehensive I/O performance section
- Added new `_plot_io_metrics()` visualization method with 4 charts:
  - IOPS comparison bar chart
  - Throughput comparison bar chart
  - Read vs Write operations grouped bar chart
  - Read vs Write latency grouped bar chart
- Updated radar chart to include I/O Performance (7 dimensions total)
- Changed default directories: `./results_pi4`, `./analysis_graphs_pi4`, output files with `_pi4` suffix
- All visualizations and CSV exports now include I/O metrics for comprehensive Pi4 benchmark analysis

---

## Entry #10
**Date:** 2025-11-28
**Model Used:** Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)

**Prompt:**
> i want to modify a bit the @analyse_results_computer.py , @analyze_results_pi4.py to split the grpah into 2 categories :
> - small model (less than 2B)
> - big model ( more than 2B)
>
> add this prompt and the summary of the answers to @PROJECT_APPENDIX.md

**Response Summary:**
Modified both `analyse_results_computer.py` and `analyze_results_pi4.py` to categorize models by size and generate separate graph sets for small models (<2B) and big models (≥2B). Key changes include:

**New Methods:**
- `_parse_model_size()`: Parses model parameter strings (e.g., '1,7B', '3.8B') into float values
- `_categorize_models()`: Splits models into two dictionaries based on 2B threshold

**Enhanced Graph Generation:**
- Modified `generate_graphs()` to create three sets of visualizations:
  1. Small models only (`*_small.png`)
  2. Big models only (`*_big.png`)
  3. All models combined (`*_all.png`)
- All plotting methods now accept a `suffix` parameter to distinguish graph categories
- Graph titles include category labels: "- Small Models (<2B)", "- Big Models (≥2B)", or no label for combined

**Updated Methods:**
- `_plot_tokens_per_second()`: Added suffix parameter and category-specific titles
- `_plot_energy_efficiency()`: Added suffix parameter and category labels
- `_plot_inference_times_by_category()`: Renamed and adapted to work with categorized summaries
- `_plot_response_analysis_by_category()`: Renamed and adapted for category filtering
- `_plot_resource_usage()`: Added suffix parameter and category labels
- `_plot_io_metrics()`: Added suffix parameter and category labels
- `_plot_radar_chart()`: Added suffix parameter and category labels

**Benefits:**
- Better comparison between models of similar sizes
- Prevents small models from being dwarfed by large model metrics in combined graphs
- Maintains backward compatibility with combined "_all" graphs for overall comparison
- Console output shows breakdown: "Small models (<2B): X" and "Big models (≥2B): Y"

---

## Entry #11
**Date:** 2025-11-28
**Model Used:** Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)

**Prompt:**
> [User requested to configure OpenAI API key - Option 3: Update the API key configuration]

**Response Summary:**
Added automatic `.env` file loading to both analysis scripts to securely manage the OpenAI API key configuration.

**Changes Made:**
1. **Modified Scripts:**
   - `analyse_results_computer.py`: Added `python-dotenv` import and `load_dotenv()` call
   - `analyze_results_pi4.py`: Added `python-dotenv` import and `load_dotenv()` call

2. **Configuration Files Created:**
   - `.env.example`: Template file with API key placeholder (safe to commit to git)
   - `.env`: Actual configuration file with API key (gitignored for security)
   - `.gitignore`: Ensures `.env` file is never committed to version control

**How It Works:**
- Scripts now automatically load environment variables from `.env` file on startup
- If `python-dotenv` is not installed, shows a warning but continues (falls back to system env vars)
- API key is read from `OPENAI_API_KEY` environment variable
- `.env` file is protected by `.gitignore` to prevent accidental API key exposure

**Setup Instructions:**
1. Ensure `python-dotenv` is installed: `pip install python-dotenv`
2. Create `.env` file from template: `copy .env.example .env` (Windows) or `cp .env.example .env` (Linux/Mac)
3. Edit `.env` and replace placeholder with actual OpenAI API key from https://platform.openai.com/api-keys
4. Run analysis scripts normally - they will automatically load the API key

**Security Notes:**
- `.env` file is gitignored and will never be committed to repository
- `.env.example` serves as a template without sensitive data
- API keys should never be hardcoded in scripts
- The key is loaded once at script startup and stored in environment variables

---

## Entry #12
**Date:** 2025-11-30
**Model Used:** Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)

**Prompt:**
> can you add the splitting into 2 categories for the performance_vs_teaching_effectiveness and teaching_effectiveness_score into:
> small models (less than 2B)
> big models (more than 2B)
>
> Do this modification for @analyse_results_computer.py and @analyze_results_pi4.py

**Response Summary:**
Modified both `analyse_results_computer.py` and `analyze_results_pi4.py` to split teaching effectiveness visualizations into three graph sets based on model size:

**Refactored `plot_teaching_scores()` method:**
- Categorizes models into small (<2B) and big (≥2B) using existing `_parse_model_size()` method
- Generates 3 graph sets: small models only, big models only, and all models combined
- Added console output showing model count per category

**Created two new helper methods:**
1. `_plot_teaching_bar_chart(models, model_scores, output_path, suffix)`: Generates bar charts of average teaching effectiveness scores with category-specific titles and filenames
2. `_plot_performance_vs_teaching(models, output_path, suffix)`: Generates scatter plots correlating tokens-per-second performance with teaching quality scores

**Generated files (6 graphs per analysis):**
- Small models: `teaching_effectiveness_scores_small.png`, `performance_vs_teaching_small.png`
- Big models: `teaching_effectiveness_scores_big.png`, `performance_vs_teaching_big.png`
- All models: `teaching_effectiveness_scores_all.png`, `performance_vs_teaching_all.png`

Each graph includes appropriate category labels in titles for easy identification.

---

## Entry #13
**Date:** 2025-11-30
**Model Used:** Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)

**Prompt:**
> now i have the teaching effectiveness in @teaching_effectiveness_ratings_pi4.json, can you create a new code that will rank every models in differents categories :
> teaching effectiveness, token per second, token per joule, eneryg consumption, time to first token, average IOPS. Then create a graph that shows those scores and the total ranking of the models, add some potential wieght to each categorie that i can change to make some categories more important than other in the final ranking.
>
> add this to @PROJECT_APPENDIX.md and add the summary of the question

**Response Summary:**
Created `model_ranking_system.py` - a comprehensive multi-criteria model ranking system with configurable weights and advanced visualizations.

**Core Features:**
- **Data Integration**: Loads teaching effectiveness from JSON and performance metrics from CSV
- **Configurable Weights**: Default weights sum to 100% (teaching: 25%, TPS: 20%, TPJ: 15%, energy: 15%, TTFT: 15%, IOPS: 10%)
- **Score Normalization**: Converts all metrics to 0-100 scale, handling "higher is better" (TPS, TPJ, teaching, IOPS) vs "lower is better" (energy, TTFT)
- **Weighted Ranking**: Calculates total scores using configurable category weights

**Key Methods:**
1. `set_weights(**kwargs)`: Update category weights with auto-normalization to sum to 1.0
2. `calculate_rankings()`: Compute normalized scores and weighted totals for all models
3. `print_detailed_rankings()`: Console output of top 10 models per category
4. `export_rankings_csv()`: Export complete rankings with both normalized scores and raw values

**Visualizations (6 graphs):**
1. **overall_rankings.png**: Horizontal bar chart of top 15 models with color-coded scores
2. **category_scores_heatmap.png**: 2D heatmap showing all categories for top 20 models with numerical annotations
3. **top_models_radar_chart.png**: Polar radar chart comparing top 6 models across all 6 dimensions
4. **category_rankings.png**: 2×3 grid showing top 10 models for each individual category
5. **weight_sensitivity.png**: Comparison of how rankings change under 4 different weight scenarios:
   - Current weights
   - Teaching Focus (50% teaching effectiveness)
   - Performance Focus (40% TPS, 25% TTFT)
   - Efficiency Focus (30% TPJ, 30% energy)

**Usage:**
```python
ranker = ModelRankingSystem()
ranker.set_weights(teaching_effectiveness=0.4, tokens_per_second=0.3, ...)
ranker.calculate_rankings()
ranker.export_rankings_csv()
ranker.generate_visualizations()
```

The system provides complete flexibility to adjust priorities based on specific use cases (teaching assistant, high performance, energy efficiency, etc.) while maintaining scientific rigor through normalized scoring.

---

## Entry #14
**Date:** 2025-12-01
**Model Used:** Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)

**Prompt:**
> i got this code that benchmark a model on MMLU, now i want it to test all the model available on my computer through ollama and then give me a graph of their performance on the test
>
> add this question to the @PROJECT_APPENDIX.md with a summary of the answer

**Response Summary:**
Completely refactored `eval_MMLU.py` to create a comprehensive multi-model MMLU benchmarking system with automatic model detection and visualization capabilities.

**Key Features Added:**

1. **MMLUBenchmarkRunner Class**: New class-based architecture for scalable benchmarking
   - `get_available_models()`: Auto-detects all Ollama models (excludes embedding models)
   - `run_benchmark_for_model()`: Tests individual models with error handling
   - `run_all_models()`: Iterates through all available models sequentially
   - `save_results()`: Incremental saving to CSV and JSON files

2. **Data Export**:
   - CSV export: `mmlu_benchmark_results.csv` with model names, overall scores, task-specific scores, and timestamps
   - JSON export: `mmlu_benchmark_results.json` with detailed results including error messages for failed tests
   - Incremental saving after each model test to prevent data loss

3. **Visualizations** (3 graphs generated):
   - **Overall Performance Bar Chart** (`mmlu_overall_performance.png`): Horizontal bar chart with color-coded scores showing all models ranked by MMLU performance
   - **Task-Specific Performance** (`mmlu_task_performance.png`): Grouped bar chart comparing performance across different MMLU tasks (Computer Science, Astronomy)
   - **Distribution & Top 10** (`mmlu_distribution_and_top10.png`): Combined visualization with histogram showing score distribution (mean/median lines) and top 10 models ranking

4. **Enhanced Output**:
   - `print_summary()`: Console summary with statistics (mean, median, best/worst scores, top 5 models)
   - Progress tracking showing [X/Total] during execution
   - Error handling with graceful failure for problematic models

**Testing Scope:**
- Found 21 available models (excluding nomic-embed-text embedding model)
- Tests: HIGH_SCHOOL_COMPUTER_SCIENCE and ASTRONOMY tasks
- 3-shot learning configuration
- Results saved to `./mmlu_results/` directory

**Technical Improvements:**
- Configurable tasks and n_shots parameters in constructor
- Automatic output directory creation
- Sorted model list for consistent testing order
- Color-coded visualizations using viridis and RdYlGn colormaps
- Dynamic figure sizing based on number of models tested

---

## Entry #15
**Date:** 2025-12-01
**Model Used:** Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)

**Prompt:**
> this code is working and iq providing the accuracy correclty, i want to keep this architecture and just add a loop that goes throught all the ollama model downloaded on my computer. All the scores will be reported in a json and csv file that will be saved at the end

**Response Summary:**
Refactored `eval_MMLU.py` to automatically test all Ollama models while preserving the original OllamaModel architecture. The enhanced version includes:

**New Functions:**
1. `get_available_models()`: Auto-detects all Ollama models (filters out embedding models)
2. `run_mmlu_benchmark()`: Main orchestration function that loops through all models
3. `save_results()`: Exports results to both JSON and CSV formats in `./mmlu_results/` directory
4. `print_summary()`: Displays comprehensive summary with rankings and statistics

**Key Features:**
- Loops through all available Ollama models automatically (sorted alphabetically)
- Tests each model with MMLU tasks (HIGH_SCHOOL_COMPUTER_SCIENCE, ASTRONOMY) using 3-shot learning
- Error handling: If a model fails, it's recorded with status 'failed' and error message, then continues to next model
- Progress tracking: Shows [X/Total] during execution
- Results stored with timestamp, overall_score, tasks, n_shots, and status for each model

**Data Export:**
- JSON file: `mmlu_results/mmlu_benchmark_results.json` with detailed results including error messages
- CSV file: `mmlu_results/mmlu_benchmark_results.csv` for easy analysis in spreadsheet tools
- Both files saved after all models complete testing

**Console Output:**
- Initial summary showing total models found
- Per-model progress with scores or error messages
- Final summary table with:
  - Top 10 models ranked by MMLU score
  - Statistics: mean, median, best/worst scores
  - Success/failure counts

**Architecture Preserved:**
- OllamaModel class remains unchanged
- Same generate() method using ollama.generate()
- Compatible with deepeval.benchmarks.MMLU framework

---

## Entry #16
**Date:** 2025-12-02
**Model Used:** Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)

**Prompt:**
> this method is perfectly working, i want you to go through every ollama model available and perform the exact same MMLU test on them; then save all of it in a CSV and json file
>
> add this question to @PROJECT_APPENDIX.md as well as a summary of the answer

**Response Summary:**
Transformed `test_single_model_mmlu.py` from a single-model testing script into a comprehensive multi-model MMLU benchmarking system that automatically tests all available Ollama models.

**Key Changes:**

1. **New Functions Added:**
   - `get_available_models()`: Auto-detects all Ollama models (filters out embedding models), returns sorted list
   - `run_mmlu_for_all_models()`: Main orchestration function that loops through all models and manages the benchmark process
   - `save_results(results, output_dir)`: Exports results to both JSON and CSV formats
   - `print_summary(results)`: Displays comprehensive console summary with rankings and statistics

2. **Architecture Preserved:**
   - `OllamaModel` class remains completely unchanged
   - Same `generate()` method with enhanced prompt formatting
   - Same regex-based answer extraction logic (A-D letter extraction)
   - Compatible with deepeval MMLU framework

3. **Benchmark Configuration:**
   - Task: HIGH_SCHOOL_COMPUTER_SCIENCE
   - 3-shot learning (consistent with previous tests)
   - Sequential testing (one model at a time)

4. **Error Handling:**
   - Try-catch blocks around each model test
   - Failed models recorded with status='failed' and error message
   - Continues testing remaining models if one fails
   - Both successful and failed results saved to output files

5. **Data Export:**
   - **JSON file** (`mmlu_all_models_results.json`): Contains detailed results including all predictions for each model
   - **CSV file** (`mmlu_all_models_results.csv`): Summary results without detailed predictions (model_name, overall_score, task, n_shots, timestamp, status, error)
   - Both files saved in `./mmlu_results/` directory
   - Automatic directory creation if it doesn't exist

6. **Console Output Features:**
   - Initial summary showing total models found
   - Progress tracking: [X/Total] for each model
   - Per-model score display or error message
   - Final summary section with:
     - Top 10 models ranked by MMLU score
     - Statistics: total tested, successful/failed counts, mean/median/best/worst scores

7. **Technical Improvements:**
   - Timestamps added to all results for tracking
   - Status field ('success'/'failed') for easy filtering
   - Sorted model list for consistent testing order
   - Results stored incrementally in memory, saved once at the end

**Usage:**
Simply run `python test_single_model_mmlu.py` and it will automatically detect all models and begin testing them sequentially.

---