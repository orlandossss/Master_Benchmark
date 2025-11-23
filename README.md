# Master_Benchmark
This repository aim to provide the code and the explanation to run a quick benchmark of Language model performance on Raspberry PI4/5/computer

Create a new environment to avoid conflict in between librairies

Requirement : 
-have the excel_models file filed with model name that you want to test

Librairies required :
-ollama
-pandas
-psutils

You can find ollama models on : https://ollama.com/library
-download models you want with ollama pull "name"

every benchmarking are gonna store the results in a separate folder (@benchmarking_pi4 to results_pi4 for exemple)
once you perform the benchmark you can run the analyze_results related to your results
Every graph will be stored in analysys_graph
