import json
import os
import sys
import time

sys.path.append('../')
sys.path.append('../src')
sys.path.append('../prompts')
sys.path.append('../src/llmperf')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from benchmarking.src.llmperf import llmperf_utils
from benchmarking.src.performance_evaluation import SyntheticPerformanceEvaluator

# Meta-Llama-3.1-70B-Instruct

# low power

# SambaNova Cloud example
model_names = ['Meta-Llama-3.1-70B-Instruct']
llm_api = 'sncloud'

# additional parameters
results_dir = 'data/results/aramco_regular_2/llama_3.1_70b'
# results_dir = 'data/results/aramco_regular/llama_3.1_405b'
# num_concurrent_requests = [1,10,100]
num_concurrent_requests = [10]
timeout = 60000
# num_input_tokens = [100, 1_000, 10_000, 50_000, 100_000] # 70b
# num_input_tokens = [100, 1_000, 10_000] # 405b
num_input_tokens = [100_000]
# num_output_tokens = [100, 1_000] # 70b 405b mixtral
num_output_tokens = [1_000]
sampling_params = {}
user_metadata = {}

ratio = 1

df_all_summary_results = pd.DataFrame()
for model_idx, model_name in enumerate(model_names):
    for input_tokens in num_input_tokens:
        for output_tokens in num_output_tokens:
            for concurrent_requests in num_concurrent_requests:
                # time.sleep(60)
                num_requests = concurrent_requests*ratio
                print(f'running model_name {model_name}, input_tokens {input_tokens}, output_tokens {output_tokens}, concurrent_requests {concurrent_requests}, num_requests {num_requests}')
                user_metadata['model_idx'] = model_idx
                # Instantiate evaluator
                evaluator = SyntheticPerformanceEvaluator(
                    model_name=model_name,
                    results_dir=results_dir,
                    num_concurrent_requests=concurrent_requests,
                    timeout=timeout,
                    user_metadata=user_metadata,
                    llm_api=llm_api,
                )

                # Run performance evaluation
                model_results_summary, model_results_per_request = evaluator.run_benchmark(
                    num_input_tokens=input_tokens,
                    num_output_tokens=output_tokens,
                    num_requests=num_requests,
                    sampling_params=sampling_params,
                )

                flatten_model_results_summary = llmperf_utils.flatten_dict(model_results_summary)
                filtered_flatten_model_results_summary = {
                    key: value for key, value in flatten_model_results_summary.items() if key not in ['model']
                }
                df_model_results_summary = pd.DataFrame.from_dict(
                    filtered_flatten_model_results_summary, orient='index', columns=[flatten_model_results_summary['model']]
                )

                df_all_summary_results = pd.concat([df_all_summary_results, df_model_results_summary], axis=1)