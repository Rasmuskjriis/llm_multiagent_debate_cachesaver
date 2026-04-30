import numpy as np
import scipy
import os

def calc_mean_sem_ci(scores):
    n = len(scores)
    
    mean = np.mean(scores)
    
    if n == 1:
        return mean, 0, 0

    sem = scipy.stats.sem(scores)
    
    # 90% confidence interval
    #p = ppf(0.95)

    # 95% confidence interval
    p = 0.975

    # 99% confidence interval
    #p= ppf(0.995)

    if (n >= 30):
        ci = scipy.stats.norm.ppf(p) * sem
    else:
        ci = scipy.stats.t.ppf(p, df = n-1) * sem

    return mean, sem, ci

def tokens_to_cost(prompt_tokens, completion_tokens, model):
    catalogue = {
        # Groq
        # LLama-4
        "meta-llama/llama-4-scout-17b-16e-instruct" : {"prompt_price" : 0.11, "completion_price" : 0.34},

        # Qwen
        "qwen/qwen3-32b" : {"prompt_price" : 0.29, "completion_price" : 0.59},


        # OpenAI
        # GPT
        "gpt-5-nano-2025-08-07" : {"prompt_price" : 0.05, "completion_price" : 0.4}
    }

    input_cost = catalogue[model]["prompt_price"] * prompt_tokens / 1000000
    output_cost = catalogue[model]["completion_price"] * completion_tokens / 1000000
    total_cost = input_cost + output_cost

    return input_cost, output_cost, total_cost

def clear_cache():
    cache_path = "cache\cache.db"
    try:
        os.remove(cache_path)
    except FileNotFoundError:
        print(f"Could not find cache file at {cache_path}. It may have already been deleted.")
        pass

def sanitize_model_name(model_name):
    new_mn = ""
    for c in model_name:
        if c == '/':
            new_mn += "-"
        else:
            new_mn += str(c)
    return new_mn

def make_random_ns():
    return "ns_" + str(np.random.randint(10000000))