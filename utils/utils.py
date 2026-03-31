import numpy as np
import scipy

# Should be move to a util folder
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
        # LLama-4
        "meta-llama/llama-4-scout-17b-16e-instruct" : {"prompt_price" : 0.11, "completion_price" : 0.34},

        # Qwen
        "qwen/qwen3-32b" : {"prompt_price" : 0.29, "completion_price" : 0.59}
    }

    input_cost = catalogue[model]["prompt_price"] * prompt_tokens / 1000000
    output_cost = catalogue[model]["completion_price"] * completion_tokens / 1000000
    total_cost = input_cost + output_cost

    return input_cost, output_cost, total_cost