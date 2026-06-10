"""
This runs module runs optimization experiments as in LLMDebate paper we fork this code from, to see how
much CacheSaver can save when running parameter optimization on this MAS.

To be exact, we ran the following tests:
- Math: We test for 1-4 agents and 2 rounds, 100 problems, with and without CacheSaver
- MMLU: We test for 1-4 agents and 2 rounds, 100 problems, with and without CacheSaver

All while we track the following metrics:
- Accuracy
- Runtime
- API calls
- Tokens used and saved (prompt and completion)
- Cost paid and saved to the inference API

The result are saved in an excel file, and can be found in param_optimization_results/
"""

import pandas as pd
import asyncio
import argparse

from maths.gen_math import main as gen_math_main
from mmlu.gen_mmlu import main as gen_mmlu_main
from mmlu.eval_mmlu import main as eval_mmlu_main

from utils.utils import tokens_to_cost, sanitize_model_name
import time

def make_result_row(agents, rounds, eval_rounds, model, result, runtime):
    """ 
    Helper function that reformats the results of our experiment into
    the format we want for our Dataframe. Also calculates the cost paid 
    and saved based on the tokens used and saved.
    """

    input_cost_used, output_cost_used, total_cost_used = tokens_to_cost(result["prompt_tokens_used"], result["completion_tokens_used"], model)
    input_cost_saved, output_cost_saved, total_cost_saved = tokens_to_cost(result["prompt_tokens_saved"], result["completion_tokens_saved"], model)
        
    runtime = round(runtime, 2)

    return {
        "agents": agents,
        "rounds": rounds,
        "problems": eval_rounds,
        "model": model,
        "API calls" : result["api_calls"],
        "accuracy": round(result["mean"], 2),
        "runtime (s)": runtime,
        "standard error": result["sem"],
        "confidence interval": (round(float(result["ci"][0]), 3), round(float(result["ci"][1]), 3)),
        "input_tokens_used": result["prompt_tokens_used"],
        "input_tokens_saved": result["prompt_tokens_saved"],
        "input_cost ($)": input_cost_used,
        "input_cost_saved ($)" : input_cost_saved,
        "output_tokens_used": result["completion_tokens_used"],
        "output_tokens_saved": result["completion_tokens_saved"],
        "output_cost ($)": output_cost_used,
        "output_cost_saved ($)" : output_cost_saved,
        "cost_paid ($)": total_cost_used,
        "cost_saved ($)" : total_cost_saved,
        "cost_paid_w/o_cs ($)": total_cost_used + total_cost_saved,
        }

async def param_turning_math(max_agents, model, problems, df, use_cachesaver):
    """
    Runs a parameter optimization experiment for number of agents for the math subtask, with and without CacheSaver.
    """
    max_agents = max_agents
    rounds = 2

    for agents in range(1, max_agents+1):
        runtime = time.time()
        result = await gen_math_main(agents=agents, rounds=rounds, problems=problems, model=model, use_cachesaver=use_cachesaver)
        runtime = time.time() - runtime
        result_row = make_result_row(agents=agents, rounds=rounds, eval_rounds=problems, model=model, result=result, runtime=runtime)

        df[f"math {"w/ cs" if use_cachesaver else ""} a:{agents} r:{rounds}"] = df.index.map(result_row)
    return df

async def parameter_optimization_mmlu(max_agents, model, problems, df, use_cachesaver):
    """
    Runs a parameter optimization experiment for number of agents for the MMLU subtask, with and without CacheSaver.
    """
    max_agents = max_agents
    rounds = 2

    for agents in range(1, max_agents+1):
        runtime = time.time()
        filename, gen_result = await gen_mmlu_main(agents=agents, rounds=rounds, problems=problems, model=model, use_cachesaver=use_cachesaver)
        eval_result = await eval_mmlu_main(file=filename)
        result = gen_result | eval_result
        runtime = time.time() - runtime
        result_row = make_result_row(agents=agents, rounds=rounds, eval_rounds=problems, model=model, result=result, runtime=runtime)

        df[f"mmlu {"w/ cs" if use_cachesaver else ""} a:{agents} r:{rounds}"] = df.index.map(result_row)
    return df

async def main(max_agents, model, problems):
    """
    Clears our cache, and then runs all the experiments from this module,
    then saves the result to an excel file.
    """
    #clear_cache()
    experiemnt_file_path = f"experiment/param_optimization_results/agents_param_optimization_{sanitize_model_name(model)}_{problems}.xlsx"

    results_df = pd.DataFrame()

    results_df.index = ["agents", 
                        "rounds", 
                        "problems", 
                        "model", 
                        "API calls", 
                        "accuracy",
                        "runtime (s)", 
                        "standard error", 
                        "confidence interval",
                        "input_tokens_used", 
                        "input_tokens_saved", 
                        "input_cost ($)", 
                        "input_cost_saved ($)", 
                        "output_tokens_used", 
                        "output_tokens_saved", 
                        "output_cost ($)", 
                        "output_cost_saved ($)", 
                        "cost_paid ($)", 
                        "cost_saved ($)",
                        "cost_paid_w/o_cs ($)"
                        ]

    results_df = await param_turning_math(
            max_agents=max_agents, 
            model=model, 
            problems=problems, 
            df=results_df,
            use_cachesaver=False
            )
    
    # Save intermediate results.
    results_df.to_excel(experiemnt_file_path, index=True)

    results_df = await param_turning_math(
            max_agents=max_agents, 
            model=model, 
            problems=problems, 
            df=results_df,
            use_cachesaver=True
            )
    
    # Save intermediate results.
    results_df.to_excel(experiemnt_file_path, index=True) 

    results_df = await parameter_optimization_mmlu(
            max_agents=max_agents, 
            model=model, 
            problems=problems, 
            df=results_df,
            use_cachesaver=False
            )¨
    
    # Save intermediate results.
    results_df.to_excel(experiemnt_file_path, index=True)

    results_df = await parameter_optimization_mmlu(
            max_agents=max_agents, 
            model=model, 
            problems=problems, 
            df=results_df,
            use_cachesaver=True
            )
    
    results_df.to_excel(experiemnt_file_path, index=True)
    print(results_df)
    

if __name__ == "__main__":

    paser = argparse.ArgumentParser()

    paser.add_argument("-m", "--model", required=True)
    paser.add_argument("-p", "--problem", type=int, required=True)

    paser.add_argument("-a", "--max_agents", type=int, required=True)

    args = paser.parse_args()

    asyncio.run(main(max_agents=args.max_agents, model=args.model, problems=args.problem))
