import pandas as pd
import asyncio
import argparse

from maths.gen_math import main as gen_math_main
from utils.utils import calc_mean_sem_ci, tokens_to_cost, clear_cache

def make_result_row(agents, rounds, eval_rounds, model, result):
        return {
            "agents": agents,
            "rounds": rounds,
            "eval_rounds": eval_rounds,
            "model": model,
            "api_calls" : result["api_calls"],
            "accuracy": round(result["mean"], 2),
            "standard error": result["sem"],
            "confidence interval": (float(result["ci"][0]), float(result["ci"][1])),
            "input_tokens_used": result["prompt_tokens_used"],
            "input_tokens_saved": result["prompt_tokens_saved"],
            "input_cost ($)": result["prompt_cost_used"],
            "input_cost_saved ($)" : result["prompt_cost_saved"],
            "output_tokens_used": result["completion_tokens_used"],
            "output_tokens_saved": result["completion_tokens_saved"],
            "output_cost ($)": result["completion_cost_used"],
            "output_cost_saved ($)" : result["completion_cost_saved"],
            "cost ($)": result["total_cost_used"],
            "cost_saved ($)" : result["total_cost_saved"]}

async def run_gen_math_experiment(model, size_of_experiment, results_df):
    agents = 2
    rounds = 3
    eval_rounds = int(100 * size_of_experiment)

    c_res = await gen_math_main(agents=agents, rounds=rounds, problems=eval_rounds, model=model, use_cachesaver=True)
    nc_res = await gen_math_main(agents=agents, rounds=rounds, problems=eval_rounds, model=model, use_cachesaver=False)

    c_row = make_result_row(agents, rounds, eval_rounds, model, c_res)
    nc_row = make_result_row(agents, rounds, eval_rounds, model, nc_res)

    results_df["gen_math"] = results_df.index.map(nc_row)
    results_df["gen_math w. CacheSaver"] = results_df.index.map(c_row)

    return results_df

async def main(model, size_of_experiment):
    clear_cache()

    results_df = pd.DataFrame()

    results_df.index = ["agents", 
                        "rounds", 
                        "eval_rounds", 
                        "model", 
                        "api_calls", 
                        "accuracy", 
                        "standard error", 
                        "confidence interval",
                        "input_tokens_used", 
                        "input_tokens_saved", 
                        "input_ cost ($)", 
                        "input_cost_saved ($)", 
                        "output_tokens_used", 
                        "output_tokens_saved", 
                        "output_cost ($)", 
                        "output_cost_saved ($)", 
                        "cost ($)", 
                        "cost_saved ($)"]
    
    results_df = await run_gen_math_experiment(model, size_of_experiment, results_df)

    results_df.to_excel(f"experiment/{model}_Experiment.xlsx", index=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-s", "--size", type=float, required=True)
    args = parser.parse_args()

    asyncio.run(main(args.model, args.size))



