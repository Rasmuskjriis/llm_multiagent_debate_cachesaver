import pandas as pd
import asyncio
import argparse

from maths.gen_math import main as gen_math_main
from gsm.eval_gsm import main as eval_gsm_main
from gsm.gen_gsm import main as gen_gsm_main
from biography.gen_conversation import main as gen_conversation_main
from biography.eval_conversation import main as eval_conversation_main
from mmlu.gen_mmlu import main as gen_mmlu_main
from mmlu.eval_mmlu import main as eval_mmlu_main

from utils.utils import calc_mean_sem_ci, tokens_to_cost, clear_cache, sanitize_model_name
import time

def make_result_row(agents, rounds, eval_rounds, model, result, runtime):
        
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

async def param_optimization_gen_math(max_agents, max_rounds, model, problems, results_df):

    all_permutations = []

    for i in range(1, max_rounds+1):
        for j in range(1, max_agents+1):
            all_permutations.append([j,i])

    async def run_gen_math_experiment(results_df, use_cachesaver):
        for permutation in all_permutations:
            agents = permutation[0]
            rounds = permutation[1]
            
            runtime = time.time()
            result_nc = await gen_math_main(agents=agents, rounds=rounds, problems=problems, model=model, use_cachesaver=use_cachesaver)
            runtime = time.time() - runtime
            result_row = make_result_row(agents=agents, 
                                        rounds=rounds, 
                                        eval_rounds=problems, 
                                        model=model, 
                                        result=result_nc,
                                        runtime=runtime
                                        )
            

            results_df[f"gen_math {"w/ cs" if use_cachesaver else ""} {agents} {rounds}"] = results_df.index.map(result_row)
        return results_df

    results_df = await run_gen_math_experiment(results_df, use_cachesaver=False)
    results_df = await run_gen_math_experiment(results_df, use_cachesaver=True)

    return results_df


async def main(model, problems):
    #clear_cache()

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

    results_df = await param_optimization_gen_math(max_agents=1, max_rounds=1, model=model, problems=1, results_df=results_df)

    print(results_df)
    results_df.to_excel(f"experiment/param_optimization_results/{sanitize_model_name(model)}_param_optimization_{problems}.xlsx", index=True)
    

if __name__ == "__main__":

    paser = argparse.ArgumentParser()

    paser.add_argument("-m", "--model", required=True)
    paser.add_argument("-p", "--problem", required=True)

    args = paser.parse_args()

    asyncio.run(main(model=args.model, problems=args.problem))

    


    
