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
        #"model": model,
        "API calls" : result["api_calls"],
        "accuracy": round(result["mean"], 2),
        "runtime (s)": runtime,
        #"standard error": result["sem"],
        #"confidence interval": (round(float(result["ci"][0]), 3), round(float(result["ci"][1]), 3)),
        "input_tokens_used": result["prompt_tokens_used"],
        "input_tokens_saved": result["prompt_tokens_saved"],
        #"input_cost ($)": input_cost_used,
        #"input_cost_saved ($)" : input_cost_saved,
        "output_tokens_used": result["completion_tokens_used"],
        "output_tokens_saved": result["completion_tokens_saved"],
        #"output_cost ($)": output_cost_used,
        #"output_cost_saved ($)" : output_cost_saved,
        "cost_paid ($)": total_cost_used,
        "cost_saved ($)" : total_cost_saved,
        "cost_paid_w/o_cs ($)": total_cost_used + total_cost_saved,
        }

async def run_gen_math_experiment(model, size_of_experiment, results_df):
    agents = 2
    rounds = 3
    eval_rounds = size_of_experiment

    runtime = time.time()
    nc_res = await gen_math_main(agents=agents, rounds=rounds, problems=eval_rounds, model=model, use_cachesaver=False)
    runtime = time.time() - runtime
    nc_row = make_result_row(agents, rounds, eval_rounds, model, nc_res, runtime)
    print(f"gen_math cost paid: {nc_row["cost_paid ($)"]}")

    runtime = time.time()
    c_res = await gen_math_main(agents=agents, rounds=rounds, problems=eval_rounds, model=model, use_cachesaver=True)
    runtime = time.time() - runtime
    c_row = make_result_row(agents, rounds, eval_rounds, model, c_res, runtime)
    print(f"gen_math cost paid with CacheSaver: {c_row["cost_paid ($)"]}")

    results_df["basic math"] = results_df.index.map(nc_row)
    results_df["basic math w/ CS"] = results_df.index.map(c_row)

    return results_df

async def run_gsm_experiment(model, size_of_experiment, results_df):
    agents = 3
    rounds = 2
    problems = size_of_experiment

    runtime = time.time()
    nc_file_name, nc_metrics = await gen_gsm_main(agents=agents, rounds=rounds, problems=problems, model=model, use_cachesaver=False)    
    nc_eval = await eval_gsm_main(file=nc_file_name)
    runtime = time.time() - runtime
    nc_res = nc_eval | nc_metrics
    nc_row = make_result_row(agents, rounds, problems, model, nc_res, runtime)
    print("GSM cost paid:", nc_row["cost_paid ($)"])

    runtime = time.time()
    c_file_name, c_metrics = await gen_gsm_main(agents=agents, rounds=rounds, problems=problems, model=model, use_cachesaver=True)
    c_eval = await eval_gsm_main(file=c_file_name)
    runtime = time.time() - runtime
    c_res = c_eval | c_metrics
    c_row = make_result_row(agents, rounds, problems, model, c_res, runtime)
    print("GSM cost paid with CacheSaver:", c_row["cost_paid ($)"])

    results_df["grade school math"] = results_df.index.map(nc_row)
    results_df["grade school math w/ CS"] = results_df.index.map(c_row)

    return results_df

async def run_biography_experiment(model, size_of_experiment, results_df):
    agents = 3
    rounds = 2
    problems = size_of_experiment

    runtime = time.time()
    nc_file_name, nc_metrics = await gen_conversation_main(agents=agents, rounds=rounds, problems=problems, model=model, use_cachesaver=False)    
    nc_eval = await eval_conversation_main(file=nc_file_name, model=model, use_cachesaver=False)
    runtime = time.time() - runtime

    nc_metrics["prompt_tokens_used"] += nc_eval["prompt_tokens_used"]
    nc_metrics["prompt_tokens_saved"] += nc_eval["prompt_tokens_saved"]
    nc_metrics["completion_tokens_used"] += nc_eval["completion_tokens_used"]
    nc_metrics["completion_tokens_saved"] += nc_eval["completion_tokens_saved"]
    nc_metrics["api_calls"] += nc_eval["api_calls"]

    nc_res = nc_eval | nc_metrics
    nc_row = make_result_row(agents, rounds, problems, model, nc_res, runtime)
    print("Biography cost paid:", nc_row["cost_paid ($)"])

    runtime = time.time()
    c_file_name, c_metrics = await gen_conversation_main(agents=agents, rounds=rounds, problems=problems, model=model, use_cachesaver=True)
    c_eval = await eval_conversation_main(file=c_file_name, model=model, use_cachesaver=True)
    runtime = time.time() - runtime

    c_metrics["prompt_tokens_used"] += c_eval["prompt_tokens_used"]
    c_metrics["prompt_tokens_saved"] += c_eval["prompt_tokens_saved"]
    c_metrics["completion_tokens_used"] += c_eval["completion_tokens_used"]
    c_metrics["completion_tokens_saved"] += c_eval["completion_tokens_saved"]
    c_metrics["api_calls"] += c_eval["api_calls"]

    c_res = c_eval | c_metrics
    c_row = make_result_row(agents, rounds, problems, model, c_res, runtime)
    print("Biography cost paid with CacheSaver:", c_row["cost_paid ($)"])

    results_df["biography"] = results_df.index.map(nc_row)
    results_df["biography w/ CS"] = results_df.index.map(c_row)

    return results_df

async def run_mmlu_experiment(model, size_of_experiment, results_df):
    agents = 3
    rounds = 2
    problems = size_of_experiment

    runtime = time.time()
    nc_file_name, nc_metrics = await gen_mmlu_main(agents=agents, rounds=rounds, problems=problems, model=model, use_cachesaver=False)    
    nc_eval = await eval_mmlu_main(file=nc_file_name)  
    runtime = time.time() - runtime
    nc_res = nc_eval | nc_metrics
    nc_row = make_result_row(agents, rounds, problems, model, nc_res, runtime)
    print("MMLU cost paid:", nc_row["cost_paid ($)"])

    runtime = time.time()
    c_file_name, c_metrics = await gen_mmlu_main(agents=agents, rounds=rounds, problems=problems, model=model, use_cachesaver=True)
    c_eval = await eval_mmlu_main(file=c_file_name)
    runtime = time.time() - runtime
    c_res = c_eval | c_metrics
    c_row = make_result_row(agents, rounds, problems, model, c_res, runtime)
    print("MMLU cost paid with CacheSaver:", c_row["cost_paid ($)"])

    results_df["mmlu"] = results_df.index.map(nc_row)
    results_df["mmlu w/ CS"] = results_df.index.map(c_row)

    return results_df

async def main(model, size_of_experiment):
    clear_cache()

    results_df = pd.DataFrame()

    results_df.index = ["agents", 
                        "rounds", 
                        "problems", 
                        #"model", 
                        "API calls", 
                        "accuracy",
                        "runtime (s)", 
                        #"standard error", 
                        #"confidence interval",
                        "input_tokens_used", 
                        "input_tokens_saved", 
                        #"input_cost ($)", 
                        #"input_cost_saved ($)", 
                        "output_tokens_used", 
                        "output_tokens_saved", 
                        #"output_cost ($)", 
                        #"output_cost_saved ($)", 
                        "cost_paid ($)", 
                        "cost_saved ($)",
                        "cost_paid_w/o_cs ($)"
                        ]
    
    results_df = await run_gen_math_experiment(model, size_of_experiment, results_df)
    results_df = await run_gsm_experiment(model, size_of_experiment, results_df)
    results_df = await run_biography_experiment(model, size_of_experiment, results_df)
    results_df = await run_mmlu_experiment(model, size_of_experiment, results_df)

    # results_df = results_df.T
    print(results_df)
    results_df.to_excel(f"experiment/{sanitize_model_name(model)}_experiment_{size_of_experiment}.xlsx", index=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-p", "--problems", type=int, default=100)
    args = parser.parse_args()

    asyncio.run(main(args.model, args.problems))



