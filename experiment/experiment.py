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

def make_result_row(agents, rounds, eval_rounds, model, result):
        
    input_cost_used, output_cost_used, total_cost_used = tokens_to_cost(result["prompt_tokens_used"], result["completion_tokens_used"], model)
    input_cost_saved, output_cost_saved, total_cost_saved = tokens_to_cost(result["prompt_tokens_saved"], result["completion_tokens_saved"], model)
        
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
        "input_cost ($)": input_cost_used,
        "input_cost_saved ($)" : input_cost_saved,
        "output_tokens_used": result["completion_tokens_used"],
        "output_tokens_saved": result["completion_tokens_saved"],
        "output_cost ($)": output_cost_used,
        "output_cost_saved ($)" : output_cost_saved,
        "cost ($)": total_cost_used,
        "cost_saved ($)" : total_cost_saved}

async def run_gen_math_experiment(model, size_of_experiment, results_df):
    agents = 2
    rounds = 3
    eval_rounds = int(100 * size_of_experiment)

    nc_res = await gen_math_main(agents=agents, rounds=rounds, problems=eval_rounds, model=model, use_cachesaver=False)
    nc_row = make_result_row(agents, rounds, eval_rounds, model, nc_res)
    print(f"gen_math cost: {nc_row["cost ($)"]}")

    c_res = await gen_math_main(agents=agents, rounds=rounds, problems=eval_rounds, model=model, use_cachesaver=True)
    c_row = make_result_row(agents, rounds, eval_rounds, model, c_res)
    print(f"gen_math cost with CacheSaver: {c_row["cost ($)"]}")

    results_df["basic math problems"] = results_df.index.map(nc_row)
    results_df["basic math problems w. CacheSaver"] = results_df.index.map(c_row)

    return results_df

async def run_gsm_experiment(model, size_of_experiment, results_df):
    agents = 3
    rounds = 2
    problems = int(100 * size_of_experiment)

    nc_file_name, nc_metrics = await gen_gsm_main(agents=agents, rounds=rounds, problems=problems, model=model, use_cachesaver=False)    
    nc_eval = await eval_gsm_main(file=nc_file_name)
    nc_res = nc_eval | nc_metrics
    nc_row = make_result_row(agents, rounds, problems, model, nc_res)
    print("GSM cost:", nc_row["cost ($)"])

    c_file_name, c_metrics = await gen_gsm_main(agents=agents, rounds=rounds, problems=problems, model=model, use_cachesaver=True)
    c_eval = await eval_gsm_main(file=c_file_name)
    c_res = c_eval | c_metrics
    c_row = make_result_row(agents, rounds, problems, model, c_res)
    print("GSM cost with CacheSaver:", c_row["cost ($)"])

    results_df["grade school math problems"] = results_df.index.map(nc_row)
    results_df["grade school math problems w. CacheSaver"] = results_df.index.map(c_row)

    return results_df

async def run_biography_experiment(model, size_of_experiment, results_df):
    agents = 3
    rounds = 2
    problems = int(100 * size_of_experiment)

    nc_file_name, nc_metrics = await gen_conversation_main(agents=agents, rounds=rounds, problems=problems, model=model, use_cachesaver=False)    
    nc_eval = await eval_conversation_main(file=nc_file_name, model=model, use_cachesaver=False)

    print("API calls - gen: ", nc_metrics["api_calls"])
    print("API calls - eval: ", nc_eval["api_calls"])

    nc_metrics["prompt_tokens_used"] += nc_eval["prompt_tokens_used"]
    nc_metrics["prompt_tokens_saved"] += nc_eval["prompt_tokens_saved"]
    nc_metrics["completion_tokens_used"] += nc_eval["completion_tokens_used"]
    nc_metrics["completion_tokens_saved"] += nc_eval["completion_tokens_saved"]
    nc_metrics["api_calls"] += nc_eval["api_calls"]

    nc_res = nc_eval | nc_metrics
    nc_row = make_result_row(agents, rounds, problems, model, nc_res)
    print("Biography cost:", nc_row["cost ($)"])

    c_file_name, c_metrics = await gen_conversation_main(agents=agents, rounds=rounds, problems=problems, model=model, use_cachesaver=True)
    c_eval = await eval_conversation_main(file=c_file_name, model=model, use_cachesaver=True)

    c_metrics["prompt_tokens_used"] += c_eval["prompt_tokens_used"]
    c_metrics["prompt_tokens_saved"] += c_eval["prompt_tokens_saved"]
    c_metrics["completion_tokens_used"] += c_eval["completion_tokens_used"]
    c_metrics["completion_tokens_saved"] += c_eval["completion_tokens_saved"]
    c_metrics["api_calls"] += c_eval["api_calls"]

    c_res = c_eval | c_metrics
    c_row = make_result_row(agents, rounds, problems, model, c_res)
    print("Biography cost with CacheSaver:", c_row["cost ($)"])

    results_df["biography problems"] = results_df.index.map(nc_row)
    results_df["biography problems w. CacheSaver"] = results_df.index.map(c_row)

    return results_df

async def run_mmlu_experiment(model, size_of_experiment, results_df):
    agents = 3
    rounds = 2
    problems = int(100 * size_of_experiment)

    nc_file_name, nc_metrics = await gen_mmlu_main(agents=agents, rounds=rounds, problems=problems, model=model, use_cachesaver=False)    
    nc_eval = await eval_mmlu_main(file=nc_file_name)  
    nc_res = nc_eval | nc_metrics
    nc_row = make_result_row(agents, rounds, problems, model, nc_res)
    print("MMLU cost:", nc_row["cost ($)"])

    c_file_name, c_metrics = await gen_mmlu_main(agents=agents, rounds=rounds, problems=problems, model=model, use_cachesaver=True)
    c_eval = await eval_mmlu_main(file=c_file_name)
    c_res = c_eval | c_metrics
    c_row = make_result_row(agents, rounds, problems, model, c_res)
    print("MMLU cost with CacheSaver:", c_row["cost ($)"])

    results_df["mmlu problems"] = results_df.index.map(nc_row)
    results_df["mmlu problems w. CacheSaver"] = results_df.index.map(c_row)

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
                        "input_cost ($)", 
                        "input_cost_saved ($)", 
                        "output_tokens_used", 
                        "output_tokens_saved", 
                        "output_cost ($)", 
                        "output_cost_saved ($)", 
                        "cost ($)", 
                        "cost_saved ($)"]
    
    # results_df = await run_gen_math_experiment(model, size_of_experiment, results_df)
    # results_df = await run_gsm_experiment(model, size_of_experiment, results_df)
    results_df = await run_biography_experiment(model, size_of_experiment, results_df)
    # results_df = await run_mmlu_experiment(model, size_of_experiment, results_df)

    print(results_df)
    results_df.to_excel(f"experiment/{sanitize_model_name(model)}_Experiment.xlsx", index=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-s", "--size", type=float, required=True)
    args = parser.parse_args()

    asyncio.run(main(args.model, args.size))



