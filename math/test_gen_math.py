import unittest
import gen_math
import asyncio
import pandas as pd
import time
from itertools import product

class Test1(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Removes debuggging messages
        asyncio.get_running_loop().set_debug(False)
        self.results = []
    
    async def experiment_without_CacheSaver(self, agents, rounds, evaluation_round, model, use_cachesaver):
        start_time = time.time()
        result = await gen_math.main(agents=agents, rounds=rounds, problems=evaluation_round, model=model, use_cachesaver=use_cachesaver)
        end_time = time.time()
        
        runtime = end_time - start_time

        row = {
            "agents": agents,
            "rounds": rounds,
            "evaluation_round": evaluation_round,
            "use_cachesaver": use_cachesaver,
            "accuracy": round(result["mean"], 2),
            "standard error": result["sem"],
            "confidence interval": (float(result["ci"][0]), float(result["ci"][1])),
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
            "input_cost ($)" : result["input_cost"],
            "output_cost ($)" : result["output_cost"],
            "total_cost ($)" : result["total_cost"],
            "api_calls" : result["api_calls"],
            "runtime (s)": round(runtime, 2)
        }

        self.results.append(row)
    
    async def test_run_experiment_without_CacheSaver(self):
        print("Starting experiment...")

        values = [1, 2]
        use_cachesaver = False

        model = "meta-llama/llama-4-scout-17b-16e-instruct"

        configs = [(a, r, e, model, use_cachesaver) for a, r, e in product(values, repeat=3)]

        print(configs)
        print("Number of configs: ", len(configs))

        for agents, rounds, evaluation_round, model, use_cachesaver in configs:
            print(agents, rounds, evaluation_round)
            await self.experiment_without_CacheSaver(agents, rounds, evaluation_round, model, use_cachesaver)

        dataframe = pd.DataFrame(self.results)

        print("\nResults:")
        print(dataframe)
        
        dataframe.to_excel("math/Experiments/Llama4_17b/Experiment.xlsx", index=False)

        self.assertTrue(len(dataframe) > 0)

    #async def test_without_CacheSaver_100_eval_rounds(self):

    #    await self.experiment_without_CacheSaver(2, 3, 3, "qwen2.5:1.5b", True)
        
    #    dataframe = pd.DataFrame(self.results)

    #    print("\nResults:")
    #    print(dataframe)
        
    #    dataframe.to_excel("math/Experiments/Qwen2.5_1.5b/experiment_results_qwen2.5_1.5b_token_saved.xlsx", index=False)

if __name__ == '__main__':
    unittest.main()