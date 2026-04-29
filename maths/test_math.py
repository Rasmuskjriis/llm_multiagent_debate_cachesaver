import unittest
import gen_math
import asyncio
import pandas as pd
import time
from itertools import product

from utils.utils import tokens_to_cost, clear_cache, sanitize_model_name


class Test1(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Removes debuggging messages
        asyncio.get_running_loop().set_debug(False)
        self.results = []
    
    async def experiment_without_CacheSaver(self, agents, rounds, problems, model, use_cachesaver):
        start_time = time.time()
        result = await gen_math.main(agents=agents, rounds=rounds, problems=problems, model=model, use_cachesaver=use_cachesaver)
        end_time = time.time()
        
        runtime = end_time - start_time

        input_cost_used, output_cost_used, total_cost_used = tokens_to_cost(result["prompt_tokens_used"], result["completion_tokens_used"], model)
        input_cost_saved, output_cost_saved, total_cost_saved = tokens_to_cost(result["prompt_tokens_saved"], result["completion_tokens_saved"], model)

        row = {
            "agents": agents,
            "rounds": rounds,
            "problems": problems,
            "use_cachesaver": use_cachesaver,
            "accuracy": round(result["mean"], 2),
            "standard error": result["sem"],
            "confidence interval": (float(result["ci"][0]), float(result["ci"][1])),
            "prompt_tokens_used": result["prompt_tokens_used"],
            "prompt_tokens_saved": result["prompt_tokens_saved"],
            "input_cost ($)": input_cost_used,
            "input_cost_saved ($)" : input_cost_saved,
            "completion_tokens_used": result["completion_tokens_used"],
            "completion_tokens_saved": result["completion_tokens_saved"],
            "output_cost ($)": output_cost_used,
            "output_cost_saved ($)" : output_cost_saved,
            "total_cost ($)": total_cost_used,
            "total_cost_saved ($)" : total_cost_saved,
            "api_calls" : result["api_calls"],
            "runtime (s)": round(runtime, 2)
        }

        self.results.append(row)
    
    # async def test_run_experiment_without_CacheSaver(self):
    #     print("Starting experiment with math...")

    #     values = [1, 2]
    #     use_cachesaver = False

    #     model = "meta-llama/llama-4-scout-17b-16e-instruct"

    #     configs = [(a, r, e, model, use_cachesaver) for a, r, e in product(values, repeat=3)]

    #     print(configs)
    #     print("Number of configs: ", len(configs))

    #     for agents, rounds, problems, model, use_cachesaver in configs:
    #         print(agents, rounds, problems)
    #         await self.experiment_without_CacheSaver(agents, rounds, problems, model, use_cachesaver)

    #     dataframe = pd.DataFrame(self.results)

    #     print("\nResults:")
    #     print(dataframe)
        
    #     dataframe.to_excel("math/Experiments/Llama4_17b/Experiment_math.xlsx", index=False)

    #     self.assertTrue(len(dataframe) > 0)

    async def test_without_CacheSaver_100_problems_1(self):

       await self.experiment_without_CacheSaver(2, 3, 1, "meta-llama/llama-4-scout-17b-16e-instruct", False)
        
       dataframe = pd.DataFrame(self.results)

       print("\nResults without CacheSaver:")
       print(dataframe)
        
       dataframe.to_excel("maths/Experiments/result_no_cachesaver.xlsx", index=False)

    async def test_with_CacheSaver_100_problems_2(self):

       clear_cache()

       await self.experiment_without_CacheSaver(2, 3, 1, "meta-llama/llama-4-scout-17b-16e-instruct", True)
        
       dataframe = pd.DataFrame(self.results)

       print("\nResults with CacheSaver:")
       print(dataframe)
        
       dataframe.to_excel("maths/Experiments/result_with_cachesaver.xlsx", index=False)

if __name__ == '__main__':
    unittest.main()