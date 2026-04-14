import unittest
import gen_mmlu, eval_mmlu
import asyncio
import pandas as pd
import time
from itertools import product

class Test1(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Removes debuggging messages
        asyncio.get_running_loop().set_debug(False)
        self.results = []

    async def experiment_without_CacheSaver(self, agents, rounds, problems, model, use_cachesaver):
        start_time = time.time()
        file_name, metrics = await gen_mmlu.main(agents=agents, rounds=rounds, problems=problems, model=model, use_cachesaver=use_cachesaver)
        eval = await eval_mmlu.main(file=file_name)
        end_time = time.time()
        
        runtime = end_time - start_time

        row = {
            "agents": agents,
            "rounds": rounds,
            "problems": problems,
            "use_cachesaver": use_cachesaver,
            "accuracy": round(eval["mean"], 2),
            "standard error": eval["sem"],
            "confidence interval": (float(eval["ci"][0]), float(eval["ci"][1])),
            "prompt_tokens": metrics["prompt_tokens"],
            "completion_tokens": metrics["completion_tokens"],
            "total_tokens": metrics["total_tokens"],
            "input_cost ($)" : metrics["input_cost"],
            "output_cost ($)" : metrics["output_cost"],
            "total_cost ($)" : metrics["total_cost"],
            "api_calls" : metrics["api_calls"],
            "runtime (s)": round(runtime, 2)
        }

        self.results.append(row)
    
    async def test_run_experiment_without_CacheSaver(self):
        print("Starting experiment with mmlu...")

        values = [1, 2]
        use_cachesaver = False

        model = "meta-llama/llama-4-scout-17b-16e-instruct"

        configs = [(a, r, e, model, use_cachesaver) for a, r, e in product(values, repeat=3)]

        print(configs)
        print("Number of configs: ", len(configs))

        for agents, rounds, problems, model, use_cachesaver in configs:
            print(agents, rounds, problems)
            await self.experiment_without_CacheSaver(agents, rounds, problems, model, use_cachesaver)

        dataframe = pd.DataFrame(self.results)

        print("\nResults:")
        print(dataframe)
        
        dataframe.to_excel("mmlu/Experiments/Llama4_17b/Experiment_mmlu.xlsx", index=False)

        self.assertTrue(len(dataframe) > 0)

    # async def test_without_CacheSaver_100_problems(self):

    #     await self.experiment_without_CacheSaver(1, 1, 10, "meta-llama/llama-4-scout-17b-16e-instruct", False)
            
    #     dataframe = pd.DataFrame(self.results)

    #     print("\nResults:")
    #     print(dataframe)
            
    #     dataframe.to_excel("mmlu/Experiments/Llama4_17b/Experiment_mmlu.xlsx", index=False)

if __name__ == '__main__':
    unittest.main()