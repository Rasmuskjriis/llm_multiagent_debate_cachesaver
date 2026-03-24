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
        result = await gen_math.main(agents=agents, rounds=rounds, evaluation_round=evaluation_round, model=model, use_cachesaver=use_cachesaver)
        end_time = time.time()
        
        runtime = end_time - start_time

        row = {
            "agents": agents,
            "rounds": rounds,
            "evaluation_round": evaluation_round,
            "use_cachesaver": use_cachesaver,
            "accuracy": round(result["mean"], 2),
            #"std": result["std"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
            "runtime (s)": round(runtime, 2)
        }

        self.results.append(row)
    
    async def test_run_experiment_without_CacheSaver(self):
        print("Starting experiment...")

        values = [1, 2]
        use_cachesaver = False

        model = "qwen3:0.6b"

        configs = [(a, r, e, model, use_cachesaver) for a, r, e in product(values, repeat=3)]

        print(configs)
        print("Number of configs: ", len(configs))

        for agents, rounds, evaluation_round, model, use_cachesaver in configs:
            print(agents, rounds, evaluation_round)
            await self.experiment_without_CacheSaver(agents, rounds, evaluation_round, model, use_cachesaver)

        dataframe = pd.DataFrame(self.results)

        print("\nResults:")
        print(dataframe)
        
        dataframe.to_excel("Experiments/Qwen3_0.6b/experiment_results_qwen3_0.6b_new.xlsx", index=False)

        self.assertTrue(len(dataframe) > 0)


if __name__ == '__main__':
    unittest.main()