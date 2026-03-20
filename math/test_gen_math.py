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
    
    async def experiment_without_CacheSaver(self, agents, rounds, evaluation_round, use_cachesaver):
        start_time = time.time()
        result = await gen_math.main(agents, rounds, evaluation_round, use_cachesaver)
        end_time = time.time()
        
        runtime = end_time - start_time

        row = {
            "agents": agents,
            "rounds": rounds,
            "evaluation_round": evaluation_round,
            "use_cachesaver": use_cachesaver,
            "accuracy": result["mean"].round(2),
            #"std": result["std"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
            "runtime (s)": round(runtime, 2)
        }

        self.results.append(row)
    
    async def test_run_experiment_without_CacheSaver(self):
        print("Starting experiment...")

        values = [1, 2, 3]
        use_cachesaver = False

        configs = [(a, r, e, use_cachesaver) for a, r, e in product(values, repeat=3)]

        print(configs)
        print("Number of configs: ", len(configs))

        for agents, rounds, evaluation_round, use_cachesaver in configs:
            print(agents, rounds, evaluation_round)
            await self.experiment_without_CacheSaver(agents, rounds, evaluation_round, use_cachesaver)

        dataframe = pd.DataFrame(self.results)

        print("\nResults:")
        print(dataframe)
        
        dataframe.to_excel("experiment_results_new.xlsx", index=False)

        self.assertTrue(len(dataframe) > 0)


if __name__ == '__main__':
    unittest.main()