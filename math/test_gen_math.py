import unittest
import gen_math
import asyncio
import pandas as pd
import time

class Test1(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Removes debuggging messages
        asyncio.get_running_loop().set_debug(False)
        self.results = []
    
    async def experiment_without_CacheSaver(self, agents, rounds, evaluation_round, use_cachesaver):
        print("Starting test_without_CacheSaver")
        start_time = time.time()
        result = await gen_math.main(agents, rounds, evaluation_round, use_cachesaver)
        end_time = time.time()
        
        runtime = end_time - start_time

        row = {
            "agents": agents,
            "rounds": rounds,
            "evaluation_round": evaluation_round,
            "use_cachesaver": use_cachesaver,
            "mean": result["mean"],
            "std": result["std"],
            "runtime": runtime
        }

        self.results.append(row)
    
    async def test_run_experiment_without_CacheSaver(self):
        print("Starting experiment...")

        configs = [
            (1, 1, 1, False),
            (2, 2, 2, False),
            (3, 3, 3, False)
        ]

        for agents, rounds, evaluation_round, use_cachesaver in configs:
            await self.experiment_without_CacheSaver(agents, rounds, evaluation_round, use_cachesaver)

        dataframe = pd.DataFrame(self.results)

        print("\nResults:")
        print(dataframe)

        dataframe.to_csv("experiment_without_CacheSaver.csv", index=False)

        self.assertTrue(len(dataframe) > 0)


if __name__ == '__main__':
    unittest.main()