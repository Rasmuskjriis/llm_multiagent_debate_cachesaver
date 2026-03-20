import unittest
import gen_math
import asyncio

class Test1(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Removes debuggging messages
        asyncio.get_running_loop().set_debug(False)
    
    async def test_without_CacheSaver(self):
        agents = 2
        rounds = 3
        evaluation_round = 5
        use_cachesaver = False

        print("Starting test_without_CacheSaver")
        await gen_math.main(agents, rounds, evaluation_round, use_cachesaver)

    async def test_with_CacheSaver(self):
        agents = 2
        rounds = 3
        evaluation_round = 5
        use_cachesaver = True

        print("Starting test_with_CacheSaver")
        await gen_math.main(agents, rounds, evaluation_round, use_cachesaver)

if __name__ == '__main__':
    unittest.main()