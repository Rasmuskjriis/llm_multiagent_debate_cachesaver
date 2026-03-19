import unittest
import gen_math

class Test1(unittest.IsolatedAsyncioTestCase):   
    async def test_without_CacheSaver(self):
        agents = 3
        rounds = 2
        evaluation_round = 5

        await gen_math.main(agents, rounds, evaluation_round)

if __name__ == '__main__':
    unittest.main()