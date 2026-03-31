import openai
import json
import numpy as np
import random
import asyncio
import argparse

import clients.client_strategies as clients

async def generate_answer(client, answer_context):
    try:
        completion = await client.create_chat_completion(
            messages = answer_context
            )
    except Exception as e:
        print(f"An error occurred: {e}")
        print("retrying due to an error......")
        await asyncio.sleep(5)
        return await generate_answer(client, answer_context)

    return completion

def construct_message(agents_contexts, question):
    if len(agents_contexts) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}."}

    prefix_string = "These are the solutions to the problem from other agents: "

    # Takes the last response from each given agent and affixes it to the message
    for agent_context in agents_contexts:
        for msg in reversed(agent_context):
            if msg["role"] == "assistant":
                agent_response = msg["content"]
                break

        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(question)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


async def main(agents, rounds, problems, model, use_cachesaver):
    if use_cachesaver:
        client = clients.CacheSaverAsyncGroq(model=model)
    else:
        client = clients.GroqClient(model=model)

    random.seed(0)

    generated_description = {}

    test_problems = read_jsonl("gsm/data/test.jsonl")
    random.shuffle(test_problems)

    for data in test_problems[:problems]:
        question = data['question']
        answer = data['answer']

        agent_contexts = [[{"role": "user", "content": """Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. """.format(question)}] for agent in range(agents)]

        for round in range(rounds):
            tasks = []
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question)
                    agent_context.append(message)

                tasks.append(generate_answer(client, agent_context))

            completions = await asyncio.gather(*tasks)

            for i, agent_context in enumerate(agent_contexts):
                assistant_message = construct_assistant_message(completions[i])
                agent_context.append(assistant_message)

        generated_description[question] = (agent_contexts, answer)

    json.dump(generated_description, open("gsm/results/gsm_{}_{}.json".format(agents, rounds), "w"))

    print(answer)
    print(agent_contexts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--agents", type=int, default=2)
    parser.add_argument("-r", "--rounds", type=int, default=3)
    parser.add_argument("-p", "--problems", type=int, default=10)
    parser.add_argument("-m","--model", type=str, default="qwen3:0.6b")
    parser.add_argument("-c","--cachesaver", action="store_true", dest="use_cachesaver")

    args = parser.parse_args()

    asyncio.run(
        main(
            agents=args.agents, 
            rounds=args.rounds,
            problems=args.problems,
            model=args.model,
            use_cachesaver=args.use_cachesaver
        )
    )