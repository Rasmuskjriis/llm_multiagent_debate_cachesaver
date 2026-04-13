import openai
import json
import numpy as np
import random
import asyncio
import argparse

import clients.client_strategies as clients
from utils.utils import tokens_to_cost

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

    api_calls = 0

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    input_cost = 0
    output_cost = 0
    total_cost = 0

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
                api_calls += 1

            completions = await asyncio.gather(*tasks)

            for i, agent_context in enumerate(agent_contexts):
                assistant_message = construct_assistant_message(completions[i])
                agent_context.append(assistant_message)

                usage = getattr(completions[i], "usage", None)

                # Add to token count
                prompt_tokens += usage.prompt_tokens
                completion_tokens += usage.completion_tokens
                total_tokens += usage.total_tokens

                # Add to cost
                input_cost += tokens_to_cost(usage.prompt_tokens, usage.completion_tokens, model)[0]
                output_cost += tokens_to_cost(usage.prompt_tokens, usage.completion_tokens, model)[1]
                total_cost += tokens_to_cost(usage.prompt_tokens, usage.completion_tokens, model)[2]

        generated_description[question] = (agent_contexts, answer)

    file_name = "gsm/results/gsm_{}_{}.json".format(agents, rounds)
    with open(file_name, "w") as f: 
        json.dump(generated_description, f)

    return file_name, {"prompt_tokens": prompt_tokens, 
            "completion_tokens": completion_tokens, 
            "total_tokens": total_tokens,
            "input_cost" : input_cost,
            "output_cost" : output_cost,
            "total_cost" : total_cost,
            "api_calls" : api_calls
            }

    # print(answer)
    # print(agent_contexts)

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