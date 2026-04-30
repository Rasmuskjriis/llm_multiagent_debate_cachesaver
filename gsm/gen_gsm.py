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
        completion, metadata = await client.create_chat_completion(
            messages = answer_context
            )
    except Exception as e:
        print(f"An error occurred: {e}")
        print("retrying due to an error......")
        await asyncio.sleep(30)
        return await generate_answer(client, answer_context)

    return completion, metadata

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
    random.seed(0)

    generated_description = {}

    api_calls = 0

    prompt_tokens_used = 0
    prompt_tokens_saved = 0
    completion_tokens_used = 0
    completion_tokens_saved = 0

    input_cost_used = 0
    output_cost_used = 0
    input_cost_saved = 0
    output_cost_saved = 0

    test_problems = read_jsonl("gsm/data/test.jsonl")
    random.shuffle(test_problems)

    for data in test_problems[:problems]:
        question = data['question']
        answer = data['answer']

        agent_contexts = [[{"role": "user", "content": """Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. """.format(question)}] for agent in range(agents)]

        for round in range(rounds):
            client = clients.make_client(model=model, use_cachesaver=use_cachesaver)

            tasks = []
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question)
                    agent_context.append(message)

                tasks.append(generate_answer(client, agent_context))

            completions_metadata = await asyncio.gather(*tasks)
            completions, metadata = zip(*completions_metadata)

            for i, agent_context in enumerate(agent_contexts):
                assistant_message = construct_assistant_message(completions[i])
                agent_context.append(assistant_message)

                usage = getattr(completions[i], "usage", None)
                usage_metadata = metadata[i]

                cached = usage_metadata.cached[0]
                duplicated = usage_metadata.duplicated[0]

                if cached: # If cached, all tokens are saved
                    prompt_tokens_saved += usage.prompt_tokens
                    completion_tokens_saved += usage.completion_tokens
                elif duplicated: # If duped only prompt tokens are saved
                    prompt_tokens_saved += usage.prompt_tokens
                    completion_tokens_used += usage.completion_tokens
                else:
                    prompt_tokens_used += usage.prompt_tokens
                    completion_tokens_used += usage.completion_tokens
                    api_calls += 1     

                # Add to cost
                #input_cost += tokens_to_cost(usage.prompt_tokens, usage.completion_tokens, model)[0]
                #output_cost += tokens_to_cost(usage.prompt_tokens, usage.completion_tokens, model)[1]
                #total_cost += tokens_to_cost(usage.prompt_tokens, usage.completion_tokens, model)[2]

        generated_description[question] = (agent_contexts, answer)

    file_name = "gsm/results/gsm_{}_{}.json".format(agents, rounds)
    with open(file_name, "w") as f: 
        json.dump(generated_description, f)

    return file_name, {
            "prompt_tokens_used" : prompt_tokens_used, 
            "completion_tokens_used" : completion_tokens_used,
            "prompt_tokens_saved" : prompt_tokens_saved,
            "completion_tokens_saved" : completion_tokens_saved,
            "input_cost_used" : input_cost_used,
            "output_cost_saved" : output_cost_saved,
            "api_calls" : api_calls
            }


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