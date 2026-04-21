import numpy as np
from tqdm import tqdm
import argparse
import re
import asyncio

import clients.client_strategies as clients
from utils.utils import calc_mean_sem_ci, tokens_to_cost

def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue

        bullet = bullet[idx:]

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets

# We need to use semaphore when running using CacheSavers api
#semaphore = asyncio.Semaphore(1)

async def generate_answer(client, answer_context):
    try:
        completion, metadata = await client.create_chat_completion(
            messages = answer_context
            )
        print("Metadata: ", metadata)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("retrying due to an error......")
        await asyncio.sleep(5)
        return await generate_answer(client, answer_context)

    return completion

def construct_message(agents_contexts, question):

    # Use introspection in the case in which there are no other agents.
    if len(agents_contexts) == 0:
        return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer, Include your reasoning step by step and at the end of your response, please write ONLY the final answer on a separate line WITH space on either side of the number like: Answer: <number> "}

    prefix_string = "The original question is: {}. These are the recent/updated opinions from other agents: ".format(question)

    agent_response = ""

    # Takes the last response from each given agent and affixes it to the message
    for agent_context in agents_contexts:
        for msg in reversed(agent_context):
            if msg["role"] == "assistant":
                agent_response = msg["content"]
                break

        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Include your reasoning step by step and at the end of your response, please write ONLY the final answer on a separate line WITH space on either side of the number like: Answer: <number> ".format(question)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}

def parse_answer(sentence):
    parts = sentence.split(" ")

    for part in parts[::-1]:
        try:
            filtered_part = re.sub(r'[^-0-9]', "", part)

            answer = float(filtered_part)
            return answer
        except:
            continue


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num


async def main(agents, rounds, problems, model, use_cachesaver):
    if use_cachesaver:
        client = clients.CacheSaverGroqClient(model=model)
    else:
        client = clients.GroqClient(model=model)

    answer = parse_answer("My answer is the same as the other agents and AI language model: the result of 12+28*19+6 is 550.")

    np.random.seed(0) # should be removed when we do our experiment

    scores = []

    generated_description = {}

    api_calls = 0

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    input_cost = 0
    output_cost = 0
    total_cost = 0

    mean = 0
    sem = 0
    ci = 0

    for round in tqdm(range(problems)):
        a, b, c, d, e, f = np.random.randint(50, 200, size=6)

        answer = a + b * c + d - e * f
        agent_contexts = [[{"role": "user", "content": """What is the result of {}+{}*{}+{}-{}*{}? Include your reasoning step by step and at the end of your response, please write ONLY the final answer on a separate line WITH space on either side of the number like: Answer: <number> """.format(a, b, c, d, e, f)}] for agent in range(agents)]

        content = agent_contexts[0][0]['content']
        question_prompt = "We seek to find the result of {}+{}*{}+{}-{}*{}?".format(a, b, c, d, e, f)

        for round in range(rounds):
            tasks = []
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question_prompt)
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

                print(f"  Round {round+1}, Agent {i+1}:")
                print(f"  Prompt tokens: {usage.prompt_tokens}")
                print(f"  Completion tokens: {usage.completion_tokens}")
                print(f"  Total tokens: {usage.total_tokens}")


        text_answers = []

        for agent_context in agent_contexts:
            text_answer = string =  agent_context[-1]['content']
            text_answer = text_answer.replace(",", ".")
            text_answer = parse_answer(text_answer)

            if text_answer is None:
                continue

            text_answers.append(text_answer)

        generated_description[(a, b, c, d, e, f)] = (agent_contexts, answer)

        try:
            text_answer = most_frequent(text_answers)
            if text_answer == answer:
                scores.append(1)
            else:
                scores.append(0)
        except:
            continue

    # Only update if LLM outputs a meaningful answer ie. a number to the list text_answers
    if len(text_answers) > 0 and len(scores) > 0:
        mean, sem, ci = calc_mean_sem_ci(scores)

    ci_low = mean-ci
    ci_high = mean+ci

    #print("CLIENT: ", client)

    #print("Prompt tokens: ", prompt_tokens)
    #print("Completion tokens: ", completion_tokens)
    #print("Total tokens: ", total_tokens)

    #print("Price -----------------")
    #print("Model: ", model)

    #print("Input cost: ", np.round(input_cost, 6))
    #print("Output cost: ", np.round(output_cost, 6))
    #print("Total cost: ", np.round(total_cost, 6))

    #print("\nAccuracy: ", mean)
    #print("CI: ", ci)

    #print("Api calls: ", api_calls)

    #print("\nConfidence interval: [", ci_low, ", ", ci_high, "]")

    #print(agent_contexts)

    return {"mean": mean, 
            "sem": sem,
            "ci": (ci_low, ci_high),
            "prompt_tokens": prompt_tokens, 
            "completion_tokens": completion_tokens, 
            "total_tokens": total_tokens,
            "input_cost" : input_cost,
            "output_cost" : output_cost,
            "total_cost" : total_cost,
            "api_calls" : api_calls
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--agents", type=int, default=2)
    parser.add_argument("-r", "--rounds", type=int, default=3)
    parser.add_argument("-p","--problems", type=int, default=10)
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