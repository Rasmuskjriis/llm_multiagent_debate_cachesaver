from openai import AsyncOpenAI
#from cachesaver.models.openai import AsyncOpenAI
import json
import numpy as np
import time
import pickle
from tqdm import tqdm
import argparse
import re
import asyncio

from clients.client_strategies import OllamaClient, CacheSaverOllamaClient

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

#semaphore = asyncio.Semaphore(1)

async def generate_answer(client, answer_context):
    try:
        completion = await client.create_chat_completion(messages=answer_context)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("retrying due to an error......")
        await asyncio.sleep(5)
        return await generate_answer(answer_context)

    return completion


def construct_message(agents, question):

    # Use introspection in the case in which there are no other agents.
    if len(agents) == 0:
        return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer, Include your reasoning step by step and at the end of your response, please write ONLY the final answer on a separate line WITH space on either side of the number like: Answer: <number> "}

    prefix_string = "The original question is: {}. These are the recent/updated opinions from other agents: ".format(question)

    agent_response = ""

    # Look only for assistant messages since in async it can be anything
    for agent in agents:
        for msg in reversed(agent):
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
    print("SENTENCE", sentence)
    parts = sentence.split(" ")

    print("SPLIT SENTENCE", parts)

    for part in parts[::-1]:
        try:
            print("PART", part)
            print("PART TYPE", type(part))

            filtered_part = re.sub(r'[^-0-9]', "", part)
            print("FILTERED PART", filtered_part)
            print("FILTERED PART TYPE", type(part))


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

async def main(agents, rounds, evaluation_round, model, use_cachesaver):
    client = OllamaClient(model=model)
    
    answer = parse_answer("My answer is the same as the other agents and AI language model: the result of 12+28*19+6 is 550.")

    #agents = 2
    #rounds = 3
    np.random.seed(0)

    #evaluation_round = 10
    scores = []

    generated_description = {}

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    mean = 0
    std = 0

    for round in tqdm(range(evaluation_round)):
        a, b, c, d, e, f = np.random.randint(0, 30, size=6)

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

                    #print("message: ", message)

                tasks.append(generate_answer(client, agent_context))
            
            completions = await asyncio.gather(*tasks)

            for i, completion in enumerate(completions):
                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                #print(completion)
            

        text_answers = []

        for agent_context in agent_contexts:
            text_answer = string =  agent_context[-1]['content']
            text_answer = text_answer.replace(",", ".")

            # Look only at assistant messages
            if agent_context[-1]["role"] != "assistant":
                continue

            print("Raw agent output:", repr(agent_context[-1]['content']))

            text_answer = parse_answer(text_answer)
            print("Parsed number:", text_answer)

            if text_answer is None:
                continue

            text_answers.append(text_answer)

        generated_description[(a, b, c, d, e, f)] = (agent_contexts, answer)

        try:
            print("TEXT ANSWERS: ", text_answers)
            text_answer = most_frequent(text_answers)
            print("MOST FREQUENT TEXT ANSWER: ", text_answer)
            print("ANSWER: ", answer)
            if text_answer == answer:
                scores.append(1)
            else:
                scores.append(0)
        except:
            continue
        print("SCORES: ", scores)

        # Prevents error if LLM doesn't output a meaningful answer
        if len(text_answers) > 0 and len(scores) > 0:
            mean = np.mean(scores)
            std = np.std(scores) / (len(scores) ** 0.5)

        usage = getattr(completion, "usage", None)
        prompt_tokens += usage.prompt_tokens
        completion_tokens += usage.completion_tokens
        total_tokens += usage.total_tokens

        #print("\nFinished")
        #print("Prompt tokens: ", usage.prompt_tokens)
        #print("Completion tokens: ", usage.completion_tokens)
        #print("Total tokens: ", usage.total_tokens)

    return {"mean": mean, 
            "std": std, 
            "prompt_tokens": prompt_tokens, 
            "completion_tokens": completion_tokens, 
            "total_tokens": total_tokens}

    #pickle.dump(generated_description, open("math_agents{}_rounds{}.p".format(agents, rounds), "wb"))
    #import pdb
    #pdb.set_trace()
    #print(answer)
    #print(agent_context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--agents", type=int, default=2)
    parser.add_argument("-r", "--rounds", type=int, default=3)
    parser.add_argument("-e","--evaluation_rounds", type=int, default=10)
    parser.add_argument("-m","--model", type=str, default="qwen3:0.6b")
    parser.add_argument("--no_cache", action="store_false", dest="use_cachesaver")

    args = parser.parse_args()

    asyncio.run(
        main(
            agents=args.agents, 
            rounds=args.rounds, 
            evaluation_round=args.evaluation_rounds, 
            model=args.model,
            use_cachesaver=args.use_cachesaver
        )
    )