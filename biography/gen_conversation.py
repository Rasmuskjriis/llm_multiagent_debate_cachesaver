import json
import openai
import random
from tqdm import tqdm
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
        await asyncio.sleep(5)
        return await generate_answer(client, answer_context)

    return completion, metadata

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


def filter_people(person):
    people = person.split("(")[0]
    return people


def construct_message(agents, idx, person, final=False):
    prefix_string = "Here are some bullet point biographies of {} given by other agents: ".format(person)

    if len(agents) == 0:
        return {"role": "user", "content": "Closely examine your biography and provide an updated bullet point biography."}


    for i, agent in enumerate(agents):
        agent_response = agent[idx]["content"]
        response = "\n\n Agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    if final:
        prefix_string = prefix_string + "\n\n Closely examine your biography and the biography of other agents and provide an updated bullet point biography.".format(person, person)
    else:
        prefix_string = prefix_string + "\n\n Using these other biographies of {} as additional advice, what is your updated bullet point biography of the computer scientist {}?".format(person, person)

    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}


async def main(agents, rounds, problems, model, use_cachesaver):
    if use_cachesaver:
        client = clients.CacheSaverAsyncGroq(model=model)
    else:
        client = clients.GroqClient(model=model)

    with open("biography/data/article.json", "r") as f:
        data = json.load(f)

    people = sorted(data.keys())
    people = [filter_people(person) for person in people]
    random.seed(1)
    random.shuffle(people)

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

    for person in tqdm(people[:problems]):
        agent_contexts = [[{"role": "user", "content": "Give a bullet point biography of {} highlighting their contributions and achievements as a computer scientist, with each fact separated with a new line character. ".format(person)}] for agent in range(agents)]

        for round in range(rounds):
            tasks = []
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]

                    if round == (rounds - 1):
                        message = construct_message(agent_contexts_other, 2*round - 1, person=person, final=True)
                    else:
                        message = construct_message(agent_contexts_other, 2*round - 1, person=person, final=False)
                    agent_context.append(message)

                tasks.append(generate_answer(client, agent_context))

            completions_metadata = await asyncio.gather(*tasks)
            completions, metadata = zip(*completions_metadata)

            for i, agent_context in enumerate(agent_contexts):
                assistant_message = construct_assistant_message(completions[i])
                agent_context.append(assistant_message)

                usage = getattr(completions[i], "usage", None)

                cached = metadata[i].cached
                duplicated = metadata[i].duplicated

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
                # input_cost += tokens_to_cost(usage.prompt_tokens, usage.completion_tokens, model)[0]
                # output_cost += tokens_to_cost(usage.prompt_tokens, usage.completion_tokens, model)[1]
                # total_cost += tokens_to_cost(usage.prompt_tokens, usage.completion_tokens, model)[2]

            bullets = parse_bullets(completions[-1].choices[0].message.content)

            # The LM just doesn't know this person so no need to create debates
            if len(bullets) == 1:
                break

        generated_description[person] = agent_contexts

        #print(agent_contexts)
    
    file_name = "biography/results/biography_{}_{}.json".format(agents, rounds)
    with open(file_name, "w") as f: 
        json.dump(generated_description, f)

    return file_name, {"prompt_tokens_used": prompt_tokens_used,
            "prompt_tokens_saved": prompt_tokens_saved,
            "completion_tokens_used": completion_tokens_used,
            "completion_tokens_saved": completion_tokens_saved,
            "input_cost_used" : input_cost_used,
            "output_cost_used" : output_cost_used,
            "input_cost_saved" : input_cost_saved,
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