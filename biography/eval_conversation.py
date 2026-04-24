import json
import openai
import numpy as np
import time
import asyncio
import argparse

import clients.client_strategies as clients
from utils.utils import calc_mean_sem_ci, tokens_to_cost

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


def parse_yes_no(string):
    """
    Parses a string containing "yes" or "no" and returns a boolean value.

    Args:
        string (str): The string to parse.

    Returns:
        bool: True if the string contains "yes", False if the string contains "no".

    Raises:
        ValueError: If the input string does not contain "yes" or "no".
    """

    if "uncertain" in string.lower():
        return None
    elif "yes" in string.lower():
        return True
    elif "no" in string.lower():
        return False
    else:
        return None

def filter_people(person):
    people = person.split("(")[0]
    return people

async def main(file, model, use_cachesaver):
    if use_cachesaver:
        client = clients.CacheSaverGroqClient(model=model)
    else:
        client = clients.GroqClient(model=model)

    with open("{}".format(file), "r") as f:
        response = json.load(f)
        
    with open("biography/data/article.json", "r") as f:
        gt_data = json.load(f)

    gt_data_filter = {}

    for k, v in gt_data.items():
        k = filter_people(k)
        gt_data_filter[k] = v

    gt_data = gt_data_filter

    people = list(response.keys())

    accuracies = []

    for person in people:

        if person not in gt_data:
            continue

        gt_description = gt_data[person]
        gt_bullets = parse_bullets(gt_description)
        bio_descriptions = response[person]# [2][-1]['content']

        for description in bio_descriptions:

            bio_description = description[-1]['content']

            bio_bullets = parse_bullets(bio_description)
            if len(bio_bullets) == 1:
                if len(bio_bullets[0]) < 400:
                    continue

            bio_bullets = " ".join(bio_bullets)
            # continue

            for bullet in gt_bullets:
                message = [{"role": "user", "content": "Consider the following biography of {}: \n {} \n\n Is the above biography above consistent with the fact below? \n\n {} \n Give a single word answer, yes, no, or uncertain. Carefully check the precise dates and locations between the fact and the above biography.".format(person, bio_bullets, bullet)}]

                completion_metadata = await generate_answer(client, message)
                completion, metadata = completion_metadata

                content = completion.choices[0].message.content

                accurate = parse_yes_no(content)

                if accurate is not None:
                    accuracies.append(float(accurate))

    # Only update if LLM outputs a meaningful answer ie. a number to the list text_answers
    if len(accuracies) > 0:
        mean, sem, ci = calc_mean_sem_ci(accuracies)

    ci_low = mean-ci
    ci_high = mean+ci

    return {"mean": mean, 
            "sem": sem,
            "ci": (ci_low, ci_high)
            }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file", action="store", type=str, default="biography/results/biography_1_1.json")
    parser.add_argument("-m","--model", type=str, default="qwen3:0.6b")
    parser.add_argument("-c","--cachesaver", action="store_true", dest="use_cachesaver")

    args = parser.parse_args()

    asyncio.run(
        main(
            file=args.file,
            model=args.model,
            use_cachesaver=args.use_cachesaver
        )
    )