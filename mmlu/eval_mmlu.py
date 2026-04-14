import json
import openai
import numpy as np
import time
import re
import argparse
import asyncio

from utils.utils import calc_mean_sem_ci

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
    if "yes" in string.lower():
        return True
    elif "no" in string.lower():
        return False
    else:
        return None


def solve_math_problems(input_str):
    pattern = r"\d+\.?\d*"

    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]

    return None

def parse_answer(input_str):
    pattern = r'\(\s*(\w)\s*\)'
    matches = re.findall(pattern, input_str)

    solution = None
    print("predicted solution")
    print(input_str)
    print("matches")
    print(matches)

    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            break

    print("final match")
    print(solution)

    return solution


def compute_accuracy(gt, pred_solutions):
    if type(pred_solutions) == list:
        pred_answers = []

        for pred_solution in pred_solutions:
            pred_answer = parse_answer(pred_solution)

            if pred_answer is None:
                pred_answer = solve_math_problems(pred_solution)

            if pred_answer is not None:
                pred_answers.append(pred_answer)

        if pred_answer is None:
            return 0
        pred_answer = most_frequent(pred_answers)
        # pred_answer = pred_answers[0]
    else:
        pred_answer = parse_answer(pred_solutions)
        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solutions)

    if gt == pred_answer:
        return 1
    else:
        return 0


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num

async def main(file):
    with open("{}".format(file), "r") as f:
        response_dict = json.load(f)
    
    questions = list(response_dict.keys())

    accuracies = []

    for question in questions:
        responses, gt = response_dict[question]

        pred_solutions = []
        for response in responses:
            pred_solution = response[-1]['content']

            pred_solutions.append(pred_solution)

        accurate = compute_accuracy(gt, pred_solutions)
        print("Actual solution: ", gt)

        if accurate is not None:
            accuracies.append(float(accurate))
        else:
            print(gt)

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

    parser.add_argument("-f", "--file", action="store", type=str, default="mmlu_1_1.json")

    args = parser.parse_args()

    asyncio.run(
        main(
            file=args.file
        )
    )