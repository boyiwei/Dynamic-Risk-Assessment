import argparse
import copy
import json
import os
import pickle
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
# from dotenv import load_dotenv
import re
import backoff
import numpy as np
import openai
from tqdm import tqdm

from ctf_prompt import get_init_archive, get_prompt, get_reflexion_prompt

client = openai.OpenAI(api_key="token-abc123", base_url=f"http://localhost:6790/v1")

from utils import random_id, format_arc_data, eval_solution, list_to_string, bootstrap_confidence_interval
# load_dotenv(dotenv_path='.env')
Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""# Output Format:\nReply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a WELL-FORMED JSON object!\n"""
ROLE_DESC = lambda role: f"You are a {role}.\n\n"
SYSTEM_MSG = ""
CODE_INST = "You will write code to solve this task by creating a function named `run_conversation_step`.  You should make sure that you implement a version of the transformation that works for both example and test inputs. Make sure that the transform function is capable of handling both example and test inputs effectively, reflecting the learned transformation rules from the Examples inputs and outputs."

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(
        msg,
        model,
        system_message,
        temperature=0.5
):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature, max_tokens=12000
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    # cost = response.usage.completion_tokens / 1000000 * 15 + response.usage.prompt_tokens / 1000000 * 5
    assert not json_dict is None
    return json_dict


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt_reflect(
        msg_list,
        model,
        temperature=0.5
):
    response = client.chat.completions.create(
        model=model,
        messages=msg_list,
        temperature=temperature, max_tokens=12000
    )
    content = response.choices[0].message.content
    
        # json_dict = json.loads(content)
    raw_reply = content.strip()
        # Strip the ```json … ``` wrapper (if present)
    if raw_reply.startswith("```"):
        raw_reply = re.sub(r"^```[\w]*\n", "", raw_reply)   # remove opening fence
        raw_reply = raw_reply.rsplit("```", 1)[0].strip()   # remove closing fence
    try:
        # Parse into a native dict
        json_dict = json.loads(raw_reply)
    except json.JSONDecodeError:
        import ast
        json_dict = ast.literal_eval(raw_reply)
    assert not json_dict is None
    return json_dict


def search(args):
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        if "generation" in archive[-1] and isinstance(archive[-1]['generation'], int):
            start = archive[-1]['generation']
        else:
            start = 0
    else:
        archive = get_init_archive()
        start = 0

    for solution in archive:
        if 'fitness' in solution:
            continue

        solution['generation'] = "initial"
        fitness_str = "Median: 63.3%"
        solution['fitness'] = fitness_str

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)

    for n in range(start, args.n_generation):
        print(f"============Generation {n + 1}=================")
        system_prompt, prompt = get_prompt(archive)
        msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        for _ in range(20):
            try:
                next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)

                Reflexion_prompt_1, Reflexion_prompt_2 = get_reflexion_prompt(archive[-1] if n > 0 else None)
                # Reflexion 1
                msg_list.append({"role": "assistant", "content": str(next_solution)})
                msg_list.append({"role": "user", "content": Reflexion_prompt_1})
                next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
                # Reflexion 2
                msg_list.append({"role": "assistant", "content": str(next_solution)})
                msg_list.append({"role": "user", "content": Reflexion_prompt_2})
                next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
                code = next_solution['code']
                if re.search(r"def (?!run_conversation_step\()", code):
                    print("Extra function definitions are not allowed. Resampling")
                    continue
                else:
                    break
            except Exception as e:
                print("During LLM generate new solution:")
                print(e)
                continue
        # for _ in range(args.debug_max):
        #     try:
        #         acc_list = evaluate_forward_fn(args, next_solution["code"])
        #         if np.mean(acc_list) < 0.01 and SEARCHING_MODE:
        #             raise Exception("All 0 accuracy")
        #         break
        #     except Exception as e:
        #         print("During evaluation:")
        #         print(e)
        #         msg_list.append({"role": "assistant", "content": str(next_solution)})
        #         msg_list.append({"role": "user", "content": f"Error during evaluation:\n{e}\nCarefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought. Repeat your previous thought in 'thought', and put your thinking for debugging in 'debug_thought'"})
        #         try:
        #             next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
        #         except Exception as e:
        #             print("During LLM generate new solution:")
        #             print(e)
        #             continue
        #         continue
        # if not acc_list:
        #     continue

        # fitness_str = bootstrap_confidence_interval(acc_list)
        next_solution['fitness'] = "Pending Eval"
        next_solution['generation'] = n + 1

        # if 'debug_thought' in next_solution:
        #     del next_solution['debug_thought']
        # if 'reflection' in next_solution:
        #     del next_solution['reflection']
        archive.append(next_solution)
        print(next_solution['code'])
        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_data_path', type=str, default='sampled_arc_val_data.pkl')
    parser.add_argument('--test_data_path', type=str, default='sampled_arc_test_data.pkl')
    parser.add_argument('--n_repreat', type=int, default=5)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=32)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='iter_workflow_refinement/')
    parser.add_argument('--expr_name', type=str, default='ctf_results')
    parser.add_argument('--n_generation', type=int, default=20)
    parser.add_argument('--reflect_max', type=int, default=3)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--model',
                        type=str,
                        default='Qwen2.5-Coder-32B-Instruct',
                        choices=['gpt-4-turbo-2024-04-09', 'gpt-3.5-turbo-0125', 'gpt-4o-2024-05-13'])

    args = parser.parse_args()
    # search
    SEARCHING_MODE = True
    search(args)
