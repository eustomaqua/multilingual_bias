from openai_batcher import ChatOpenAIBatcher
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import os
import random
from collections import defaultdict
import numpy as np
from utils import output_gpt_request_jsons_for_instructions, read_jsonl, write_jsonl, \
    create_batch, retrieve_batch, retrieve_batch_response, list_batch, generate_with_vllm_multiprocessing, save_jsonl_rows_to_hf_dataset, \
    postprocess_outputs_row, get_values_by_key_ranges, extract_gpt_ranking_score


TRANSLATE_SYSTEM_PROMPT = """Instruction: %s

Orignal answer is:
%s

Answer in %s is:
"""


def translate_llama_responses():
    cross_instructions = read_jsonl('final/cross_infinity_firefly_instructions.jsonl')
    keys = [(i, k) for i, e in enumerate(cross_instructions) for k in e]
    cross_infinity_firefly_answers = read_jsonl('final/llama3_8b_infinity_firefly_instruction_answers.jsonl')
    cross_infinity_firefly_dict = defaultdict(dict)
    for i, e in enumerate(cross_infinity_firefly_answers):
        if i >= 177755*3+2:
            index = keys[i+1][0]
            tag = keys[i+1][1]
        else:
            tag = keys[i][1]
            index = keys[i][0] # if i < 177755*3+2 else e['index'] + 1
        cross_infinity_firefly_dict[index][tag] = e['instruction_' + keys[i][1]]
        cross_infinity_firefly_dict[index][tag.replace('instruction', 'answer')] = e['answer']
    languages = ['English', 'Chinese', 'Indonesian']
    language_tags = ['en', 'zh-cn', 'id']
    language_tags_map = {e: i for i, e in enumerate(language_tags)}
    requests = []
    requests_metadata = []
    for k, v in cross_infinity_firefly_dict.items():
        for each_key in v:
            if each_key.startswith('instruction'):
                answer = v['answer_' + each_key[12:]]
                for j, each_other_key in enumerate(language_tags):
                    if each_other_key != each_key[12:]:
                        requests.append(TRANSLATE_SYSTEM_PROMPT % (v[each_key], answer, languages[language_tags_map[each_other_key]]))
                        requests_metadata.append((k, each_key[12:], each_other_key))

    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    requests = requests[:512]
    all_answers = generate_with_vllm_multiprocessing(requests, model_name, num_gpus=8, tensor_parallel_size=1)

    final_ans = []
    for i in range(len(requests)):
        final_ans.append({'answer': all_answers[i], 'metadata': requests_metadata[i]})
    write_jsonl(final_ans, 'final/llama3_8b_infinity_firefly_instruction_multilingual_answers.jsonl')
    


if __name__ == '__main__':
    translate_llama_responses()