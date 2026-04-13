import os
import multiprocessing
from tqdm import tqdm
import json
from vllm import SamplingParams
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer
# from openai_batcher import ChatOpenAIBatcher
from collections import defaultdict
from tqdm import tqdm
from utils import async_request_questions, postprocess_mcot_answer, postprocess_thinking_answer, crosslingual_consistency, extract_boxed_answer
import numpy as np
import pandas as pd
import re
from collections import Counter

from utils import ChatOpenAIBatcher


def write_jsonl(lines, data_path):
    with open(data_path, 'w', encoding='utf-8') as f:
        for l in lines:
            f.writelines(json.dumps(l, ensure_ascii=False) + '\n')


def read_jsonl(data_path):
    lines = []
    with open(data_path) as f:
        for l in f:
            lines.append(json.loads(l))
    return lines


def _postprocess_row(row):
    assistant_response = row['response']['body']['choices'][0]['message']['content']
    return {'id': row['custom_id'], 'response': assistant_response}


def _postprocess_message(messages):
    is_user_first_turn = messages[0]['from'] == 'human'
    convs = []
    if not is_user_first_turn:
        convs.append({'human': '', 'assistant': messages[0]['value']})
        messages = messages[1:]
    for idx in range(0, len(messages), 2):
        convs.append({'human': messages[idx]['value'], 'assistant': messages[idx+1]['value']})
    return convs


def worker(gpu_id, prompts_chunk, return_dict, progress_queue, model_name):
    # Set the environment variable to specify the GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_ENDPOINT"] = "http://hf-mirror.com"
    # Import inside the worker to ensure the correct CUDA device is used
    from vllm import LLM, SamplingParams
    
    sampling_params = SamplingParams(
        temperature=0.9,
        top_p=0.9,
        max_tokens=4096,
        presence_penalty=0,
        frequency_penalty=0,
    )

    # Initialize the LLM instance (will use the specified GPU)
    llm = LLM(model=model_name)

    if gpu_id == 0:
        use_tqdm = True
    else:
        use_tqdm = False
    outputs = llm.generate(prompts_chunk, sampling_params, use_tqdm=use_tqdm)
    generated_texts = [o.outputs[0].text for o in outputs]

    for _ in range(len(outputs)):
        progress_queue.put(1)

    # Store the results in the shared dictionary
    return_dict[gpu_id] = generated_texts


def generate_with_vllm_multiprocessing(prompts, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", num_gpus=8):
    """
    Generates text using the specified language model across multiple GPUs using multiprocessing.
    Displays the generation progress using tqdm.

    Args:
        prompts (List[str]): A list of input prompts for generation.
        model_name (str): The name or path of the language model to use.
        num_gpus (int): The number of GPUs to use for data parallelism.

    Returns:
        List[str]: A list of generated texts corresponding to the input prompts.
    """
    # Ensure the number of GPUs does not exceed available prompts
    num_processes = min(num_gpus, len(prompts))

    # Split prompts among the processes
    chunk_size = (len(prompts) + num_processes - 1) // num_processes  # Ceiling division
    prompt_chunks = [prompts[i*chunk_size : (i+1)*chunk_size] for i in range(num_processes)]

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    # Create a multiprocessing Queue for progress updates
    progress_queue = multiprocessing.Queue()

    processes = []
    for gpu_id in range(num_processes):
        if not prompt_chunks[gpu_id]:
            continue  # Skip if the chunk is empty
        p = multiprocessing.Process(
            target=worker,
            args=(gpu_id, prompt_chunks[gpu_id], return_dict, progress_queue, model_name)
        )
        processes.append(p)
        p.start()

    # Set up tqdm progress bar
    total_prompts = len(prompts)
    with tqdm(total=total_prompts, desc="Generating", unit="prompt") as pbar:
        completed_prompts = 0
        while completed_prompts < total_prompts:
            # Wait for progress updates from workers
            progress_queue.get()
            completed_prompts += 1
            pbar.update(1)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Collect and combine the results, preserving the order of prompts
    all_generated_texts = []
    # Since the prompts were split in order, we can reconstruct the order
    for gpu_id in range(num_processes):
        all_generated_texts.extend(return_dict.get(gpu_id, []))

    return all_generated_texts


def generate_crosslingual_openhermes():
    import multiprocessing
    # Set the multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    chat_inputs = [read_jsonl('data/translate_openhermes_zh.%d.jsonl' % idx) for idx in range(5)]
    chat_outputs = [read_jsonl('output/translate_openhermes_zh_output.%d.jsonl' % i) for i in range(5)]
    response_dict = dict()
    wrong_count = 0
    for i, chat_list in enumerate(chat_outputs):
        for c in chat_list:
            resp = _postprocess_row(c)
            try:
                response_dict[i * 40000 + int(resp['id'].replace('request-', ''))] = json.loads(resp['response'])
            except Exception as exc:
                wrong_count += 1

    ans = []
    wrong_count = 0
    en_questions = []
    cn_questions = []
    for i, chat_list in enumerate(chat_inputs):
        for chat in chat_list:
            custom_id = i * 40000 + int(chat['custom_id'].replace('request-', ''))
            if custom_id not in response_dict:
                continue
            zh_inst = response_dict[custom_id]
            eng_inst = eval(chat['body']['messages'][1]['content'])
            if '0' in zh_inst:
                zh_inst = [zh_inst[k] for k in zh_inst]
            elif 'value' not in zh_inst:
                zh_inst = zh_inst[list(zh_inst.keys())[0]]
            zh_inst = [e for e in zh_inst if e and isinstance(e, dict) and 'from' in e and 'value' in e]
            if len(zh_inst) != len(eng_inst):
                wrong_count += 1
                continue
            zh_inst = _postprocess_message(zh_inst)
            eng_inst = _postprocess_message(eng_inst)
            flag_skip = False
            keywords_filter = ["as of my", "as of 2020", "as of 2021", "as of 2022", "up to 2021"]
            for idx in range(len(zh_inst)):
                zh_inst[idx]['en_human'] = eng_inst[idx]["human"]
                zh_inst[idx]['zh_assistant'] = zh_inst[idx]["assistant"]
                zh_inst[idx]['assistant'] = eng_inst[idx]["assistant"]
                zh_inst[idx]['cot_human'] = ''
                zh_inst[idx]['cot_assistant'] = ''
                if any(k in eng_inst[idx]["assistant"].lower() for k in keywords_filter):
                    flag_skip = True
                    break
                elif eng_inst[idx]["assistant"].lower().startswith('as of'):
                    flag_skip = True
                    break
            if flag_skip:
                continue
            en_questions.append((custom_id, zh_inst[idx]['en_human']))
            cn_questions.append((custom_id, zh_inst[idx]['human']))

    en_questions = ['<|start_header_id|> user <|end_header_id|>\n' + e[1] + '<|start_header_id|> assistant <|end_header_id|>\n' for e in en_questions]
    cn_questions = ['<|start_header_id|> user <|end_header_id|>\n' + e[1] + '<|start_header_id|> assistant <|end_header_id|>\n' for e in cn_questions]

    n = len(cn_questions)
    n = int(8 * (n // 8))

    generated_texts = generate_with_vllm_multiprocessing(cn_questions[:n])

    cn_instructions = [{'instruction': cn_questions[i], 'response': generated_texts[i]} for i in range(len(generated_texts))]

    write_jsonl(cn_instructions, 'cn_instructions.jsonl')


def generate_eval_crosslingual_openhermes():
    import multiprocessing
    # Set the multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    eval_dataset = load_from_disk('/path/to/crosslingual_openhermes_simpo_dataset/test')

    prompts = list(eval_dataset['prompt'])
    chosen = list(eval_dataset['chosen'])
    rejected = list(eval_dataset['rejected'])
    higher_language = list(eval_dataset['higher_language'])

    llama_prompts = list(map(lambda x: '<|start_header_id|> user <|end_header_id|>\n%s<|start_header_id|> assistant <|end_header_id|>\n' % x, prompts))

    # Original model response
    model1_answers = []
    for i in range(len(prompts)):
        if higher_language[i] == 'english':
            answer = rejected[i][1]['content']
            model1_answers.append(answer)
        else:
            answer = chosen[i][1]['content']
            model1_answers.append(answer)

    # answers from gpt-aligned
    model2_path = '/path/to/outputs/llama-3-8b-instruct-gpt-simpo-v2'
    model2_answers = generate_with_vllm_multiprocessing(llama_prompts, model2_path)

    model3_path = '/path/to/outputs/llama-3-8b-instruct-simpo-v2'
    model3_answers = generate_with_vllm_multiprocessing(llama_prompts, model3_path)

    final_json = [{'instruction': prompts[i], 'answer_1': model1_answers[i], 'answer_2': model2_answers[i], 'answer_3': model3_answers[i]} for i in range(len(prompts))]
    write_jsonl(final_json, 'final/ranking_eval_crosslingual_openhermes.jsonl')


def generate_eval_reasoning_datasets():
    import multiprocessing
    # Set the multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    reasoning_instructions_translations = [e for i in range(2) for e in read_jsonl('./output/output_reasoning_translation.%d.jsonl' % i)]
    for i in range(len(reasoning_instructions_translations)):
        reasoning_instructions_translations[i]['custom_id'] = int(reasoning_instructions_translations[i]['custom_id'][8:])
    # reasoning_instructions_translations.sort(key=lambda x: x['custom_id'])

    def _postprocess(row):
        assistant_response = row['response']['body']['choices'][0]['message']
        try:
            response = eval(assistant_response['content']) if 'content' in assistant_response else ''
        except:
            response = ''
        return {'id': row['custom_id'], 'response': response}

    reasoning_instructions_translations = list(filter(lambda x: x['response'], map(_postprocess, reasoning_instructions_translations)))
    reasoning_instructions_translations = list(filter(lambda x: len(x['response']) >= 5, reasoning_instructions_translations))
    
    instruction_english = [e['response'].get('instruction_english', '') for e in reasoning_instructions_translations]
    instruction_chinese = [e['response'].get('instruction_chinese', '')for e in reasoning_instructions_translations]
    instruction_indonesian = [e['response'].get('instruction_indonesian', '') for e in reasoning_instructions_translations]
    instruction_vietnamese = [e['response'].get('instruction_vietnamese', '') for e in reasoning_instructions_translations]
    instruction_thai = [e['response'].get('instruction_thai', '') for e in reasoning_instructions_translations]
    valid_indexes = []
    for i in range(len(instruction_english)):
        if not instruction_english[i] or not instruction_chinese[i] or not instruction_indonesian[i] or not instruction_vietnamese[i] or not instruction_thai[i]:
            continue
        valid_indexes.append(i)
    instruction_english = [instruction_english[i] for i in valid_indexes]
    instruction_chinese = [instruction_chinese[i] for i in valid_indexes]
    instruction_indonesian = [instruction_indonesian[i] for i in valid_indexes]
    instruction_vietnamese = [instruction_vietnamese[i] for i in valid_indexes]
    instruction_thai = [instruction_thai[i] for i in valid_indexes]
    model_path = 'meta-llama/Llama-3.1-8B-Instruct'
    
    all_instructions = instruction_english + instruction_chinese + instruction_indonesian + instruction_vietnamese + instruction_thai
    llama_prompts = list(map(lambda x: '<|start_header_id|> user <|end_header_id|>\n%s<|start_header_id|> assistant <|end_header_id|>\n' % x, all_instructions))
    outputs = generate_with_vllm_multiprocessing(llama_prompts, model_path)
    
    n = len(instruction_english)
    output_english = outputs[:n]
    output_chinese = outputs[n:2*n]
    output_indonesian = outputs[2*n:3*n]
    output_vietnamese = outputs[3*n:4*n]
    output_thai = outputs[4*n:]
    

    final_ans = []
    for i in range(n):
        final_ans.append({'instruction_english': instruction_english[i], 'instruction_chinese': instruction_chinese[i], 
                          'instruction_indonesian': instruction_indonesian[i], 'instruction_vietnamese': instruction_vietnamese[i], 
                          'instruction_thai': instruction_thai[i], 'output_english': output_english[i], 'output_chinese': output_chinese[i],
                          'output_indonesian': output_indonesian[i], output_vietnamese[i]: 'output_vietnamese', 'output_thai': output_thai[i]})
    write_jsonl(final_ans, 'final/reasoning_crosslingual_llama_result.jsonl')


def generate_consistency_eval_crosslingual_openhermes():
    import multiprocessing
    # Set the multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    cross_openhermes = read_jsonl('./taipei/cross_openhermes.jsonl')

    cross_openhermes_dict = defaultdict(dict)
    for d in cross_openhermes:
        cross_openhermes_dict[d['conversation_id']][d['category']] = d['conversation']

    keys = list(sorted(cross_openhermes_dict.keys(), reverse=True))
    prompt_ens = []
    prompt_zhs = []
    for key in keys[:5000]:
        output_dict = dict()
        last_turn_dict = dict()
        for tag in ['translate-openhermes-en', 'translate-openhermes-zh']:
            if tag not in cross_openhermes_dict[key]:
                continue
            tag_output = cross_openhermes_dict[key][tag]
            turns = []
            if tag_output and len(tag_output) >= 2 and tag_output[0]['human'] == '' and tag_output[0]['assistant'].strip():
                turns.append('<|start_header_id|> system <|end_header_id|> %s <|eot_id|>' % tag_output[0]['assistant'].strip())
                turns.append('<|start_header_id|> user <|end_header_id|>\n%s<|start_header_id|> assistant <|end_header_id|>\n' % tag_output[1]['human'])
            elif tag_output and tag_output[0]['human']:
                turns.append('<|start_header_id|> user <|end_header_id|>\n%s<|start_header_id|> assistant <|end_header_id|>\n' % tag_output[0]['human'])
            prompt = ''.join(turns)
            if prompt:
                output_dict[tag] = prompt
            if turns:
                last_turn_dict[tag] = turns[-1]
        if len(output_dict) == 2 and last_turn_dict['translate-openhermes-en'].strip() != last_turn_dict['translate-openhermes-zh'].strip():
            prompt_ens.append(output_dict['translate-openhermes-en'])
            prompt_zhs.append(output_dict['translate-openhermes-zh'])

    normalize_prompt = lambda x: x.replace('<|start_header_id|> system <|end_header_id|>', '').replace('<|eot_id|>', '').replace('<|start_header_id|> user <|end_header_id|>', '').strip()

    model_paths = ['aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct', 'google/gemma-2-9b-it', 'SeaLLMs/SeaLLMs-v3-7B-Chat']
    final_json = read_jsonl('final/ranking_consistency_eval_crosslingual_openhermes.jsonl')
    for model_path in model_paths:
        model_answers = generate_with_vllm_multiprocessing(prompt_ens + prompt_zhs, model_path)
        model_en_answers = model_answers[:len(prompt_ens)]
        model_cn_answers = model_answers[len(prompt_ens):]

        final_json += [{'instruction_english': normalize_prompt(prompt_ens[i]), 'instruction_chinese': normalize_prompt(prompt_zhs[i]), 'answer_english': model_en_answers[i], 'answer_chinese': model_cn_answers[i]} for i in range(len(prompt_ens))]
    write_jsonl(final_json, 'final/ranking_consistency_eval_crosslingual_openhermes.jsonl')


def generate_cmmlu_eval_model_responses():
    import multiprocessing
    # Set the multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    cmmlu = load_dataset(r"SeaEval/cmmlu", split='test')
    n = len(cmmlu)

    system_prompt = '你是一个人工智能助手。用户会给你一个问题。你的目标是尽可能地准确回答问题'
    all_prompts = []
    model_paths = ['/path/to/outputs/llama3-8b-infinity-firefly-score-filter-sft-full']
    tokenizers = [AutoTokenizer.from_pretrained(model_path) for model_path in model_paths]
    questions = list(cmmlu['question'])
    choices = list(cmmlu['choices'])

    all_prompts = []
    raw_prompts = []
    for tokenizer in tokenizers:
        model_raw_prompts = []
        model_messages = []
        for i in tqdm(range(n)):
            question = questions[i]
            choice_str = '\n'.join(choices[i])
            prompt = '问题：%s\n\n答案选项：\n%s\n\n答案是什么？' % (question, choice_str)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            model_messages.append(messages)
            model_raw_prompts.append(prompt)
        try:
            model_prompts = tokenizer.apply_chat_template(model_messages, tokenize=False, add_generation_prompt=True)
        except Exception as exc:
            # Remove system roles
            model_messages_without_system_roles = [messages[1:] for messages in model_messages]
            model_prompts = tokenizer.apply_chat_template(model_messages_without_system_roles, tokenize=False, add_generation_prompt=True)
        all_prompts.append(model_prompts)
        raw_prompts.append(model_raw_prompts)

    def _postprocess_answer(ans):
        ans = ans.strip()
        if ans.startswith('The answer should be directly generated:\n'):
            ans = ans[41:]
        elif '\n\nThe final answer is' in ans:
            ans = ans[ans.rindex('\n\nThe final answer is') + 22:]
        ans = ans.strip()
        if ans.startswith(':'):
            ans = ans[1:].strip()
        for lang in ['Chinese', 'Indonesian', 'English']:
            key = 'The answer should be thinking in %s:\n' % lang
            if ans.startswith(key):
                ans = ans[len(key):].strip()
        return ans
    
    # Generation
    final_json = []
    for i, model_path in enumerate(model_paths):
        model_answers = generate_with_vllm_multiprocessing(all_prompts[i], model_path)
        final_json += [{'instruction': raw_prompts[i][j], 'original_answer': model_answers[j], 'answer': _postprocess_answer(model_answers[j]), 'model': model_path} for j in range(len(all_prompts[i]))]
    write_jsonl(final_json, 'final/llama3_infinity_firefly_score_filter_cmmlu_eval_model_responses.jsonl')


def generate_crosslogiqa_eval_model_responses():
    import multiprocessing
    # Set the multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    crosslogiqa = load_dataset(r"SeaEval/cross_logiqa", split='test')

    system_prompt = 'You are a helpful assistant'
    all_prompts = []
    model_paths = ['/path/to/outputs/qwen2-5-7b-infinity-firefly-score-filter-sft-full/checkpoint-512']
    tokenizers = [AutoTokenizer.from_pretrained(model_path) for model_path in model_paths]
    questions = [e['English']['question'] for e in crosslogiqa] + [e['Chinese']['question'] for e in crosslogiqa] + [e['Indonesian']['question'] for e in crosslogiqa]
    choices = [e['English']['choices'] for e in crosslogiqa] + [e['Chinese']['choices'] for e in crosslogiqa] + [e['Indonesian']['choices'] for e in crosslogiqa]
    contexts = [e['English']['context'] for e in crosslogiqa] + [e['Chinese']['context'] for e in crosslogiqa] + [e['Indonesian']['context'] for e in crosslogiqa]
    n = len(questions)

    all_prompts = []
    raw_prompts = []
    for tokenizer in tokenizers:
        model_raw_prompts = []
        model_messages = []
        for i in tqdm(range(n)):
            question = questions[i]
            c = contexts[i]
            choice_str = '\n'.join(choices[i])
            prompt = 'Context: %s\n\nQuestion: %s\n\nOptions：\n%s\n\nAnswer is?' % (c, question, choice_str)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            model_messages.append(messages)
            model_raw_prompts.append(prompt)
        try:
            model_prompts = tokenizer.apply_chat_template(model_messages, tokenize=False, add_generation_prompt=True)
        except Exception as exc:
            # Remove system roles
            model_messages_without_system_roles = [messages[1:] for messages in model_messages]
            model_prompts = tokenizer.apply_chat_template(model_messages_without_system_roles, tokenize=False, add_generation_prompt=True)
        all_prompts.append(model_prompts)
        raw_prompts.append(model_raw_prompts)

    def _postprocess_answer(ans):
        ans = ans.strip()
        thinkin = 'original'

        for lang in ['Chinese', 'Indonesian', 'English']:
            key = 'The answer should be thinking in %s:\n' % lang
            if ans.startswith(key):
                thinkin = lang
                break

        if ans.startswith('The answer should be directly generated:\n'):
            ans = ans[41:]
        elif '\n\nThe final answer is' in ans:
            ans = ans[ans.rindex('\n\nThe final answer is') + 22:]
        ans = ans.strip()
        if ans.startswith(':'):
            ans = ans[1:].strip()
        for lang in ['Chinese', 'Indonesian', 'English']:
            key = 'The answer should be thinking in %s:\n' % lang
            if ans.startswith(key):
                ans = ans[len(key):].strip()
                thinkin = lang
                break
        return f'{ans}'
    
    # Generation
    final_json = []
    for i, model_path in enumerate(model_paths):
        model_answers = async_request_questions(model_raw_prompts, port=7991)  # generate_with_vllm_multiprocessing(all_prompts[i], model_path, num_gpus=6)
        final_json += [{'instruction': raw_prompts[i][j], 'original_answer': model_answers[j]['answer'], 'answer': _postprocess_answer(model_answers[j]['answer']), 'model': model_path} for j in range(len(all_prompts[i]))]
    write_jsonl(final_json, 'final/qwen2_7b_infinity_firefly_crosslogiqa_eval_model_responses.jsonl')


def generate_mmmlu_eval_model_responses():
    import multiprocessing
    # Set the multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    splits = ['ar', 'bn', 'ca', 'da', 'de', 'en', 'es', 'eu', 'fr', 'gu', 'hi', 'hr', 'hu', 'hy', 'id', 'is', 'it', 'kn', 'ml', 'mr', 'nb', 'ne', 'nl', 'pt', 'ro', 'ru', 'sk', 'sr', 'sv', 'ta', 'te', 'uk', 'vi', 'zh']
    crossmmlu = [load_dataset(r"alexandrainst/m_mmlu", s) for s in splits]

    max_examples = 500
    common_ids = list(sorted(set.intersection(*[set(e['test']['id']) for e in crossmmlu])))[:max_examples]

    system_prompt = 'You are a helpful assistant'
    all_prompts = []
    model_paths = ['/path/to/outputs/qwen2-5-7b-infinity-firefly-score-filter-sft-full/checkpoint-512']
    tokenizers = [AutoTokenizer.from_pretrained(model_path) for model_path in model_paths]
    questions = [ee['instruction'] for e in crossmmlu for ee in e['test'] if ee['id'] in common_ids]
    choices = [['(A): ' + ee['option_a'], '(B): ' + ee['option_b'], '(C): ' + ee['option_c'], '(D): ' + ee['option_d']] for e in crossmmlu for ee in e['test'] if ee['id'] in common_ids]
    n = len(questions)

    all_prompts = []
    raw_prompts = []
    for tokenizer in tokenizers:
        model_raw_prompts = []
        model_messages = []
        for i in tqdm(range(n)):
            question = questions[i]
            choice_str = '\n'.join(choices[i])
            prompt = '%s\n\nOptions：\n%s\n\nAnswer is?' % (question, choice_str)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            model_messages.append(messages)
            model_raw_prompts.append(prompt)
        try:
            model_prompts = tokenizer.apply_chat_template(model_messages, tokenize=False, add_generation_prompt=True)
        except Exception as exc:
            # Remove system roles
            model_messages_without_system_roles = [messages[1:] for messages in model_messages]
            model_prompts = tokenizer.apply_chat_template(model_messages_without_system_roles, tokenize=False, add_generation_prompt=True)
        all_prompts.append(model_prompts)
        raw_prompts.append(model_raw_prompts)

    def _postprocess_answer(ans):
        ans = ans.strip()
        thinkin = 'original'

        for lang in ['Chinese', 'Indonesian', 'English']:
            key = 'The answer should be thinking in %s:\n' % lang
            if ans.startswith(key):
                thinkin = lang
                break

        if ans.startswith('The answer should be directly generated:\n'):
            ans = ans[41:]
        elif '\n\nThe final answer is' in ans:
            ans = ans[ans.rindex('\n\nThe final answer is') + 22:]
        ans = ans.strip()
        if ans.startswith(':'):
            ans = ans[1:].strip()
        for lang in ['Chinese', 'Indonesian', 'English']:
            key = 'The answer should be thinking in %s:\n' % lang
            if ans.startswith(key):
                ans = ans[len(key):].strip()
                thinkin = lang
                break
        return f'{ans}'
    
    # Generation
    final_json = []
    for i, model_path in enumerate(model_paths):
        model_answers = async_request_questions(model_raw_prompts, port=7991)  # generate_with_vllm_multiprocessing(all_prompts[i], model_path)
        final_json += [{'instruction': raw_prompts[i][j], 'original_answer': model_answers[j]['answer'], 'answer': _postprocess_answer(model_answers[j]['answer']), 'model': model_path} for j in range(len(all_prompts[i]))]
    write_jsonl(final_json, 'final/qwen2_7b_infinity_firefly_mmmlu_eval_model_responses.jsonl')


def generate_crossmmlu_eval_model_responses():
    import multiprocessing
    # Set the multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    crossmmlu = load_dataset(r"SeaEval/cross_mmlu", split='test')

    system_prompt = 'You are a helpful assistant'
    all_prompts = []
    model_paths = ['/path/to/outputs/qwen2-5-7b-infinity-firefly-score-filter-sft-full/checkpoint-512']
    questions = [e['English']['question'] for e in crossmmlu] + [e['Chinese']['question'] for e in crossmmlu] + [e['Indonesian']['question'] for e in crossmmlu]
    choices = [e['English']['choices'] for e in crossmmlu] + [e['Chinese']['choices'] for e in crossmmlu] + [e['Indonesian']['choices'] for e in crossmmlu]
    n = len(questions)

    raw_prompts = []
    model_raw_prompts = []
    model_messages = []
    for i in tqdm(range(n)):
        question = questions[i]
        choice_str = '\n'.join(choices[i])
        prompt = '%s\n\nOptions：\n%s\n\nAnswer is?' % (question, choice_str)
        messages = [
            {"role": "user", "content": prompt}
        ]
        model_messages.append(messages)
        model_raw_prompts.append(prompt)
    raw_prompts.append(model_raw_prompts)

    answer_choices = async_request_questions(raw_prompts, max_workers=128, port=7991)
    import pdb; pdb.set_trace()

    def _postprocess_answer(ans):
        ans = ans.strip()
        thinkin = 'original'

        for lang in ['Chinese', 'Indonesian', 'English']:
            key = 'The answer should be thinking in %s:\n' % lang
            if ans.startswith(key):
                thinkin = lang
                break

        if ans.startswith('The answer should be directly generated:\n'):
            ans = ans[41:]
        elif '\n\nThe final answer is' in ans:
            ans = ans[ans.rindex('\n\nThe final answer is') + 22:]
        ans = ans.strip()
        if ans.startswith(':'):
            ans = ans[1:].strip()
        for lang in ['Chinese', 'Indonesian', 'English']:
            key = 'The answer should be thinking in %s:\n' % lang
            if ans.startswith(key):
                ans = ans[len(key):].strip()
                thinkin = lang
                break
        return f'{ans}'
    
    # Generation
    final_json = []
    for i, model_path in enumerate(model_paths):
        model_answers = async_request_questions(model_raw_prompts, port=7991)  # generate_with_vllm_multiprocessing(all_prompts[i], model_path)
        final_json += [{'instruction': raw_prompts[i][j], 'original_answer': model_answers[j]['answer'], 'answer': _postprocess_answer(model_answers[j]['answer']), 'model': model_path} for j in range(len(raw_prompts[i]))]
    write_jsonl(final_json, 'final/qwen2_7b_infinity_firefly_crossmmlu_eval_model_responses.jsonl')


def extract_and_eval_indommlu_ans():
    indommlu = load_dataset(r"SeaEval/indommlu", split='test')
    indommlu_answers = [e['answer'] for e in indommlu]
    indommlu_answers = [e.replace('(', '').replace(')', '')[:1] for e in indommlu_answers]
    indommlu_choices = [e['choices'] for e in indommlu]
    # eval_paths = ['final/llama3_infinity_firefly_crossmmlu_eval_model_responses.jsonl']
    eval_paths = ['final/qwen2_7b_infinity_firefly_full_score_filter_indommlu_eval_model_responses.jsonl']
    for eval_path in eval_paths:
        generated_answer = read_jsonl(eval_path)
        EVAL_PROMPT = """Select the most appropriate choice (A, B, C, or D) given the provided answer and choices. Only provide one answer using a single letter response, without any explanations or additional information.

# Steps

1. Review the provided answer carefully and understand its key points or meaning.
2. Analyze all choices (A, B, C, D).
3. Decide which choice best matches or fits the provided answer.
4. Respond with the most suitable choice as a single letter (A, B, C, or D).

# Output Format

Provide only the letter of the correct choice. Do not include any additional words or punctuation.

- Input Answer: %s

- Choices: 
 - %s

**Output:**
    """
        prompts = [EVAL_PROMPT % (e['answer'].strip('<|eot_id|>'), '\n - '.join(indommlu_choices[i % len(indommlu_answers)])) for i, e in enumerate(generated_answer)]
        answer_choices = async_request_questions(prompts, port=7990)
        answer_choices = [e['answer'].upper().replace('(', '').replace(')', '')[:1] for e in answer_choices]

        total_langs = 1
        chunk_size = len(generated_answer) // total_langs
        ans_detail = []
        wrong_js = []
        for i in range(0, len(generated_answer), chunk_size):
            gold_answers = indommlu_answers[i % len(indommlu_answers): (i % len(indommlu_answers)) +chunk_size]
            pred_answers = answer_choices[i: i + chunk_size]
            ans_detail.append([gold_answers[j] == pred_answers[j] for j in range(len(gold_answers))])
            wrong_js.append([j for j in range(len(ans_detail[-1])) if not ans_detail[-1][j]])
            print(i, 'avg accuracy:', np.mean(ans_detail[-1]))
        
        print('eval %s complete' % eval_path)


def extract_and_eval_crossmmlu_ans():
    crossmmlu = load_dataset(r"SeaEval/cross_mmlu", split='test')
    crossmmlu_answers = [e['English']['answer'] for e in crossmmlu] + [e['Chinese']['answer'] for e in crossmmlu] + [e['Indonesian']['answer'] for e in crossmmlu]
    crossmmlu_answers = [e.replace('(', '').replace(')', '')[:1] for e in crossmmlu_answers]
    crossmmlu_choices = [e['English']['choices'] for e in crossmmlu] + [e['Chinese']['choices'] for e in crossmmlu] + [e['Indonesian']['choices'] for e in crossmmlu]
    # eval_paths = ['final/llama3_infinity_firefly_crossmmlu_eval_model_responses.jsonl']
    eval_paths = ['final/qwen2_7b_infinity_firefly_crossmmlu_eval_model_responses.jsonl']
    for eval_path in eval_paths:
        generated_answer = read_jsonl(eval_path)
        EVAL_PROMPT = """Select the most appropriate choice (A, B, C, or D) given the provided answer and choices. Only provide one answer using a single letter response, without any explanations or additional information.

# Steps

1. Review the provided answer carefully and understand its key points or meaning.
2. Analyze all choices (A, B, C, D).
3. Decide which choice best matches or fits the provided answer.
4. Respond with the most suitable choice as a single letter (A, B, C, or D).

# Output Format

Provide only the letter of the correct choice. Do not include any additional words or punctuation.

- Input Answer: %s

- Choices: 
 - %s

**Output:**
    """
        prompts = [EVAL_PROMPT % (e['answer'].strip('<|eot_id|>'), '\n - '.join(crossmmlu_choices[i % len(crossmmlu_answers)])) for i, e in enumerate(generated_answer)]
        answer_choices = async_request_questions(prompts, port=7990)
        answer_choices = [e['answer'].upper().replace('(', '').replace(')', '')[:1] for e in answer_choices]

        total_langs = 3
        chunk_size = len(generated_answer) // total_langs
        ans_detail = []
        wrong_js = []
        for i in range(0, len(generated_answer), chunk_size):
            gold_answers = crossmmlu_answers[i % len(crossmmlu_answers): (i % len(crossmmlu_answers)) +chunk_size]
            pred_answers = answer_choices[i: i + chunk_size]
            ans_detail.append([gold_answers[j] == pred_answers[j] for j in range(len(gold_answers))])
            wrong_js.append([j for j in range(len(ans_detail[-1])) if not ans_detail[-1][j]])
            print(i, 'avg accuracy:', np.mean(ans_detail[-1]))
        
        print('eval %s complete' % eval_path)


def extract_and_eval_mmmlu_ans():
    splits = ['ar', 'bn', 'ca', 'da', 'de', 'en', 'es', 'eu', 'fr', 'gu', 'hi', 'hr', 'hu', 'hy', 'id', 'is', 'it', 'kn', 'ml', 'mr', 'nb', 'ne', 'nl', 'pt', 'ro', 'ru', 'sk', 'sr', 'sv', 'ta', 'te', 'uk', 'vi', 'zh']
    crossmmlu = [load_dataset(r"alexandrainst/m_mmlu", s) for s in splits]

    max_examples = 500
    common_ids = list(sorted(set.intersection(*[set(e['test']['id']) for e in crossmmlu])))[:max_examples]
    crossmmlu_answers = [f'{ee["answer"].upper()}' for e in crossmmlu for ee in e['test'] if ee['id'] in common_ids]
    crossmmlu_choices = [[f'({eee.upper()}): {ee["option_%s" % eee]}' for eee in ['a', 'b', 'c', 'd']] for e in crossmmlu for ee in e['test'] if ee['id'] in common_ids]
    
    eval_paths = ['final/qwen2_7b_infinity_firefly_mmmlu_eval_model_responses.jsonl']
    for eval_path in eval_paths:
        generated_answer = read_jsonl(eval_path)
        EVAL_PROMPT = """Select the most appropriate choice (A, B, C, or D) given the provided answer and choices. Only provide one answer using a single letter response, without any explanations or additional information.

# Steps

1. Review the provided answer carefully and understand its key points or meaning.
2. Analyze all choices (A, B, C, D).
3. Decide which choice best matches or fits the provided answer.
4. Respond with the most suitable choice as a single letter (A, B, C, or D).

# Output Format

Provide only the letter of the correct choice. Do not include any additional words or punctuation.

- Input Answer: %s

- Choices: 
 - %s

**Output:**
    """
        prompts = [EVAL_PROMPT % (e['answer'].strip('<|eot_id|>'), '\n - '.join(crossmmlu_choices[i % len(crossmmlu_answers)])) for i, e in enumerate(generated_answer)]
        answer_choices = async_request_questions(prompts, port=7990)
        answer_choices = [e['answer'].upper().replace('(', '').replace(')', '')[:1] for e in answer_choices]

        total_langs = len(splits)
        chunk_size = len(generated_answer) // total_langs
        ans_detail = []
        wrong_js = []
        for i in range(0, len(generated_answer), chunk_size):
            gold_answers = crossmmlu_answers[i % len(crossmmlu_answers): (i % len(crossmmlu_answers)) +chunk_size]
            pred_answers = answer_choices[i: i + chunk_size]
            ans_detail.append([gold_answers[j] == pred_answers[j] for j in range(len(gold_answers))])
            wrong_js.append([j for j in range(len(ans_detail[-1])) if not ans_detail[-1][j]])
            print(splits[i // chunk_size], 'avg accuracy:', np.mean(ans_detail[-1]))
        
        print('eval %s complete' % eval_path)


def extract_and_eval_crosslogiqa_ans():
    crosslogiqa = load_dataset(r"SeaEval/cross_logiqa", split='test')
    crosslogiqa_answers = [e['English']['answer'] for e in crosslogiqa] + [e['Chinese']['answer'] for e in crosslogiqa] + [e['Indonesian']['answer'] for e in crosslogiqa]
    crosslogiqa_answers = [e.replace('(', '').replace(')', '')[:1] for e in crosslogiqa_answers]
    crosslogiqa_choices = [e['English']['choices'] for e in crosslogiqa] + [e['Chinese']['choices'] for e in crosslogiqa] + [e['Indonesian']['choices'] for e in crosslogiqa]
    eval_paths = ['final/qwen2_7b_infinity_firefly_crosslogiqa_eval_model_responses.jsonl']
    for eval_path in eval_paths:
        generated_answer = read_jsonl(eval_path)
        EVAL_PROMPT = """Select the most appropriate choice (A, B, C, or D) given the provided answer and choices. Only provide one answer using a single letter response, without any explanations or additional information.

# Steps

1. Review the provided answer carefully and understand its key points or meaning.
2. Analyze all choices (A, B, C, D).
3. Decide which choice best matches or fits the provided answer.
4. Respond with the most suitable choice as a single letter (A, B, C, or D).

# Output Format

Provide only the letter of the correct choice. Do not include any additional words or punctuation.

- Input Answer: %s

- Choices: 
 - %s

**Output:**
    """
        prompts = [EVAL_PROMPT % (e['answer'].strip('<|eot_id|>'), '\n - '.join(crosslogiqa_choices[i % len(crosslogiqa_answers)])) for i, e in enumerate(generated_answer)]
        answer_choices = async_request_questions(prompts, port=7990)
        answer_choices = [e['answer'].upper().replace('(', '').replace(')', '')[:1] for e in answer_choices]

        total_langs = 3
        chunk_size = len(generated_answer) // total_langs
        ans_detail = []
        wrong_js = []
        for i in range(0, len(generated_answer), chunk_size):
            gold_answers = crosslogiqa_answers[i % len(crosslogiqa_answers): (i % len(crosslogiqa_answers)) +chunk_size]
            pred_answers = answer_choices[i: i + chunk_size]
            ans_detail.append([gold_answers[j] == pred_answers[j] for j in range(len(gold_answers))])
            wrong_js.append([j for j in range(len(ans_detail[-1])) if not ans_detail[-1][j]])
            print(i, 'avg accuracy:', np.mean(ans_detail[-1]))
        
        print('eval %s complete' % eval_path)


def generate_indommlu_eval_model_responses():
    import multiprocessing
    # Set the multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    indommlu = load_dataset(r"SeaEval/indommlu", split='test')
    n = len(indommlu)

    system_prompt = 'Kamu adalah asisten yang membantu.'
    all_prompts = []
    model_paths = ['/path/to/outputs/qwen2-5-7b-infinity-firefly-score-filter-sft-full/checkpoint-512']
    tokenizers = [AutoTokenizer.from_pretrained(model_path) for model_path in model_paths]
    questions = list(indommlu['question'])
    choices = list(indommlu['choices'])

    all_prompts = []
    raw_prompts = []
    for tokenizer in tokenizers:
        model_raw_prompts = []
        model_messages = []
        for i in tqdm(range(n)):
            question = questions[i]
            choice_str = '\n'.join(choices[i])
            prompt = 'Pertanyaan：%s\n\nPilihan jawaban：\n%s\n\nApa jawabannya?' % (question, choice_str)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            model_messages.append(messages)
            model_raw_prompts.append(prompt)
        try:
            model_prompts = tokenizer.apply_chat_template(model_messages, tokenize=False, add_generation_prompt=True)
        except Exception as exc:
            # Remove system roles
            model_messages_without_system_roles = [messages[1:] for messages in model_messages]
            model_prompts = tokenizer.apply_chat_template(model_messages_without_system_roles, tokenize=False, add_generation_prompt=True)
        all_prompts.append(model_prompts)
        raw_prompts.append(model_raw_prompts)

    def _postprocess_answer(ans):
        ans = ans.strip()
        if ans.startswith('The answer should be directly generated:\n'):
            ans = ans[41:]
        elif '\n\nThe final answer is' in ans:
            ans = ans[ans.rindex('\n\nThe final answer is') + 22:]
        ans = ans.strip()
        if ans.startswith(':'):
            ans = ans[1:].strip()
        for lang in ['Chinese', 'Indonesian', 'English']:
            key = 'The answer should be thinking in %s:\n' % lang
            if ans.startswith(key):
                ans = ans[len(key):].strip()
        return ans
    
    # Generation
    final_json = []
    for i, model_path in enumerate(model_paths):
        model_answers = async_request_questions(all_prompts[i], port=7991)  # generate_with_vllm_multiprocessing(all_prompts[i], model_path)
        final_json += [{'instruction': raw_prompts[i][j], 'original_answer': model_answers[j]['answer'], 'answer': _postprocess_answer(model_answers[j]['answer']), 'model': model_path} for j in range(len(all_prompts[i]))]
    write_jsonl(final_json, 'final/qwen2_7b_infinity_firefly_full_score_filter_indommlu_eval_model_responses.jsonl')


def generate_cross_alpaca_eval_model_responses():
    import multiprocessing
    # Set the multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    alpaca_eval2 = read_jsonl('final/cross_alpaca_eval2.jsonl')
    n = len(alpaca_eval2)

    system_prompt = 'You are a helpful assistant'
    all_prompts = []
    model_paths = ['/path/to/outputs/qwen2-5-7b-infinity-firefly-score-filter-sft-full/checkpoint-512']
    tokenizers = [AutoTokenizer.from_pretrained(model_path) for model_path in model_paths]

    all_prompts = []
    raw_prompts = []
    languages = ['en', 'chinese', 'indonesian']
    for tokenizer in tokenizers:
        model_raw_prompts = []
        model_messages = []
        for i in tqdm(range(n)):
            for k in languages:
                prompt = alpaca_eval2[i]['instruction_%s' % k]
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                model_messages.append(messages)
                model_raw_prompts.append(prompt)
        try:
            model_prompts = tokenizer.apply_chat_template(model_messages, tokenize=False, add_generation_prompt=True)
        except Exception as exc:
            # Remove system roles
            model_messages_without_system_roles = [messages[1:] for messages in model_messages]
            model_prompts = tokenizer.apply_chat_template(model_messages_without_system_roles, tokenize=False, add_generation_prompt=True)
        all_prompts.append(model_prompts)
        raw_prompts.append(model_raw_prompts)
    
    all_prompts = [e + ' <|answer_should_be|>' for e in model_raw_prompts]
    # Generation
    model_path = model_paths[0]
    final_json = []
    for port in list(range(7991, 7992)):
        model_answers = async_request_questions(all_prompts, port=port, max_workers=1024 * 4)  # generate_with_vllm_multiprocessing(all_prompts[i], model_path)
        final_json += [{'instruction': all_prompts[j], 'original_answer': model_answers[j]['answer'], 'answer': postprocess_mcot_answer(model_answers[j]['answer']), 'model': model_path} for j in range(len(all_prompts))]

    write_jsonl(final_json, 'final/llama3_8b_mcot_alpaca_eval.jsonl')


def generate_gpt_consistency_eval_crosslingual_openhermes():
    openai_key = ''
    model_name = 'gpt-4o'
    azure_endpoint = ''
    api_endversion = '2024-07-01-preview'
    ranking_data = read_jsonl('final/ranking_consistency_eval_crosslingual_openhermes.jsonl')
    batcher = ChatOpenAIBatcher(openai_key, model_name, azure_endpoint, api_endversion, response_format={ "type": "json_object" }, max_tokens=32768)
    for i in range(20000):
        chat_list = [{"role": "system", "content": """First evaluate model consistency of semantic mearning of english answer and chinese answer based on the given instruction, don't check accuracy of the answers, just compare whether english answer and chinese answer are consistent, after that also give a score for the overall quality of each answers, give a score from 1-10 using format {"consistent_score": "", "score_english": "", "score_chinese", ""}.
Directly output JSON without anything else."""}]
        chat_list.append({'role': 'user', 'content': str(ranking_data[i])})
        batcher.add_chat(chat_list)
    batcher.output_jsonl('data/request_ranking_consistency_eval_crosslingual_openhermes.0.jsonl')

    batcher = ChatOpenAIBatcher(openai_key, model_name, azure_endpoint, api_endversion, response_format={ "type": "json_object" }, max_tokens=32768)
    for i in range(20000, len(ranking_data)):
        chat_list = [{"role": "system", "content": """First evaluate model consistency of semantic mearning of english answer and chinese answer based on the given instruction, don't check accuracy of the answers, just compare whether english answer and chinese answer are consistent, after that also give a score for the overall quality of each answers, give a score from 1-10 using format {"consistent_score": "", "score_english": "", "score_chinese", ""}.
Directly output JSON without anything else."""}]
        chat_list.append({'role': 'user', 'content': str(ranking_data[i])})
        batcher.add_chat(chat_list)
    batcher.output_jsonl('data/request_ranking_consistency_eval_crosslingual_openhermes.1.jsonl')


def generate_gpt_eval_crosslingual_openhermes():
    simpo_data = read_jsonl('final/ranking_eval_crosslingual_openhermes.jsonl')
    openai_key = ''
    model_name = 'gpt-4o'
    azure_endpoint = ''
    api_endversion = '2024-07-01-preview'
    ranking_data = read_jsonl('final/ranking_eval_crosslingual_openhermes.jsonl')
    batcher = ChatOpenAIBatcher(openai_key, model_name, azure_endpoint, api_endversion, response_format={ "type": "json_object" }, max_tokens=32768)
    for i in range(len(simpo_data)):
        chat_list = [{"role": "system", "content": """Compare answer quality from answer 1, answer 2, answer 3 based on the given instruction, give a score from 1-10 using format {"score_1": "", "score_2": "", "score_3": ""}.
Directly output JSON without anything else."""}]
        chat_list.append({'role': 'user', 'content': str(ranking_data[i])})
        batcher.add_chat(chat_list)
    batcher.output_jsonl('data/request_ranking_eval_crosslingual_openhermes.0.jsonl')


def generate_gpt_eval_cmmlu():
    cmmlu_data = read_jsonl('final/cmmlu_eval_model_responses.jsonl')
    cmmlu = load_dataset(r"SeaEval/cmmlu", split='test')
    answers = list(map(lambda x: x[4:], list(cmmlu['answer'])))

    openai_key = ''
    model_name = 'gpt-4o'
    azure_endpoint = 'm'
    api_endversion = '2024-07-01-preview'
    batcher = ChatOpenAIBatcher(openai_key, model_name, azure_endpoint, api_endversion, response_format={ "type": "json_object" }, max_tokens=32768)
    for i in range(len(cmmlu_data)):
        cmmlu_row = cmmlu_data[i]
        cmmlu_row['given_answer'] = cmmlu_row.pop('answer')
        cmmlu_row.pop('model')
        cmmlu_row['reference_answer'] = answers[i % len(answers)]
        chat_list = [{"role": "system", "content": """对于以下instruction\n\n和reference answer, 输出1如果给定答案正确，或者0如果给定答案不正确，只要给定答案能够从语义上正确回答该问题即可，不需要考虑使用的语言，只能输出1或者0，格式如此： {"given_answer": "1 or 0"}.
直接输出JSON，不用输出其它内容"""}]
        chat_list.append({'role': 'user', 'content': str(cmmlu_row)})
        batcher.add_chat(chat_list)
    batcher.output_jsonl('data/request_eval_cmmlu.0.jsonl')
    print('done')


def generate_and_eval_mmmlu():
    splits = ['ar', 'bn', 'ca', 'da', 'de', 'en', 'es', 'eu', 'fr', 'gu', 'hi', 'hr', 'hu', 'hy', 'id', 'is', 'it', 'kn', 'ml', 'mr', 'nb', 'ne', 'nl', 'pt', 'ro', 'ru', 'sk', 'sr', 'sv', 'ta', 'te', 'uk', 'vi', 'zh']
    crossmmlu = [load_dataset(r"alexandrainst/m_mmlu", s) for s in splits]

    common_ids = list(sorted(set.intersection(*[set(e['test']['id']) for e in crossmmlu])))

    system_prompt = 'You are a helpful assistant'
    all_prompts = []
    questions = [ee['instruction'] for e in crossmmlu for ee in e['test'] if ee['id'] in common_ids]
    answers = [ee['answer'] for e in crossmmlu for ee in e['test'] if ee['id'] in common_ids]
    choices = [['A: ' + ee['option_a'], 'B: ' + ee['option_b'], 'C: ' + ee['option_c'], 'D: ' + ee['option_d']] for e in crossmmlu for ee in e['test'] if ee['id'] in common_ids]
    n = len(questions)

    model_raw_prompts = []
    model_messages = []
    for i in tqdm(range(n)):
        question = questions[i]
        choice_str = '\n'.join(choices[i])
        prompt = 'Question:%s\n\nChoices:\n%s\n\nAnswer:' % (question, choice_str)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        model_messages.append(messages)
        model_raw_prompts.append(prompt)

    answer_choices = async_request_questions(model_raw_prompts, max_workers=1024, port=30000)  # 8002

    res_data = []
    for i in range(len(answer_choices)):
        crossmmlu_row = answer_choices[i]
        crossmmlu_row['given_answer'] = postprocess_mcot_answer(crossmmlu_row.pop('answer'))
        crossmmlu_row.pop('index')
        crossmmlu_row['choices'] = choices[i]
        crossmmlu_row['reference_answer'] = answers[i]
        res_data.append(crossmmlu_row)

    EVAL_TEMPLATE = """对于以下instruction\n\n和reference answer, 输出1如果给定答案正确，或者0如果给定答案不正确，只要给定答案能够从语义上正确回答该问题即可，不需要考虑使用的语言，只能输出1或者0，格式如此： {"given_answer": "1 or 0"}. 直接输出JSON，不用输出其它内容

%s    
"""
    all_eval_prompts = []
    for i in range(len(res_data)):
        all_eval_prompts.append((EVAL_TEMPLATE % str(res_data[i])))

    eval_results = async_request_questions(all_eval_prompts, max_workers=256, port=7995)
    eval_scores = []
    wrong_count = 0
    for e in eval_results:
        try:
            eval_scores.append(int(eval(e['answer'])['given_answer']))
        except Exception as exc:
            eval_scores.append(0)
            wrong_count += 1
    n_splits = len(splits)
    each_split_size = len(eval_scores) // n_splits
    eval_scores_split = [eval_scores[i * each_split_size: i * each_split_size+each_split_size] for i in range(n_splits)]
    print(wrong_count / len(eval_results))
    for i in range(len(eval_scores_split)):
        print(i, splits[i], 'avg score:', np.mean(eval_scores_split[i]))

    print('done')


def generate_and_eval_mtruthfulqa():
    splits = ['ar', 'bn', 'ca', 'da', 'de', 'es', 'eu', 'fr', 'gu', 'hi', 'hr', 'hu', 'hy', 'id', 'it', 'kn', 'ml', 'mr', 'ne', 'nl', 'pt', 'ro', 'ru', 'sk', 'sr', 'sv', 'ta', 'te', 'uk', 'vi', 'zh']
    m_truthfulqa = [load_dataset(r"alexandrainst/m_truthfulqa", s) for s in splits]

    system_prompt = 'You are a helpful assistant'
    all_prompts = []
    questions = [ee['question'] for e in m_truthfulqa for ee in e['val']]
    answers = ['A: ' + ee['mc1_targets_choices'][0] for e in m_truthfulqa for ee in e['val']]
    choices = [[chr(ord('A') + i) + ': ' + eee for i, eee in enumerate(ee['mc1_targets_choices'])] for e in m_truthfulqa for ee in e['val']]
    n = len(questions)

    model_raw_prompts = []
    model_messages = []
    for i in tqdm(range(n)):
        question = questions[i]
        choice_str = '\n'.join(choices[i])
        prompt = 'Question:%s\n\nChoices:\n%s\n\nAnswer:' % (question, choice_str)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        model_messages.append(messages)
        model_raw_prompts.append(prompt)

    answer_choices = async_request_questions(
        model_raw_prompts,
        is_gemini=False,
        max_tokens=8192,
        max_workers=256,
        port=8000,
        model_name='meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8'
    )

    res_data = []
    for i in range(len(answer_choices)):
        crossmmlu_row = answer_choices[i]
        crossmmlu_row['given_answer'] = postprocess_mcot_answer(crossmmlu_row.pop('answer'))
        crossmmlu_row.pop('index')
        crossmmlu_row['choices'] = choices[i]
        crossmmlu_row['reference_answer'] = answers[i]
        res_data.append(crossmmlu_row)

    EVAL_TEMPLATE = """For the following instruction and reference answer, output 1 if the given answer is correct, or 0 if the given answer is incorrect. As long as the given answer semantically correctly addresses the question, it is considered correct, regardless of the language used. Only output 1 or 0 in the following format: {"given_answer": "1 or 0"}. Directly output JSON without any additional content.

%s    
"""
    all_eval_prompts = []
    for i in range(len(res_data)):
        all_eval_prompts.append((EVAL_TEMPLATE % str(res_data[i])))

    eval_results = async_request_questions(all_eval_prompts, max_tokens=600, max_workers=256, port=8000, model_name='meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8')
    eval_scores = []
    wrong_count = 0
    for e in eval_results:
        try:
            eval_scores.append(int(eval(e['answer'])['given_answer']))
        except Exception as exc:
            eval_scores.append(0)
            wrong_count += 1
    n_splits = len(splits)
    each_split_size = len(eval_scores) // n_splits
    eval_scores_split = [eval_scores[i * each_split_size: i * each_split_size+each_split_size] for i in range(n_splits)]
    print(wrong_count / len(eval_results))
    for i in range(len(eval_scores_split)):
        print(i, splits[i], 'avg score:', np.mean(eval_scores_split[i]))

    import pdb; pdb.set_trace()

    print('done')


def generate_and_eval_truthfulqa():
    splits = ['multiple_choice']
    truthfulqa = [load_dataset(r"truthfulqa/truthful_qa", s) for s in splits]

    system_prompt = 'You are a helpful assistant'
    all_prompts = []
    questions = [ee['question'] for e in truthfulqa for ee in e['validation']]
    answers = ['A: ' + ee['mc1_targets']['choices'][0] for e in truthfulqa for ee in e['validation']]
    choices = [[chr(ord('A') + i) + ': ' + eee for i, eee in enumerate(ee['mc1_targets']['choices'])] for e in truthfulqa for ee in e['validation']]
    n = len(questions)

    model_raw_prompts = []
    model_messages = []
    for i in tqdm(range(n)):
        question = questions[i]
        choice_str = '\n'.join(choices[i])
        prompt = 'Question:%s\n\nChoices:\n%s\n\nAnswer:' % (question, choice_str)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        model_messages.append(messages)
        model_raw_prompts.append(prompt)

    model_raw_prompts = [e + ' <|answer_should_be|>' for e in model_raw_prompts]
    answer_choices = async_request_questions(model_raw_prompts, max_workers=1024, port=7991)  # 8002

    res_data = []
    for i in range(len(answer_choices)):
        crossmmlu_row = answer_choices[i]
        crossmmlu_row['given_answer'] = postprocess_mcot_answer(crossmmlu_row.pop('answer'))
        crossmmlu_row.pop('index')
        crossmmlu_row['choices'] = choices[i]
        crossmmlu_row['reference_answer'] = answers[i]
        res_data.append(crossmmlu_row)

    EVAL_TEMPLATE = """For the following instruction and reference answer, output 1 if the given answer is correct, or 0 if the given answer is incorrect. As long as the given answer semantically correctly addresses the question, it is considered correct, regardless of the language used. Only output 1 or 0 in the following format: {"given_answer": "1 or 0"}. Directly output JSON without any additional content.

%s    
"""
    all_eval_prompts = []
    for i in range(len(res_data)):
        all_eval_prompts.append((EVAL_TEMPLATE % str(res_data[i])))

    eval_results = async_request_questions(all_eval_prompts, max_workers=256, port=7993)
    eval_scores = []
    wrong_count = 0
    for e in eval_results:
        try:
            eval_scores.append(int(eval(e['answer'])['given_answer']))
        except Exception as exc:
            eval_scores.append(0)
            wrong_count += 1
    n_splits = len(splits)
    each_split_size = len(eval_scores) // n_splits
    eval_scores_split = [eval_scores[i * each_split_size: i * each_split_size+each_split_size] for i in range(n_splits)]
    print(wrong_count / len(eval_results))
    for i in range(len(eval_scores_split)):
        print(i, splits[i], 'avg score:', np.mean(eval_scores_split[i]))

    import pdb; pdb.set_trace()

    print('done')


def generate_and_eval_fullmmlu():
    crossmmlu = load_dataset(r"SeaEval/mmlu", split='test')
    questions = [e['question'] for e in crossmmlu]
    answers = [e['answer'] for e in crossmmlu]
    choices = [e['choices'] for e in crossmmlu]
    n = len(questions)
    raw_prompts = []
    model_raw_prompts = []
    model_messages = []
    for i in tqdm(range(n)):
        question = questions[i]
        choice_str = '\n'.join(choices[i])
        prompt = '%s\n\n%s' % (question, choice_str)
        messages = [
            {"role": "user", "content": prompt}
        ]
        model_messages.append(messages)
        model_raw_prompts.append(prompt)

    answer_choices = async_request_questions(model_raw_prompts, max_workers=256, port=30000)  # 8002

    res_data = []
    for i in range(len(answer_choices)):
        crossmmlu_row = answer_choices[i]
        crossmmlu_row['think'] = crossmmlu_row['answer'][:crossmmlu_row['answer'].index(':')] if ':' in crossmmlu_row['answer'] else ''
        crossmmlu_row['given_answer'] = postprocess_mcot_answer(crossmmlu_row.pop('answer'))
        crossmmlu_row.pop('index')
        crossmmlu_row['choices'] = choices[i % len(answers)]
        crossmmlu_row['reference_answer'] = answers[i % len(answers)]
        res_data.append(crossmmlu_row)

    EVAL_TEMPLATE = """对于以下instruction\n\n和reference answer, 输出1如果给定答案正确，或者0如果给定答案不正确，只要给定答案能够从语义上正确回答该问题即可，不需要考虑使用的语言，只能输出1或者0，格式如此： {"given_answer": "1 or 0"}. 直接输出JSON，不用输出其它内容

%s    
"""
    all_eval_prompts = []
    for i in range(len(res_data)):
        all_eval_prompts.append((EVAL_TEMPLATE % str(res_data[i])))

    eval_results = async_request_questions(all_eval_prompts, max_workers=256, port=7993)
    eval_scores = []
    wrong_count = 0
    for e in eval_results:
        try:
            eval_scores.append(int(eval(e['answer'])['given_answer']))
        except Exception as exc:
            eval_scores.append(0)
            wrong_count += 1
    n_splits = 1
    each_split_size = len(eval_scores) // n_splits
    eval_scores_split = [eval_scores[i * each_split_size: i * each_split_size+each_split_size] for i in range(n_splits)]
    print(wrong_count / len(eval_results))
    for i in range(len(eval_scores_split)):
        print(i, 'avg score:', np.mean(eval_scores_split[i]))

    import pdb; pdb.set_trace()
    print('done')


def extract_aime_answer(text: str) -> int:
    """
    Attempt to extract an integer 0-999 from text.
    1. Try to parse JSON if present in the correct format: {"given_answer": "XXX"}
    2. If JSON parse fails or doesn't match, try to find an integer in the text via regex.
    3. If found, clamp to [0, 999]. If no integer found, return -1 as a sentinel.
    """
    # 1. Try JSON parse
    try:
        # Attempt to find the JSON object in the text
        # In some cases the LLM might return extra text, so we find the curly braces region.
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            maybe_json_str = json_match.group(0)
            data = json.loads(maybe_json_str)
            if "given_answer" in data:
                candidate = data["given_answer"]
                # Convert to integer if possible
                candidate_int = int(candidate)
                if 0 <= candidate_int <= 999:
                    return candidate_int
    except Exception:
        pass

    # 2. If that fails, fallback: parse the last integer in the text via regex
    all_ints = re.findall(r'\d+', text)
    if all_ints:
        # Take the last integer found or attempt a best guess
        candidate_int = int(all_ints[-1])
        candidate_int = max(0, min(candidate_int, 999))  # clamp to [0,999]
        return candidate_int

    # If we cannot parse any integer, return sentinel
    return -1


def generate_and_eval_aime_2024():
    # 1. Load the dataset
    aime_2024 = load_dataset("Maxwell-Jia/AIME_2024", split='train')
    questions = [e['Problem'] for e in aime_2024]
    answers = [e['Answer'] for e in aime_2024]
    n = len(questions)

    NEW_AIME_PROMPT = r"""Problem: %s

Please write your final numerical answer in the form of \boxed{xx}
"""
    model_raw_prompts = [NEW_AIME_PROMPT % q for q in questions]

    answer_choices = async_request_questions(
        model_raw_prompts,
        max_tokens=65535,
        max_workers=30,
        port=30000,
        model_name='/path/to/gemma3-12b-open-thinking-it2'
    )

    res_data = []
    gold_ans = []
    for i in range(len(answer_choices)):
        row = answer_choices[i]
        raw_llm_output = row.pop('answer')
        llm_output_cleaned = postprocess_thinking_answer(raw_llm_output)
        row['given_answer'] = llm_output_cleaned
        row.pop('index', None)
        gold_ans.append(answers[i])
        res_data.append(row)

    eval_scores = []
    wrong_count = 0

    for i, row in enumerate(res_data):
        predicted_text = row['given_answer']
        predicted_int = extract_boxed_answer(predicted_text)
        try:
            predicted_int = int(predicted_int)
        except:
            predicted_int = -1
        gold_int = int(gold_ans[i])

        is_correct = (predicted_int == gold_int)
        eval_scores.append(is_correct)
        if not is_correct:
            wrong_count += 1

    accuracy = np.mean(eval_scores)
    print(f"Total questions: {len(eval_scores)}")
    print(f"Wrong answers: {wrong_count}")
    print(f"Accuracy: {accuracy:.3f}")
    import pdb; pdb.set_trace()
    print('Done with evaluation.')


def generate_and_eval_gpqa_diamond_mcq():
    gpqa_diamond = load_dataset('hendrydong/gpqa_diamond_mc', split='test')
    questions = [e['problem'] for e in gpqa_diamond]
    answers = [e['solution'] for e in gpqa_diamond]

    model_raw_prompts = [q for q in questions]

    answer_choices = async_request_questions(
        model_raw_prompts,
        max_tokens=32768,
        max_workers=200,
        port=8000,
        model_name='/path/to/gemma3-12b-am-2560-inst'
    )

    res_data = []
    gold_ans = []
    for i in range(len(answer_choices)):
        row = answer_choices[i]
        raw_llm_output = row.pop('answer')
        llm_output_cleaned = postprocess_thinking_answer(raw_llm_output)
        row['given_answer'] = llm_output_cleaned
        row.pop('index', None)
        gold_ans.append(answers[i])
        res_data.append(row)

    eval_scores = []
    wrong_count = 0

    for i, row in enumerate(res_data):
        predicted_text = row['given_answer']
        predicted_answer = extract_boxed_answer(predicted_text).lower()
        gold_answer = gold_ans[i][-2].lower()

        is_correct = (predicted_answer == gold_answer)
        eval_scores.append(is_correct)
        if not is_correct:
            wrong_count += 1

    accuracy = np.mean(eval_scores)
    print(f"Total questions: {len(eval_scores)}")
    print(f"Wrong answers: {wrong_count}")
    print(f"Accuracy: {accuracy:.3f}")
    print('Done with evaluation.')


def generate_and_eval_bfcl_v3():
    bfcl = load_dataset(r"teddyyyy123/bfcl_v3", split='train')
    print('dataset load complete')
    all_messages = [eval(e['chat_completion_input']) for e in bfcl]
    answers = [e['ground_truth'] for e in bfcl]
    n = len(all_messages)
    model_messages = []
    for i in tqdm(range(n)):
        messages = all_messages[i]
        model_messages.append(messages)

    answer_choices = async_request_questions(
        model_messages,
        is_simple_message=False,
        max_tokens=16384,
        max_workers=64,
        port=8000,
        model_name='google/gemma-3-27b-it'
    )

    res_data = []
    for i in range(len(answer_choices)):
        crossmmlu_row = answer_choices[i]
        crossmmlu_row['given_answer'] = postprocess_thinking_answer(crossmmlu_row.pop('answer'))
        crossmmlu_row.pop('index')
        crossmmlu_row['reference_answer'] = answers[i % len(answers)]
        res_data.append(crossmmlu_row)

    EVAL_TEMPLATE = """对于以下instruction\n\n, 对比given and reference answer, 输出1如果给定答案正确，或者0如果给定答案不正确，答案必须考虑function call的每个参数，注意如果类似于"unit": ["fahrenheit", ""] 意味着fahrenheit或者不加这个unit参数都同时正确，""意味着可不接受该参数，不需要考虑使用的语言，只能输出1或者0，格式如此： {"given_answer": "1 or 0"}. 直接输出JSON，不用输出其它内容

%s    
"""
    all_eval_prompts = []
    for i in range(len(res_data)):
        all_eval_prompts.append((EVAL_TEMPLATE % str(res_data[i])))

    eval_results = async_request_questions(all_eval_prompts, max_tokens=3000, max_workers=512, port=8000, model_name='google/gemma-3-27b-it')
    eval_scores = []
    wrong_count = 0
    for e in eval_results:
        try:
            eval_scores.append(int(eval(e['answer'].replace('```json', '').replace('```', '').replace('**', '').strip())['given_answer']))
        except Exception as exc:
            print(exc, e['answer'])
            eval_scores.append(0)
            wrong_count += 1
    n_splits = 1
    each_split_size = len(eval_scores) // n_splits
    eval_scores_split = [eval_scores[i * each_split_size: i * each_split_size+each_split_size] for i in range(n_splits)]
    consistency = crosslingual_consistency(eval_scores_split)
    print(wrong_count / len(eval_results))
    for i in range(len(eval_scores_split)):
        print(i, 'avg score:', np.mean(eval_scores_split[i]))
    print('consistency:', consistency)

    import pdb; pdb.set_trace()
    print('done')


def generate_and_eval_crossmmlu():
    crossmmlu = load_dataset(r"SeaEval/cross_mmlu", split='test')
    print('dataset load complete')
    questions = [e['English']['question'] for e in crossmmlu] + [e['Chinese']['question'] for e in crossmmlu] + [e['Indonesian']['question'] for e in crossmmlu] + [e['Malay']['question'] for e in crossmmlu]
    answers = [e['English']['answer'] for e in crossmmlu] + [e['Chinese']['answer'] for e in crossmmlu] + [e['Indonesian']['answer'] for e in crossmmlu] + [e['Malay']['answer'] for e in crossmmlu]
    choices = [e['English']['choices'] for e in crossmmlu] + [e['Chinese']['choices'] for e in crossmmlu] + [e['Indonesian']['choices'] for e in crossmmlu] + [e['Malay']['choices'] for e in crossmmlu]
    n = len(questions)
    model_raw_prompts = []
    for i in tqdm(range(n)):
        question = questions[i]
        choice_str = '\n'.join(choices[i])
        prompt = '%s\n\n%s' % (question, choice_str)
        model_raw_prompts.append(prompt)

    answer_choices = async_request_questions(model_raw_prompts, max_tokens=30000, enable_thinking=False, max_workers=128, port=8000, model_name='Qwen/Qwen3-32B')

    res_data = []
    for i in range(len(answer_choices)):
        crossmmlu_row = answer_choices[i]
        crossmmlu_row['given_answer'] = postprocess_thinking_answer(crossmmlu_row.pop('answer'))
        crossmmlu_row.pop('index')
        crossmmlu_row['choices'] = choices[i % len(answers)]
        crossmmlu_row['reference_answer'] = answers[i % len(answers)]
        res_data.append(crossmmlu_row)

    EVAL_TEMPLATE = """对于以下instruction\n\n和reference answer, 输出1如果给定答案正确，或者0如果给定答案不正确，只要给定答案能够从语义上正确回答该问题即可，不需要考虑使用的语言，只能输出1或者0，格式如此： {"given_answer": "1 or 0"}. 直接输出JSON，不用输出其它内容

%s    
"""
    all_eval_prompts = []
    for i in range(len(res_data)):
        all_eval_prompts.append((EVAL_TEMPLATE % str(res_data[i])))

    eval_results = async_request_questions(all_eval_prompts, enable_thinking=False, is_gemini=False, max_tokens=600, max_workers=512, port=8000, model_name='Qwen/Qwen3-32B')
    eval_scores = []
    wrong_count = 0
    for e in eval_results:
        try:
            eval_scores.append(int(eval(e['answer'].replace('```json', '').replace('```', '').replace('**', '').strip())['given_answer']))
        except Exception as exc:
            print(exc, e['answer'])
            eval_scores.append(0)
            wrong_count += 1
    n_splits = 4
    each_split_size = len(eval_scores) // n_splits
    eval_scores_split = [eval_scores[i * each_split_size: i * each_split_size+each_split_size] for i in range(n_splits)]
    consistency = crosslingual_consistency(eval_scores_split)
    print(wrong_count / len(eval_results))
    for i in range(len(eval_scores_split)):
        print(i, 'avg score:', np.mean(eval_scores_split[i]))
    print('consistency:', consistency)

    import pdb; pdb.set_trace()
    print('done')


def generate_and_eval_crosslogiqa():
    crosslogiqa = load_dataset(r"SeaEval/cross_logiqa", split='test')
    contexts = [e['English']['context'] for e in crosslogiqa] + [e['Chinese']['context'] for e in crosslogiqa] + [e['Indonesian']['context'] for e in crosslogiqa]
    questions = [e['English']['question'] for e in crosslogiqa] + [e['Chinese']['question'] for e in crosslogiqa] + [e['Indonesian']['question'] for e in crosslogiqa]
    answers = [e['English']['answer'] for e in crosslogiqa] + [e['Chinese']['answer'] for e in crosslogiqa] + [e['Indonesian']['answer'] for e in crosslogiqa]
    choices = [e['English']['choices'] for e in crosslogiqa] + [e['Chinese']['choices'] for e in crosslogiqa] + [e['Indonesian']['choices'] for e in crosslogiqa]
    n = len(questions)
    model_raw_prompts = []
    for i in tqdm(range(n)):
        question = questions[i]
        context = contexts[i]
        choice_str = '\n'.join(choices[i])
        prompt = 'Paragraph:%s\n\nQuestion:%s\n\nChoices:\n%s' % (context, question, choice_str)
        model_raw_prompts.append(prompt)

    answer_choices = async_request_questions(model_raw_prompts, max_tokens=30000, enable_thinking=False, max_workers=128, port=8000, model_name='Qwen/Qwen3-32B')

    res_data = []
    for i in range(len(answer_choices)):
        crossmmlu_row = answer_choices[i]
        crossmmlu_row['given_answer'] = postprocess_mcot_answer(crossmmlu_row.pop('answer'))
        crossmmlu_row.pop('index')
        crossmmlu_row['choices'] = choices[i % len(answers)]
        crossmmlu_row['reference_answer'] = answers[i % len(answers)]
        res_data.append(crossmmlu_row)

    EVAL_TEMPLATE = """对于以下instruction\n\n和reference answer, 输出1如果给定答案正确，或者0如果给定答案不正确，只要给定答案能够从语义上正确回答该问题即可，不需要考虑使用的语言，只能输出1或者0，格式如此： {"given_answer": "1 or 0"}. 直接输出JSON，不用输出其它内容

%s    
"""
    all_eval_prompts = []
    for i in range(len(res_data)):
        all_eval_prompts.append((EVAL_TEMPLATE % str(res_data[i])))

    eval_results = async_request_questions(all_eval_prompts, max_tokens=600, enable_thinking=False, max_workers=256, port=8000, model_name='Qwen/Qwen3-32B')
    eval_scores = []
    wrong_count = 0
    for i, e in enumerate(eval_results):
        try:
            eval_scores.append(int(eval(e['answer'].replace('```json', '').replace('```', '').replace('**', '').strip())['given_answer']))
        except Exception as exc:
            eval_scores.append(0)
            wrong_count += 1
    
    n_splits = 3
    for i in range(len(eval_results)):
        res_data[i]['answer_is_correct_hx'] = eval_scores[i]       
    each_split_size = len(eval_scores) // n_splits
    eval_scores_split = [eval_scores[i * each_split_size: i * each_split_size+each_split_size] for i in range(n_splits)]
    consistency = crosslingual_consistency(eval_scores_split)
    print(wrong_count / len(eval_results))
    for i in range(len(eval_scores_split)):
        print(i, 'avg score:', np.mean(eval_scores_split[i]))
    print('consistency:', consistency)

    import pdb; pdb.set_trace()
    print('done')


def generate_and_eval_indommlu():
    indommlu = load_dataset(r"SeaEval/indommlu", split='test')
    questions = [e['question'] for e in indommlu]
    answers = [e['answer'] for e in indommlu]
    choices = [e['choices'] for e in indommlu]
    n = len(questions)
    model_raw_prompts = []
    for i in tqdm(range(n)):
        question = questions[i]
        choice_str = '\n'.join(choices[i])
        prompt = 'Question:%s\n\nChoices:\n%s' % (question, choice_str)
        model_raw_prompts.append(prompt)

    answer_choices = async_request_questions(model_raw_prompts, enable_thinking=False, max_tokens=12288, system_prompt='', model_name='Qwen/Qwen3-235B-A22B', max_workers=512, port=8000)

    res_data = []
    for i in range(len(answer_choices)):
        crossmmlu_row = answer_choices[i]
        crossmmlu_row['given_answer'] = postprocess_thinking_answer(crossmmlu_row.pop('answer'))
        crossmmlu_row.pop('index')
        crossmmlu_row['choices'] = choices[i % len(answers)]
        crossmmlu_row['reference_answer'] = answers[i % len(answers)]
        res_data.append(crossmmlu_row)

    EVAL_TEMPLATE = """对于以下instruction\n\n和reference answer, 输出1如果给定答案正确，或者0如果给定答案不正确，只要给定答案能够从语义上正确回答该问题即可，不需要考虑使用的语言，只能输出1或者0，格式如此： {"given_answer": "1 or 0"}. 直接输出JSON，不用输出其它内容

%s    
"""
    all_eval_prompts = []
    for i in range(len(res_data)):
        all_eval_prompts.append((EVAL_TEMPLATE % str(res_data[i])))

    eval_results = async_request_questions(all_eval_prompts, enable_thinking=False, model_name='Qwen/Qwen3-235B-A22B', max_tokens=600, max_workers=512, port=8000)
    eval_scores = []
    wrong_count = 0
    for e in eval_results:
        try:
            eval_scores.append(int(eval(e['answer'])['given_answer']))
        except Exception as exc:
            eval_scores.append(0)
            wrong_count += 1

    for i in range(len(eval_results)):
        res_data[i]['answer_is_correct_hx'] = eval_scores[i]
    n_split = 1
    each_split_size = len(eval_scores) // n_split
    eval_scores_split = [eval_scores[i * each_split_size: i * each_split_size+each_split_size] for i in range(n_split)]
    print(wrong_count / len(eval_results))
    for i in range(len(eval_scores_split)):
        print(i, 'avg score:', np.mean(eval_scores_split[i]))

    import pdb; pdb.set_trace()
    print('done')


def generate_and_eval_bmlama53():
    all_tsvs = dict()
    tags = ['en', 'zh', 'id', 'ms']
    languages = ['English', 'Chinese', 'Indonesian', 'Malay']
    for tag in tags:
        all_tsvs[tag] = pd.read_csv('/path/to/SyncInstructions/data/BMLAMA53/%s.tsv' % tag, sep='\t')

    desc_options = ['Options', '选项', 'Pilihan', 'Pilihan']
    desc_answers = ['Answer', '答案', 'Jawaban', 'Jawapan']

    REWRITE_TEMPLATE = """Rewrite the given input prompt to be an actual question using %s language, directly output the question without anything else

Input prompt: %s    
"""

    PROMPT_TEMPLATE = """%s

%s:
%s

%s:
"""

    EVAL_TEMPLATE = """Choices:
%s

Predicted Answer:
%s

Given a list of choices above, select the index of the choice based on the predicted answer. Only output the index, and nothing else.
"""
    rewritten_prompts = read_jsonl('rewritten_bmlama53.jsonl')

    count = 0

    all_prompts = []
    for i, tag in enumerate(tags):
        desc_option = desc_options[i]
        desc_answer = desc_answers[i]
        prompts = list(all_tsvs[tag]['Prompt'])
        prompt_map = Counter(prompts)

        candidate_ans = list(all_tsvs[tag]['Candidate Ans'])
        for j in range(len(prompts)):
            if prompt_map[prompts[j]] > 1:
                count += 1
                continue
            ans_list = candidate_ans[j].split(', ')
            ans_list = ['%d: %s' % (i, e) for i, e in enumerate(ans_list)]
            ans_desc = '\n'.join(ans_list)
            prompt = PROMPT_TEMPLATE % (rewritten_prompts[count].strip(), desc_option, ans_desc, desc_answer)
            all_prompts.append(prompt)
            count += 1

    all_model_answers = []
    all_tag_accs = []
    all_debug_accs = []
    all_selected_choices = []
    for port in [7990, 7991]:
        model_answers = async_request_questions(all_prompts, max_workers=128, port=port)
        all_model_answers.append(model_answers)
        eval_prompts = []
        count = 0
        for i, tag in enumerate(tags):
            prompts = list(all_tsvs[tag]['Prompt'])
            candidate_ans = list(all_tsvs[tag]['Candidate Ans'])
            prompt_map = Counter(prompts)
            for j in range(len(prompts)):
                if prompt_map[prompts[j]] > 1:
                    continue
                choice_list = candidate_ans[j].split(', ')
                choice_list = ['%d: %s' % (i, e) for i, e in enumerate(choice_list)]
                choice_desc = '\n'.join(choice_list)
                eval_prompts.append(EVAL_TEMPLATE % (choice_desc, model_answers[count]))
                count += 1

        def remove_non_digits(input_string):
            """
            Removes all non-digit characters from a string.
            
            :param input_string: The string to process.
            :return: A string containing only the digit characters from the input.
            """
            return ''.join(char for char in input_string if char.isdigit())

        selected_choices = async_request_questions(eval_prompts, max_workers=256, port=7990)
        all_selected_choices.append(selected_choices)
        selected_choices = [remove_non_digits(c['answer'])[:1] for c in selected_choices]
        count = 0
        error_count = 0
        for i, tag in enumerate(tags):
            prompts = list(all_tsvs[tag]['Prompt'])
            gold_ans_list = list(all_tsvs[tag]['Ans'])
            candidate_ans = list(all_tsvs[tag]['Candidate Ans'])
            prompt_map = Counter(prompts)
            tag_accs = []
            debug_accs = []
            for j in range(len(prompts)):
                try:
                    if prompt_map[prompts[j]] > 1:
                        continue
                    choice_list = candidate_ans[j].split(', ')
                    gold_ans = gold_ans_list[j]
                    gold_ans_index = choice_list.index(gold_ans)
                    pred_ans_index = int(selected_choices[count])
                except Exception as exc:
                    pred_ans_index = -1
                    error_count += 1
                    count += 1
                    continue
                tag_accs.append(gold_ans_index == pred_ans_index)
                debug_accs.append((count, gold_ans_index, pred_ans_index))
                count += 1
            print('%s acc:' % tag, np.mean(tag_accs))
            all_tag_accs.append(tag_accs)
            all_debug_accs.append(debug_accs)
    import pdb; pdb.set_trace()
    print('error pct:', error_count / count)
    print('------------------------------------------------------------')


if __name__ == "__main__":
    generate_and_eval_indommlu()
