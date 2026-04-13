from openai_batcher import ChatOpenAIBatcher
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import os
import random
import numpy as np
from collections import defaultdict
from utils import output_gpt_request_jsons_for_instructions, read_jsonl, write_jsonl, \
    create_batch, retrieve_batch, retrieve_batch_response, list_batch, generate_with_vllm_multiprocessing, save_jsonl_rows_to_hf_dataset, \
    postprocess_outputs_row, get_values_by_key_ranges, extract_gpt_ranking_score, postprocess_inputs_row, flatten_gpt_inputs_list, flatten_gpt_output_list, \
    submit_batches, download_batches


QC_SYSTEM_PROMPT = """Evaluate the quality of an answer in response to a corresponding question and assign a score based on specific criteria.

Consider the following points when scoring:

- **Hallucinations**: Identify any content that seems fabricated or not based on the provided information.
- **Factual Incorrectness**: Check for any factual errors by comparing against known data.
- **Repetition**: Determine if the answer unnecessarily repeats statements.
- **Instruction Adherence**: Assess if the answer follows the given instructions accurately.

# Steps

1. **Analyze the Answer**: Begin by thoroughly reading and understanding the answer provided in response to the question.
2. **Check for Hallucinations**: Identify any elements that are unrealistic, not grounded in any given or known information.
3. **Verify Factual Content**: Fact-check the statements made in the answer to ensure accuracy.
4. **Detect Repetitions**: Look for repeated phrases or sentences that do not contribute additional value.
5. **Evaluate Instruction Adherence**: Ensure the answer follows all provided instructions and guidelines.
6. **Assign a Quality Score**: Use a range (for example, 1 to 10, with 10 being the highest quality) to score the answer based on the above evaluation criteria.

# Output Format

Provide a numerical score for the answer on a standardized scale (e.g., a score from 1 to 10). Optionally, include brief feedback or notes on why this score was given.

Example format:
- Score: [Numerical Score]
- Feedback: [Short explanation of the reasoning behind the score]

# Examples

**Example 1**
- **Question**: "What is the capital of France?"
- **Answer**: "Paris is the capital of France. Paris is also known as the city of love."
- **Evaluation**: The answer is factually correct and concise, without hallucinations or unnecessary repetition.
- **Output**: 
  - Score: 9
  - Feedback: The answer is accurate, adheres to the question, and provides additional relevant information.

**Example 2**
- **Question**: "Explain photosynthesis."
- **Answer**: "Photosynthesis is a process by which plants make food. Plants use sunlight, water, and carbon dioxide to produce oxygen and glucose. It happens in the chloroplasts. Plants use sunlight, water, and carbon dioxide to produce oxygen and glucose."
- **Evaluation**: The answer provides a correct basic explanation but unnecessarily repeats information.
- **Output**: 
  - Score: 7
  - Feedback: Correct explanation but contains some repetitive statements which slightly reduce quality.

# Notes

- For answers that meet all criteria without any errors, factual inaccuracies, or repetitions, consider assigning the highest possible score.
- Answers closer to the lowest score may include multiple issues such as hallucinations, factual errors, and poor adherence to instructions.
"""

COMPARE_QC_SYSTEM_PROMT = """Evaluate the quality of multiple answers provided in response to a given instruction and reference answer by assigning a score to each based on four evaluation criteria: response hallucinations, factual incorrectness, repetition, and instruction adherence.

# Steps

1. **Check Instruction Types**:
   -  **Instruction:**:  If the instruction involves any of the following: translation, writing poems, correcting grammar, making puns or playing with words, using idiomatic expressions, homonyms, homophones, sarcasm, metaphors, or cultural references, then the output should be:

{
  "output": "the instruction can't be translated into another language for better answer",
  "reason": "[explanation for the instruction can't be translated into another language for better answer]"
}

This means that for any instruction falling into these categories, you should not provide a direct answer but instead use the specified output format, including the reason why it's not suitable for ranking and which task it's falling into.

2. **Analyze Each Answer**:
   - **Response Hallucinations**: Check if any content is invented or not backed by the reference or plausible source.
   - **Factual Incorrectness**: Identify factual errors in comparison to verifiable information or within the reference answer.
   - **Repetition**: Assess if there's unnecessary repetition that doesn't enhance the clarity or quality of the answer.
   - **Instruction Adherence**: Determine if the answer follows the instruction provided in the input.

3. **Scoring**:
   - Assign a score for each answer (`answer_1`, `answer_2`, `answer_3`) on a score rating of 0-10 based on how well it meets the criteria above.
   - Scores should reflect how effectively each answer addresses the criteria, with higher scores for better adherence and accuracy.

# Output Format

The output should be in JSON format if the instruction is suitable for ranking:
{
  "rational_1": "[rational for the score_1]",
  "score_1": "[score_for_answer_1]",
  "rational_2": "[rational for the score_2]",
  "score_2": "[score_for_answer_2]",
  "rational_3": "[rational for the score_3]",
  "score_3": "[score_for_answer_3]"
}

# Notes

- Consider the context in reference to the original instruction and reference answer to ensure that the scores reflect a fair evaluation of each response.

"""

POST_TRANSLATE_SYSTEM_PROMPT = """Translate the given answer in a JSONL from its current language to the specified target languages, structuring each translation as a JSON object.

# Steps

1. **Identify the Answer**: Extract the current 'answer' value from the JSONL input.
2. **Target Languages**: Retrieve the list of 'target_languages' from the JSONL input.
3. **Translate Answer**: For each target language, translate the 'answer' into the specified language using the corresponding language code.
4. **Construct JSON Object**: For each translation, create a key in the JSON object in the format "answer_[language_code]" where `[language_code]` is the ISO code for the language. The value will be the translated answer.

# Output Format

Provide the output as a JSON object where each key is labeled "answer_[language_code]" and contains the translated text as the value. Ensure no additional text or formatting outside the JSON object.

# Examples

**Input**: 
{
  "Instruction": "What color is the sky?",
  "answer": "Blue",
  "answer_language": "en",
  "target_languages": ["vi", "th"]
}

**Output**: 
{
  "answer_vi": "[translated_answer_to_vietnamese]",
  "answer_th": "[translated_answer_to_thai]"
}

(Note: Replace `[translated_text_to_vietnamese]` and `[translated_text_to_thai]` with actual translations. Real translations should be provided.)
"""

def generate_ranking_gpt_requests():
    instruction_answers = read_jsonl('final/llama3_8b_instruction_answers.jsonl')
    n1 = len('<|start_header_id|> user <|end_header_id|>\n')
    n2 = len('<|start_header_id|> assistant <|end_header_id|>\n')
    for i in range(len(instruction_answers)):
        instruction_answers[i]['instruction'] = instruction_answers[i]['instruction'][n1:-n2].strip()
        instruction_answers[i]['answer'] = instruction_answers[i].pop('llama3_8b')
    model_name = 'gpt-4o-mini'
    output_gpt_request_jsons_for_instructions(
        instruction_answers, 'data/request_ranking/request_ranking_llama3.jsonl',
        return_json=False, max_rows_per_file=20000, system_prompt=QC_SYSTEM_PROMPT, model_name=model_name
    )


def generate_ranking_alpaca_eval():
    instruction_answers = read_jsonl('final/llama3_cross2_dpo_full_alpaca_eval_model_responses.jsonl')
    for i in range(len(instruction_answers)):
        instruction_answers[i].pop('model')
    model_name = 'gpt-4o'
    output_gpt_request_jsons_for_instructions(
        instruction_answers, 'data/request_ranking_alpaca_eval/request_ranking_llama3_cross2_dpo.jsonl',
        return_json=False, max_rows_per_file=20000, system_prompt=QC_SYSTEM_PROMPT, model_name=model_name
    )


def generate_ranking_alpaca_eval_with_qwen2():
    instruction_answers = read_jsonl('final/llama3_cross2_dpo_full_alpaca_eval_model_responses.jsonl')
    for i in range(len(instruction_answers)):
        instruction_answers[i].pop('model')
    prompts = []
    for e in instruction_answers:
        messages = [
            {"role": "system", "content": QC_SYSTEM_PROMPT},
            {"role": "user", "content": str(e)}
        ]
        prompts.append(messages)
    model_path = 'Qwen/Qwen2.5-72B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_prompts = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
    model_answers = generate_with_vllm_multiprocessing(model_prompts, model_path, num_dp=1, tensor_parallel_size=8)
    print()


def analyse_ranking_infinity_firefly():
    request_ranking_metadata = read_jsonl('data/request_ranking_infinity_firefly_multilingual_answers/metadata_ranking_infinity_firefly_multilingual_answers.jsonl')
    request_ranking_responses = flatten_gpt_output_list('output/output_ranking_infinity_firefly_multilingual_answers/output_request_ranking_infinity_firefly_multilingual_answers_%d.jsonl', 69, 'reading infinity ranking')

    gpt_reference_answers = flatten_gpt_output_list('output/output_generating_infinity_firefly_answers/output_request_generating_infinity_firefly_answers_%d.jsonl', 35)
    gpt_reference_answere_metadata = read_jsonl('data/request_generating_infinity_firefly_answers/metadata_generating_infinity_firefly_answers.jsonl')

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
    metadata = read_jsonl('data/request_translating_llama_multilingual/metadata_llama3_8b_infinity_firefly_instruction_multilingual_answers.jsonl')
    answer_translation_responses = flatten_gpt_output_list('output/output_translating_llama_multilingual/output_request_llama3_8b_infinity_firefly_instruction_multilingual_answers_%d.jsonl', 70, 'reading answer translation')
    
    answer_translation_responses_dict = dict()
    for e in answer_translation_responses:
        try:
            answer_translation_responses_dict[e['id']] = json.loads(e['response'])
        except:
            pass

    for k in answer_translation_responses_dict:
        index, source_lang, target_lang = metadata[k]
        if 'translated_answer' in answer_translation_responses_dict[k]:
            cross_infinity_firefly_dict[index]['answer_%s_%s' % (source_lang, target_lang)] = answer_translation_responses_dict[k]['translated_answer']

    for e in gpt_reference_answers:
        index = e['id']
        response = e['response']
        cross_infinity_firefly_index = gpt_reference_answere_metadata[index]['index']
        cross_infinity_firefly_reference_answer_key = gpt_reference_answere_metadata[index]['key'].replace('instruction_', 'reference_answer_')
        cross_infinity_firefly_dict[cross_infinity_firefly_index][cross_infinity_firefly_reference_answer_key] = response

    request_ranking_dict = dict()
    for v in request_ranking_responses:
        try:
            request_ranking_dict[v['id']] = json.loads(v['response'])
        except:
            continue
        index, instruction_tag, answer_tags = \
            request_ranking_metadata[v['id']]['index'], request_ranking_metadata[v['id']]['instruction_tag'], request_ranking_metadata[v['id']]['answer_tags'],
        if all('score_%d' % i in request_ranking_dict[v['id']] for i in range(1, 4)):
            try:
                scores = list(map(int, [request_ranking_dict[v['id']]['score_%d' % i] for i in range(1, 4)]))
            except:
                continue
            for i in range(len(answer_tags)):
                cross_infinity_firefly_dict[index]['scores_%s_%s' % (answer_tags[i][0], answer_tags[i][1])] = scores[i]
                if 'rational_%d' % (i+1) in request_ranking_dict[v['id']]:
                    cross_infinity_firefly_dict[index]['rationals_%s_%s' % (answer_tags[i][0], answer_tags[i][1])] = request_ranking_dict[v['id']]['rational_%d' % (i+1)]
            cross_infinity_firefly_dict[index]['allow_ranking_%s' % instruction_tag] = True
        elif 'output' in request_ranking_dict[v['id']]:
            cross_infinity_firefly_dict[index]['allow_ranking_%s' % instruction_tag] = False
            if 'reason' in request_ranking_dict[v['id']]:
                cross_infinity_firefly_dict[index]['fail_allow_ranking_%s_reason' % instruction_tag] = request_ranking_dict[v['id']]['reason']

    json.dump(cross_infinity_firefly_dict, open('final/cross_infinity_firefly_scoring_results.json', 'w'))
    print('done')


def generate_infinity_firefly_full_cot_datasets():
    cross_infinity_firefly_dict = json.load(open('final/cross_infinity_firefly_scoring_results.json'))
    cross_infinity_firefly_dict = {int(k): v for k, v in cross_infinity_firefly_dict.items()}

    SCORE_THRESHOLD = 8

    final_ans = []
    lang_map = {
        'en': 'English',
        'id': 'Indonesian',
        'zh-cn': 'Chinese',
    }
    count_direct = 0
    count_thinking_in_x = defaultdict(int)

    for v in cross_infinity_firefly_dict.values():
        all_keys = [k[12:] for k in v if k.startswith('instruction_')]
        for each_key in all_keys:
            if ('allow_ranking_%s' % each_key) in v and not v['allow_ranking_%s' % each_key]:
                final_ans.append({
                    'prompt_id': len(final_ans),
                    'prompt': v['instruction_%s' % each_key],
                    'messages': [
                        {'content': v['instruction_%s' % each_key], 'role': 'user'},
                        {'content': 'The answer should be directly generated:\n' + v['answer_%s' % each_key], 'role': 'assistant'}
                    ]
                })
                count_direct += 1
            elif ('allow_ranking_%s' % each_key) in v and v['allow_ranking_%s' % each_key]:
                scores = {thinking_language_tag: v['scores_%s_%s' % (thinking_language_tag, each_key)] for thinking_language_tag in all_keys if ('scores_%s_%s' % (thinking_language_tag, each_key)) in v}
                scores_ranked = list(sorted(list(scores.items()), key=lambda x: x[1], reverse=True))
                if scores_ranked[0][1] < SCORE_THRESHOLD:
                    continue
                # Thinking in another language when the top score of other lanugage is greater than thinking in current language
                if scores_ranked[0][0] != each_key and scores_ranked[0][1] > scores[each_key] and ('answer_%s' % scores_ranked[0][0]) in v and ('answer_%s_%s' % (scores_ranked[0][0], each_key)) in v:
                    cot_answer = v['answer_%s' % scores_ranked[0][0]]
                    cot_translated_answer = v['answer_%s_%s' % (scores_ranked[0][0], each_key)]
                    if not isinstance(cot_translated_answer, str):
                        continue
                    final_ans.append({
                        'prompt_id': len(final_ans),
                        'prompt': v['instruction_%s' % each_key],
                        'messages': [
                            {'content': v['instruction_%s' % each_key], 'role': 'user'},
                            {'content': f'The answer should be thinking in {lang_map.get(scores_ranked[0][0], scores_ranked[0][0])}:\n{cot_answer.strip()}\n\nThe final answer is: {cot_translated_answer.strip()}', 'role': 'assistant'}
                        ]
                    })
                    count_thinking_in_x[scores_ranked[0][0]] += 1
                elif scores_ranked[0][0] == each_key:
                    # Directly in thinking in current language otherwise
                    final_ans.append({
                        'prompt_id': len(final_ans),
                        'prompt': v['instruction_%s' % each_key],
                        'messages': [
                            {'content': v['instruction_%s' % each_key], 'role': 'user'},
                            {'content': 'The answer should be directly generated:\n' + v['answer_%s' % each_key], 'role': 'assistant'}
                        ]
                    })
                    count_direct += 1
    random.seed(42)
    random.shuffle(final_ans)
    save_jsonl_rows_to_hf_dataset(final_ans, 'sft_infinity_firefly_datasets_score_filtered')
    print(count_direct)
    for k, v in count_thinking_in_x.items():
        print(k, ':', v)
    print('done')


def analyse_ranking_alpaca_eval():
    instruction_answers = read_jsonl('final/llama3_cross2_dpo_full_alpaca_eval_model_responses.jsonl')
    n = len(instruction_answers)
    total_models = 4
    each_model_n = n // total_models
    models = ['ours-llama3-cross2-dpo-%d' % i for i in range(1, 5)] #['gpt-aligned-llama3.1-8b-inst']# ['llama3.1-8b-inst', 'filtered-self-aligned-llama3.1-8b-inst']
    languages = ['en', 'zh', 'id', 'th', 'vi']
    ranking_output = read_jsonl('output/output_ranking_alpaca_eval/output_request_ranking_llama3_cross2_dpo_%d.jsonl' % 0)
    ranking_output = [postprocess_outputs_row(e) for e in ranking_output]
    ranking_output = {e['id']: extract_gpt_ranking_score(e['response']) for e in ranking_output}

    model_rankings = [{lang: [] for lang in languages} for _ in range(total_models)]
    model_index = 0
    tmp = each_model_n
    for k, v in ranking_output.items():
        if k >= tmp:
            model_index += 1
            tmp += each_model_n
        lang = languages[k % len(languages)]
        if v['score'] and 0 <= v['score'] <= 10:
            model_rankings[model_index][lang].append(v['score'])
    for i in range(total_models):
        for lang in languages:
            print('model (%s) has mean score of %.2f in %s' % (models[i], np.mean(model_rankings[i][lang]), lang))
        print()

def submit_file(idx):
    openai_key = ''
    model_name = 'gpt-4o'
    azure_endpoint = ''
    api_endversion = '2024-07-01-preview'
    batcher = ChatOpenAIBatcher(openai_key, model_name, azure_endpoint, api_endversion, response_format=None)
    return batcher.submit_file('data/request_ranking_infinity_firefly_multilingual_answers/request_ranking_infinity_firefly_multilingual_answers_%d.jsonl' % (idx)).id


def analyse_llama_ranking_for_preference_learning():
    n_langauges = 5
    # requests = [read_jsonl('data/request_ranking/request_ranking_llama3_%d.jsonl' % (idx)) for idx in tqdm(range(63), desc='reading requests')]
    gpt_ranking_responses = [read_jsonl('output/output_ranking_llama3/output_ranking_llama3_%d.jsonl' % (idx)) for idx in tqdm(range(63), desc='reading responses')]
    # requests = [ee for e in requests for ee in e]
    gpt_ranking_responses = [postprocess_outputs_row(ee) for e in gpt_ranking_responses for ee in e]
    gpt_ranking_responses = {e['id']: e['response'] for e in gpt_ranking_responses}

    n_request_crossopenhermes = 993250
    n_request_reasoning = 251590
    # requests_reasoning = requests[:n_request_reasoning]
    # requests_crossopenhermes = requests[n_request_reasoning:]

    ranking_reasoning = get_values_by_key_ranges(gpt_ranking_responses, 0, n_request_reasoning)
    ranking_crossopenhermes = get_values_by_key_ranges(gpt_ranking_responses, n_request_reasoning, n_request_crossopenhermes+n_request_reasoning)
    
    consistency_reasoning_results = read_jsonl('final/consistency_reasoning.jsonl')
    consistency_crossopenhermes_results = read_jsonl('final/consistency_crossopenhermes.jsonl')

    # Filter out inconsistent queries.
    ranking_reasoning = {k: v for k, v in ranking_reasoning.items() if consistency_reasoning_results[k // n_langauges]['consistency']}
    ranking_crossopenhermes = {k: v for k, v in ranking_crossopenhermes.items() if consistency_crossopenhermes_results[(k - n_request_reasoning) // n_langauges]['consistency']}
    
    ranking_reasoning_languages = [get_values_by_key_ranges(ranking_reasoning, i, i + (n_request_reasoning // n_langauges)) \
                                   for i in range(0, n_request_reasoning - (n_request_reasoning // n_langauges) + 1, n_request_reasoning // n_langauges)]
    
    ranking_crossopenhermes_langauges = [[] for _ in range(n_langauges)]
    for i in range(n_request_reasoning, n_request_reasoning + n_request_crossopenhermes):
        if i in ranking_crossopenhermes:
            ranking_crossopenhermes_langauges[(i - n_request_reasoning) % 5].append(ranking_crossopenhermes[i])
    
    ranking_reasoning_languages = [list(filter(lambda x: x and 0 <= x <= 10, map(extract_gpt_ranking_score, e.values()))) for e in ranking_reasoning_languages]
    ranking_crossopenhermes_langauges = [list(filter(lambda x: x and 0 <= x <= 10, map(extract_gpt_ranking_score, e))) for e in ranking_crossopenhermes_langauges]
    languages = ['en', 'zh', 'id', 'th', 'vi']
    
    for i in range(n_langauges):
        print('langauge (%s) has mean score of %.2f for reasoning (#%d)' % (languages[i], np.mean(ranking_reasoning_languages[i]), len(ranking_reasoning_languages[i])))
        print('langauge (%s) has mean score of %.2f for crossopenhermes (#%d)' % (languages[i], np.mean(ranking_crossopenhermes_langauges[i]), len(ranking_crossopenhermes_langauges[i])))

    for i in range(n_langauges):
        for score in range(10, 5, -1):
            print('langauge (%s) has score >= %d : %.2f%%' % (languages[i], score, 100 * len([e for e in ranking_reasoning_languages[i] if e >= score]) / len(ranking_reasoning_languages[i])))


def generate_llama3_responses_ranking_requests_for_infinity_firefly():
    gpt_reference_answers = flatten_gpt_output_list('output/output_generating_infinity_firefly_answers/output_request_generating_infinity_firefly_answers_%d.jsonl', 35)
    gpt_reference_answere_metadata = read_jsonl('data/request_generating_infinity_firefly_answers/metadata_generating_infinity_firefly_answers.jsonl')

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
    metadata = read_jsonl('data/request_translating_llama_multilingual/metadata_llama3_8b_infinity_firefly_instruction_multilingual_answers.jsonl')
    answer_translation_responses = flatten_gpt_output_list('output/output_translating_llama_multilingual/output_request_llama3_8b_infinity_firefly_instruction_multilingual_answers_%d.jsonl', 70, 'reading answer translation')
    
    answer_translation_responses_dict = dict()
    for e in answer_translation_responses:
        try:
            answer_translation_responses_dict[e['id']] = json.loads(e['response'])
        except:
            pass

    for k in answer_translation_responses_dict:
        index, source_lang, target_lang = metadata[k]
        if 'translated_answer' in answer_translation_responses_dict[k]:
            cross_infinity_firefly_dict[index]['answer_%s_%s' % (source_lang, target_lang)] = answer_translation_responses_dict[k]['translated_answer']

    for e in gpt_reference_answers:
        index = e['id']
        response = e['response']
        cross_infinity_firefly_index = gpt_reference_answere_metadata[index]['index']
        cross_infinity_firefly_reference_answer_key = gpt_reference_answere_metadata[index]['key'].replace('instruction_', 'reference_answer_')
        cross_infinity_firefly_dict[cross_infinity_firefly_index][cross_infinity_firefly_reference_answer_key] = response
    
    gpt_requests = []
    gpt_requests_metadata = []
    for k, v in cross_infinity_firefly_dict.items():
        lang_tags = {e[12:] for e in v if e.startswith('instruction_')}
        for tag in lang_tags:
            instruction = v['instruction_%s' % tag]
            if 'reference_answer_%s' % tag not in v:
                continue
            reference_answer = v['reference_answer_%s' % tag]
            if 'answer_%s' % tag not in v:
                continue
            all_answers = [v['answer_%s' % tag]]
            answer_tags = [(tag, tag)]
            flag_continue = False
            for another_tag in lang_tags:
                if another_tag == tag:
                    continue
                if 'answer_%s_%s' % (another_tag, tag) not in v:
                    flag_continue = True
                    break
                all_answers.append(v['answer_%s_%s' % (another_tag, tag)])
                answer_tags.append((another_tag, tag))
            if flag_continue:
                continue
            body = {'instruction': instruction, 'reference_answer': reference_answer}
            for i, a in enumerate(all_answers):
                body['answer_%d' % (i + 1)] = a
            gpt_requests_metadata.append({'index': k, 'instruction_tag': tag, 'answer_tags': answer_tags})
            gpt_requests.append(body)
    
    output_gpt_request_jsons_for_instructions(gpt_requests, 
                                              'data/request_ranking_infinity_firefly_multilingual_answers/request_ranking_infinity_firefly_multilingual_answers.jsonl',
                                              return_json=True, max_tokens=16384, system_prompt=COMPARE_QC_SYSTEM_PROMT, model_name='gpt-4o', max_rows_per_file=10000)
    write_jsonl(gpt_requests_metadata, 'data/request_ranking_infinity_firefly_multilingual_answers/metadata_ranking_infinity_firefly_multilingual_answers.jsonl')
    print('done')


def generate_gpt_responses_for_cross_infinity_firefly():
    cross_instructions = read_jsonl('final/cross_infinity_firefly_instructions.jsonl')
    keys_instructions = [(i, k, e[k]) for i, e in enumerate(cross_instructions) for k in e]
    keys = list(map(lambda x: {'index': x[0], 'key': x[1]}, keys_instructions))
    instructions = list(map(lambda x: x[2], keys_instructions))

    output_gpt_request_jsons_for_instructions(instructions, 'data/request_generating_infinity_firefly_answers/request_generating_infinity_firefly_answers.jsonl')
    write_jsonl(keys, 'data/request_generating_infinity_firefly_answers/metadata_generating_infinity_firefly_answers.jsonl')


def generate_llama3_scores():
    n_langauges = 5
    requests = [read_jsonl('data/request_ranking/request_ranking_llama3_%d.jsonl' % (idx)) for idx in tqdm(range(63), desc='reading requests')]
    gpt_ranking_responses = [read_jsonl('output/output_ranking_llama3/output_ranking_llama3_%d.jsonl' % (idx)) for idx in tqdm(range(63), desc='reading responses')]
    requests = [ee for e in requests for ee in e]
    gpt_ranking_responses = [postprocess_outputs_row(ee) for e in gpt_ranking_responses for ee in e]
    gpt_ranking_responses = {e['id']: e['response'] for e in gpt_ranking_responses}

    n_request_crossopenhermes = 993250
    n_request_reasoning = 251590
    requests_reasoning = requests[:n_request_reasoning]
    requests_crossopenhermes = requests[n_request_reasoning:]
    requests_reasoning = list(map(postprocess_inputs_row, requests_reasoning))
    requests_crossopenhermes = list(map(postprocess_inputs_row, requests_crossopenhermes))
    requests_reasoning = [eval(e['response']) for e in requests_reasoning]
    requests_crossopenhermes = [eval(e['response']) for e in requests_crossopenhermes]

    requests_reasoning = list(zip(*[requests_reasoning[i: i + (n_request_reasoning // n_langauges)] \
                          for i in range(0, n_request_reasoning - (n_request_reasoning // n_langauges) + 1, n_request_reasoning // n_langauges)]))
    requests_crossopenhermes = [requests_crossopenhermes[i: i+5] for i in range(0, n_request_crossopenhermes, 5)]

    ranking_reasoning = get_values_by_key_ranges(gpt_ranking_responses, 0, n_request_reasoning)
    ranking_crossopenhermes = get_values_by_key_ranges(gpt_ranking_responses, n_request_reasoning, n_request_crossopenhermes+n_request_reasoning)

    ranking_reasoning_languages = [get_values_by_key_ranges(ranking_reasoning, i, i + (n_request_reasoning // n_langauges)) \
                                   for i in range(0, n_request_reasoning - (n_request_reasoning // n_langauges) + 1, n_request_reasoning // n_langauges)]
    new_ranking_reasoning = dict()
    for i in range(n_request_reasoning):
        language_split = i % n_langauges
        row = (i // n_langauges) + language_split * (n_request_reasoning // 5)
        if row in ranking_reasoning_languages[language_split]:
            new_ranking_reasoning[i] = ranking_reasoning_languages[language_split][row]
    ranking_reasoning = new_ranking_reasoning
    
    consistency_reasoning_results = read_jsonl('final/consistency_reasoning.jsonl')
    consistency_crossopenhermes_results = read_jsonl('final/consistency_crossopenhermes.jsonl')

    # Filter out inconsistent queries.
    ranking_reasoning = {k: v for k, v in ranking_reasoning.items() if consistency_reasoning_results[k // n_langauges]['consistency']}
    ranking_crossopenhermes = {k: v for k, v in ranking_crossopenhermes.items() if consistency_crossopenhermes_results[(k - n_request_reasoning) // n_langauges]['consistency']}

    scores_reasoning = [[{'score': None} for _ in range(n_langauges)] for _ in range(n_request_reasoning // n_langauges)]
    scores_crossopenhermes = [[{'score': None} for _ in range(n_langauges)] for _ in range(n_request_crossopenhermes // n_langauges)]
    for k, v in ranking_reasoning.items():
        score = extract_gpt_ranking_score(v)
        if score and score['score'] and 0 <= score['score'] <= 10:
            scores_reasoning[k // 5][k % 5] = score
    for k, v in ranking_crossopenhermes.items():
        score = extract_gpt_ranking_score(v)
        k = k - n_request_reasoning
        if score and score['score'] and 0 <= score['score'] <= 10:
            scores_crossopenhermes[k // 5][k % 5] = score
    
    for i in range(len(scores_reasoning)):
        for j in range(n_langauges):
            if scores_reasoning[i][j]:
                scores_reasoning[i][j].update(requests_reasoning[i][j])
    for i in range(len(scores_crossopenhermes)):
        for j in range(n_langauges):
            if scores_crossopenhermes[i][j]:
                scores_crossopenhermes[i][j].update(requests_crossopenhermes[i][j])
    write_jsonl(scores_reasoning, 'final/llama3_8b_reasoning_scores.jsonl')
    write_jsonl(scores_crossopenhermes, 'final/llama3_8b_crossopenhermes_scores.jsonl')


def generate_post_translate_requests_for_perference_learning():
    """
    Generate the translation requests after obtaining the reward scores.
    Also sample the pairs for perference learning based on a strategy.
    """
    scores_reasoning = read_jsonl('final/llama3_8b_reasoning_scores.jsonl')
    scores_crossopenhermes = read_jsonl('final/llama3_8b_crossopenhermes_scores.jsonl')

    # output_gpt_request_jsons_for_instructions(
    #     instruction_answers, 'data/request_ranking/request_ranking_llama3.jsonl',
    #     return_json=False, max_rows_per_file=20000, system_prompt=QC_SYSTEM_PROMPT, model_name=model_name
    # )
    max_score_margin = 9
    score_diff_margin = 2
    prompts = [[] for _ in range(2)]
    languages = ['en', 'zh', 'id', 'th', 'vi']

    for data_index, scores in enumerate([scores_reasoning, scores_crossopenhermes]):
        for i in range(len(scores)):
            max_score = 0
            max_language = 0
            for j in range(len(scores[i])):
                if scores[i][j]['score'] and scores[i][j]['score'] > max_score:
                    max_language = j
                    max_score = scores[i][j]['score']
            target_languages = []
            if max_score >= max_score_margin:
                for j in range(len(scores[i])):
                    if scores[i][j]['score'] is None:
                        continue
                    if j != max_language and max_score - scores[i][j]['score'] >= score_diff_margin:
                        target_languages.append(j)
                if target_languages:
                    prompts[data_index].append({'index': i, 'instruction': scores[i][max_language]['instruction'], 'answer': scores[i][max_language]['answer'], 'answer_language': languages[max_language], 'target_languages': str([languages[j] for j in target_languages])})
    
    model_name = 'gpt-4o-mini'
    output_gpt_request_jsons_for_instructions(
        prompts[0], 'data/request_post_translate/reasoning/request_post_translate_reasoning.jsonl',
        return_json=True, max_rows_per_file=20000, system_prompt=POST_TRANSLATE_SYSTEM_PROMPT, model_name=model_name, max_tokens=16384
    )
    output_gpt_request_jsons_for_instructions(
        prompts[1], 'data/request_post_translate/crossopenhermes/request_post_translate_crossopenhermes.jsonl',
        return_json=True, max_rows_per_file=20000, system_prompt=POST_TRANSLATE_SYSTEM_PROMPT, model_name=model_name, max_tokens=16384
    )
    print('done')


def submit_post_translation_reasoning_file(idx):
    openai_key = ''
    model_name = 'gpt-4o-mini'
    azure_endpoint = ''
    api_endversion = '2024-07-01-preview'
    max_tokens = 16384
    batcher = ChatOpenAIBatcher(openai_key, model_name, azure_endpoint, api_endversion, response_format=None, max_tokens=max_tokens)
    return batcher.submit_file('data/request_post_translate/reasoning/request_post_translate_reasoning_%d.jsonl' % (idx)).id


def submit_post_translation_crossopenhermes_file(idx):
    openai_key = ''
    model_name = 'gpt-4o-mini'
    azure_endpoint = ''
    api_endversion = '2024-07-01-preview'
    max_tokens = 16384
    batcher = ChatOpenAIBatcher(openai_key, model_name, azure_endpoint, api_endversion, response_format=None, max_tokens=max_tokens)
    return batcher.submit_file('data/request_post_translate/crossopenhermes/request_post_translate_crossopenhermes_%d.jsonl' % (idx)).id



if __name__ == '__main__':
    generate_ranking_alpaca_eval_with_qwen2()
    
