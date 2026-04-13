from openai_batcher import ChatOpenAIBatcher
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import os
import argparse
import random
from utils import output_gpt_request_jsons_for_instructions, read_jsonl, write_jsonl, \
    create_batch, retrieve_batch, retrieve_batch_response, list_batch, generate_with_vllm_multiprocessing, save_jsonl_rows_to_hf_dataset, async_request_questions


def _postprocess_outputs_row(row):
    assistant_response = row['response']['body']['choices'][0]['message'].get('content', '')
    return {'id': int(row['custom_id'][8:]), 'response': assistant_response}


def _postprocess_inputs_row(row):
    assistant_response = row['body']['messages'][1]['content']
    return {'id': int(row['custom_id'][8:]), 'response': assistant_response}


def _postprocess_translation_responses(response):
    if '0' in response:
        response = [response[k] for k in response]
    elif 'value' not in response:
        response = response[list(response.keys())[0]]
    messages = [e for e in response if e and isinstance(e, dict) and 'from' in e and 'value' in e]
    is_user_first_turn = messages[0]['from'] == 'human'
    convs = []
    if not is_user_first_turn:
        convs.append({'human': '', 'assistant': messages[0]['value']})
        messages = messages[1:]
    for idx in range(0, len(messages), 2):
        convs.append({'human': messages[idx]['value'], 'assistant': messages[idx+1]['value']})
    return convs


def _generate_reasoning_datasets_requests():
    reasoning_datasets = read_jsonl('./final/reasoning_crosslingual_llama_result.jsonl')
    instruction_english = list(map(lambda x: x['instruction_english'], reasoning_datasets))
    instruction_chinese = list(map(lambda x: x['instruction_chinese'], reasoning_datasets))
    instruction_indonesian = list(map(lambda x: x['instruction_indonesian'], reasoning_datasets))
    instruction_thai = list(map(lambda x: x['instruction_thai'], reasoning_datasets))
    instruction_vietnamese = list(map(lambda x: x['instruction_vietnamese'], reasoning_datasets))

    all_instructions = instruction_english + instruction_chinese + instruction_indonesian + instruction_thai + instruction_vietnamese
    output_gpt_request_jsons_for_instructions(
        all_instructions, 'output/request_reasoning_instructions.jsonl',
        return_json=False, max_rows_per_file=20000
    )


def _generate_crossopenhermes_datasets():
    input_translates = [read_jsonl('data/translate_openhermes_zh.%d.jsonl' % i) for i in range(5)]
    input_translates = [list(map(_postprocess_inputs_row, e)) for e in input_translates]
    output_translates = []
    for tag in ['translate_openhermes_zh_output.{}.jsonl', 
                'output_translate_openhermes_id.{}.jsonl',
                'output_translate_openhermes_th.{}.jsonl',
                'output_translate_openhermes_vi.{}.jsonl']:  # Added missing comma
        data_list = [read_jsonl('output/{}'.format(tag.format(i))) for i in range(5)]
        data_list = [list(map(_postprocess_outputs_row, e)) for e in data_list]
        data_list = [{e['id']: e['response'] for e in l} for l in data_list]
        output_translates.append(data_list)
    total_splits = len(input_translates)
    all_instructions = []
    index = 0
    error_count = 0
    for split in range(total_splits):
        for i in tqdm(range(len(input_translates[split])), desc='Processing split(%d)' % split):
            if all(i in output_translates[j][split] for j in range(4)):
                english_responses = input_translates[split][i]['response']
                other_responses = [output_translates[j][split][i] for j in range(4)]
                try:
                    other_responses = [_postprocess_translation_responses(json.loads(e)) for e in other_responses]
                    english_responses = eval(english_responses)
                except:
                    error_count += 1
                    continue
                all_instructions.append({'instruction': english_responses[0]['value'], 'index': index, 'language': 'en'})
                all_instructions.append({'instruction': other_responses[0][0]['human'], 'index': index, 'language': 'zh'})
                all_instructions.append({'instruction': other_responses[1][0]['human'], 'index': index, 'language': 'id'})
                all_instructions.append({'instruction': other_responses[2][0]['human'], 'index': index, 'language': 'th'})
                all_instructions.append({'instruction': other_responses[3][0]['human'], 'index': index, 'language': 'vi'})
                index += 1
    print('error pct:', (error_count / len(all_instructions)))
    write_jsonl(all_instructions, 'final/crosslingual_openhermes_5_languages_instructions.jsonl')


def _generate_crossopenhermes_requests():
    datasets = read_jsonl('final/crosslingual_openhermes_5_languages_instructions.jsonl')
    instructions = list(map(lambda x: x['instruction'], datasets))
    output_gpt_request_jsons_for_instructions(
        instructions, 'output/crossopenhermes_requests/request_crossopenhermes.jsonl',
        return_json=False, max_rows_per_file=20000
    )


def _generate_answers_via_models():
    request_reasoning_instructions = [read_jsonl('output/reasoning_datasets_requests/request_reasoning_instructions_%d.jsonl' % i) for i in tqdm(range(13), desc='reading reasoning dataset')]
    request_crossopenherms = [read_jsonl('output/crossopenhermes_requests/request_crossopenhermes_%d.jsonl' % i) for i in tqdm(range(50), desc='reading crossopenhermes dataset')]

    all_flattened = [ee for e in request_reasoning_instructions for ee in e] + [ee for e in request_crossopenherms for ee in e]
    all_flattened = list(map(_postprocess_inputs_row, all_flattened))

    all_instructions = list(map(lambda x: '<|start_header_id|> user <|end_header_id|>\n' + x['response'] + '<|start_header_id|> assistant <|end_header_id|>\n', all_flattened))

    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"

    all_answers = generate_with_vllm_multiprocessing(all_instructions, model_name, num_gpus=2, tensor_parallel_size=4)

    final_ans = []
    for i in range(len(all_instructions)):
        final_ans.append({'instruction': all_instructions[i], 'llama3_8b': all_answers[i]})
    write_jsonl(final_ans, 'final/llama3_70b_instruction_answers.jsonl')


def generate_answers_infinity_firefly():
    cross_instructions = read_jsonl('final/cross_infinity_firefly_instructions.jsonl')
    keys = [(i, k) for i, e in enumerate(cross_instructions) for k in e]
    instructions = [str(cross_instructions[i][k]) for i, k in keys]

    # all_instructions = list(map(lambda x: '<|start_header_id|> user <|end_header_id|>\n' + x + '<|start_header_id|> assistant <|end_header_id|>\n', instructions))

    # model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"

    # all_answers = generate_with_vllm_multiprocessing(all_instructions, model_name, num_gpus=8, tensor_parallel_size=1)
    all_answers = async_request_questions(instructions)

    final_ans = []
    for i in range(len(instructions)):
        final_ans.append({'index': keys[i][0], 'instruction_%s' % keys[i][1]: instructions[i], 'answer': all_answers[i]['answer']})
    write_jsonl(final_ans, 'final/qwen2_5_7b_infinity_firefly_instruction_answers.jsonl')


def _generate_preference_pairs_from_all_instructions():
    gpt_answers_reasoning_instructions = [list(map(_postprocess_outputs_row, read_jsonl('output/crossopenhermes_reasoning_outputs/output_request_reasoning_instructions_%d.jsonl' % i))) for i in tqdm(range(13), desc='reading reasoning dataset')]
    gpt_answers_crossopenherms = [list(map(_postprocess_outputs_row, read_jsonl('output/crossopenhermes_reasoning_outputs/output_request_crossopenhermes_%d.jsonl' % i))) for i in tqdm(range(50), desc='reading crossopenhermes dataset')]

    gpt_answers_reasoning_instructions = {ee['id']: ee['response'] for e in gpt_answers_reasoning_instructions for ee in e}
    gpt_answers_crossopenherms = {ee['id']: ee['response'] for e in gpt_answers_crossopenherms for ee in e}

    n_request_crossopenhermes = 993250
    n_request_reasoning = 251590

    instruction_answers = read_jsonl('final/llama3_8b_instruction_answers.jsonl')
    instruction_answers_reasoning = instruction_answers[:n_request_reasoning]
    instruction_answers_crossopenhermes = instruction_answers[n_request_reasoning:]

    valid_idx_reasoning = [i for i in range(len(instruction_answers_reasoning)) if i in gpt_answers_reasoning_instructions]
    valid_idx_crossopenhermes = [i for i in range(len(instruction_answers_crossopenhermes)) if i in gpt_answers_crossopenherms]

    n1 = len('<|start_header_id|> user <|end_header_id|>\n')
    n2 = len('<|start_header_id|> assistant <|end_header_id|>\n')
    final_ans = []
    index = 0
    pairs = [(valid_idx_reasoning, instruction_answers_reasoning, gpt_answers_reasoning_instructions), (valid_idx_crossopenhermes, instruction_answers_crossopenhermes, gpt_answers_crossopenherms)]
    for valid_idx, inst_ans, gpt_ans in pairs:
        for i in valid_idx:
            final_ans.append({'prompt_id': index, 'prompt': inst_ans[i]['instruction'][n1:-n2].strip(), 
                                        'all_generated_responses': [gpt_ans[i], inst_ans[i]['llama3_8b']],
                                        'all_rm_scores': [10.0, 0.0], 'chosen': [{"content": inst_ans[i]['instruction'][n1:-n2].strip(), "role": "user"},
                                                                                {"content": gpt_ans[i], "role": "assistant"}],
                                                                    'rejected': [{"content": inst_ans[i]['instruction'][n1:-n2].strip(), "role": "user"},
                                                                                {"content": inst_ans[i]['llama3_8b'], "role": "assistant"}]})
            index += 1
    random.seed(42)
    random.shuffle(final_ans)
    save_jsonl_rows_to_hf_dataset(final_ans, 'simpo_crossopenhermes_reasoning_full_datasets')


def submit_file(idx, type):
    openai_key = ''
    model_name = 'gpt-4o'
    azure_endpoint = ''
    api_endversion = '2024-07-01-preview'
    batcher = ChatOpenAIBatcher(openai_key, model_name, azure_endpoint, api_endversion, response_format=None)
    if type == 'reasoning':
        return batcher.submit_file('output/reasoning_datasets_requests/request_reasoning_instructions_%d.jsonl' % (idx)).id
    elif type == 'crossopenhermes':
        return batcher.submit_file('output/crossopenhermes_requests/request_crossopenhermes_%d.jsonl' % (idx)).id


def shard_infinity_gen_r1_datasets():
    # Load the dataset from Hugging Face Hub
    dataset = load_dataset('BAAI/Infinity-Instruct', 'Gen', split='train')
    
    # Number of shards you want to create
    num_shards = 8

    # Loop over each shard index
    for idx in range(num_shards):
        # Create a shard of the dataset
        shard = dataset.shard(num_shards=num_shards, index=idx)
        
        # Define the output path with the given naming convention
        output_path = f"./SyncInstructions/infinity_instruct_gen/Infinity-Instruct-{idx}"
        
        # Save the shard to disk
        shard.save_to_disk(output_path)
        print(f"Saved shard {idx} to {output_path}")


def shard_open_r1_datasets():
    # Load the dataset from Hugging Face Hub
    dataset = load_dataset('open-r1/OpenR1-Math-220k', 'default', split='train')
    
    # Number of shards you want to create
    num_shards = 8

    # Loop over each shard index
    for idx in range(num_shards):
        # Create a shard of the dataset
        shard = dataset.shard(num_shards=num_shards, index=idx)
        
        # Define the output path with the given naming convention
        output_path = f"./SyncInstructions/open_r1_math/OpenR1-Math-220k-{idx}"
        
        # Save the shard to disk
        shard.save_to_disk(output_path)
        print(f"Saved shard {idx} to {output_path}")


def shard_kodcode_v1_datasets():
    # Load the dataset from Hugging Face Hub
    dataset = load_dataset('KodCode/KodCode-V1-SFT-R1', split='train')
    
    # Number of shards you want to create
    num_shards = 8

    # Loop over each shard index
    for idx in range(num_shards):
        # Create a shard of the dataset
        shard = dataset.shard(num_shards=num_shards, index=idx)
        
        # Define the output path with the given naming convention
        output_path = f"./SyncInstructions/kodcode_v1/KodCode-V1-SFT-R1-{idx}"
        
        # Save the shard to disk
        shard.save_to_disk(output_path)
        print(f"Saved shard {idx} to {output_path}")


def generate_infinity_gen_r1_datasets(indices):
    def convert_role(original_role: str) -> str:
        """
        Convert the original role from the dataset to one of:
        "user", "assistant", or "system".
        In particular, convert "human" to "user" and "gpt" to "assistant".
        """
        role_lower = original_role.lower().strip()
        if role_lower == 'human':
            return 'user'
        elif role_lower in ['gpt', 'assistant']:
            return 'assistant'
        elif role_lower == 'system':
            return 'system'
        else:
            # If an unexpected role is encountered, default to "user".
            return 'user'

    def process_conversation(conversation: list) -> list:
        """
        Process a single conversation (a list of message dictionaries) by:
        1. Converting each message into the format {"role": ..., "content": ...}.
        2. Converting roles using the convert_role() function.
        3. Removing the last message if it is from a non-user (i.e. from "assistant" or "system").
        4. If after removal the conversation does not end with a user message, return None.
        """
        # Convert messages to the new format.
        messages = []
        for msg in conversation:
            role = convert_role(msg.get('from', ''))
            content = msg.get('value', '')
            messages.append({'role': role, 'content': content})
        
        # Remove the last message if it is from an assistant/system.
        if messages and messages[-1]['role'] in ['assistant', 'system']:
            messages.pop()
        
        # Finally, ensure that the conversation ends with a user message.
        if not messages or messages[-1]['role'] != 'user':
            # If not, skip (filter out) this conversation.
            return None
        return messages

    def get_processed_conversations(shard_index):
        dataset = load_from_disk('./SyncInstructions/infinity_instruct_gen/Infinity-Instruct-%d' % shard_index)

        processed_conversations = []
        # Iterate over the dataset
        index = 0
        for item in tqdm(dataset):
            conversation = item.get('conversations', [])
            processed = process_conversation(conversation)
            if processed is not None:
                processed_conversations.append(processed)
            index += 1
        return processed_conversations
    
    save_every = 10000
    for idx in indices:
        convs = get_processed_conversations(idx)
        # For demonstration, print out the first few processed conversations.
        # You might also write them to a file or use them for further processing.
        # print(json.dumps(processed_conversations[:3], indent=2, ensure_ascii=False))
        j_start = save_every if idx == 0 else 0
        for s_idx in range(j_start, len(convs), save_every):
            save_path = './SyncInstructions/infinity_instruct_gen/output_infinity_instruct_gen_%d_%d.jsonl' % (idx, s_idx)
            if os.path.exists(save_path):
                continue
            subset_convs = convs[s_idx: s_idx+save_every]
            responses = async_request_questions(subset_convs, is_simple_message=False, max_tokens=28672, max_workers=512, port=30000)
            write_jsonl(responses, save_path)


def generate_open_r1_datasets(indices):
    def get_processed_conversations(shard_index):
        dataset = load_from_disk('./SyncInstructions/open_r1_math/OpenR1-Math-220k-%d' % shard_index)

        processed_conversations = []
        # Iterate over the dataset
        index = 0
        for item in tqdm(dataset):
            problem = item.get('problem', '').strip()
            if not problem:
                continue
            processed_conversations.append(problem)
            index += 1
        return processed_conversations
    
    save_every = 20000
    for idx in indices:
        convs = get_processed_conversations(idx)
        # For demonstration, print out the first few processed conversations.
        # You might also write them to a file or use them for further processing.
        # print(json.dumps(processed_conversations[:3], indent=2, ensure_ascii=False))
        j_start = 0
        for s_idx in range(j_start, len(convs), save_every):
            save_path = './SyncInstructions/open_r1_math/output_open_r1_math_gen_%d_%d.jsonl' % (idx, s_idx)
            if os.path.exists(save_path):
                continue
            subset_convs = convs[s_idx: s_idx+save_every]
            responses = async_request_questions(subset_convs, is_simple_message=True, max_tokens=28672, max_workers=512, port=30000)
            write_jsonl(responses, save_path)


def generate_kod_code_v1_datasets(indices):
    def get_processed_conversations(shard_index):
        dataset = load_from_disk('./SyncInstructions/kodcode_v1/KodCode-V1-SFT-R1-%d' % shard_index)

        processed_conversations = []
        # Iterate over the dataset
        index = 0
        for item in tqdm(dataset):
            problem = item.get('question', '').strip()
            if not problem:
                continue
            processed_conversations.append(problem)
            index += 1
        return processed_conversations
    
    save_every = 10000
    for idx in indices:
        convs = get_processed_conversations(idx)
        # For demonstration, print out the first few processed conversations.
        # You might also write them to a file or use them for further processing.
        # print(json.dumps(processed_conversations[:3], indent=2, ensure_ascii=False))
        j_start = 0
        for s_idx in range(j_start, len(convs), save_every):
            save_path = './SyncInstructions/kodcode_v1/output_kodcode_v1_gen_%d_%d.jsonl' % (idx, s_idx)
            if os.path.exists(save_path):
                continue
            subset_convs = convs[s_idx: s_idx+save_every]
            responses = async_request_questions(subset_convs, is_simple_message=True, max_tokens=32768, max_workers=512, port=30000)
            write_jsonl(responses, save_path)


def generate_open_thoughts_datasets(indices):
    """
    Generates instruction-following requests based on the 'question' field
    from the 'open-thoughts/OpenThoughts2-1M' Hugging Face dataset.
    """
    def get_processed_conversations():
        """
        Loads the dataset and extracts the 'question' field from the 'train' split.
        """
        print("Loading dataset 'open-thoughts/OpenThoughts2-1M'...")
        # Load the dataset from Hugging Face Hub
        # Using streaming=True might be beneficial for very large datasets if memory is a concern,
        # but requires adjusting the iteration logic. Sticking to standard loading for now.
        try:
            dataset = load_dataset('open-thoughts/OpenThoughts2-1M', split='train')
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []

        processed_conversations = []
        print("Processing conversations from the dataset...")
        # Iterate over the dataset
        for item in tqdm(dataset, desc="Processing OpenThoughts2-1M"):
            if not item.get('question', ''):
                continue
            question = item.get('question', '').strip()
            if not question:
                continue
            # The input to async_request_questions seems to be just the prompt string
            processed_conversations.append(question)
        print(f"Extracted {len(processed_conversations)} questions.")
        return processed_conversations

    save_every = 20000 # Adjust batch size as needed
    print(f"Generating datasets for indices: {indices}")
    convs = get_processed_conversations() # Load and process the dataset

    if not convs:
        print("No conversations processed, exiting.")
        return

    for idx in indices:
        print(f"Processing for index {idx}...")
        j_start = 0
        for s_idx in range(j_start, len(convs), save_every):
            # Define the output path
            output_dir = './scratch/huang_xin/SyncInstructions/open_thoughts'
            save_path = os.path.join(output_dir, f'output_open_thoughts_gen_{idx}_{s_idx}.jsonl')

            # Ensure the output directory exists
            os.makedirs(output_dir, exist_ok=True)

            if os.path.exists(save_path):
                print(f"Output file already exists, skipping: {save_path}")
                continue

            # Get the subset for the current batch
            subset_convs = convs[s_idx : s_idx + save_every]
            print(f"Processing batch {s_idx // save_every} (size: {len(subset_convs)}) for index {idx}...")

            # Generate responses using the async function
            # Adjust parameters like max_tokens if necessary for this dataset
            try:
                responses = async_request_questions(
                    subset_convs,
                    is_simple_message=True, # Assuming questions are simple prompts
                    max_tokens=32000,        # Adjust max_tokens as needed
                    max_workers=512,        # Keep consistent worker count
                    port=30000              # Keep consistent port
                )
                # Write the generated responses to the output file
                print(f"Writing {len(responses)} responses to {save_path}")
                write_jsonl(responses, save_path)
            except Exception as e:
                print(f"Error during async request or writing for batch {s_idx}: {e}")

        print(f"Finished processing for index {idx}.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate Infinity Gen R1 datasets for given shard indices."
    )
    parser.add_argument(
        '--indices',
        type=int,
        nargs='+',
        default=[0],
        help="List of shard indices to process. E.g., --indices 0 1 2"
    )
    args = parser.parse_args()

    generate_open_thoughts_datasets(args.indices)

