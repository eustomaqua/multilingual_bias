# from openai_batcher import ChatOpenAIBatcher
from typing import List
import json
import re
import faulthandler
import os
import multiprocessing
import threading
import time
import concurrent
import openai
from openai import AzureOpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from datasets import Dataset
faulthandler.enable()

DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant.'

class ChatOpenAIBatcher:
    def __init__(self, api_key, model, endpoint, api_version, response_format=None, max_tokens=None):
        self.client = OpenAI(api_key=api_key, base_url=endpoint)
        self.model = model
        self.api_version = api_version
        self.response_format = response_format
        self.max_tokens = max_tokens

    def create_batch(self, input_file_id):
        return self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "model": self.model,
                "api_version": self.api_version,
                "response_format": self.response_format,
                "max_tokens": self.max_tokens,
            }
        )
    

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



def create_batch(file_id, model_name='gpt-4o'):
    openai_key = ''
    azure_endpoint = ''
    api_endversion = '2024-07-01-preview'
    batcher = ChatOpenAIBatcher(openai_key, model_name, azure_endpoint, api_endversion, response_format=None)
    batch = batcher.create_batch(file_id)
    return batch.id


def retrieve_batch(batch_id):
    openai_key = ''
    model_name = 'gpt-4o'
    azure_endpoint = ''
    api_endversion = '2024-07-01-preview'
    batcher = ChatOpenAIBatcher(openai_key, model_name, azure_endpoint, api_endversion, response_format=None)
    batch = batcher.retrieve_batch(batch_id)
    print(batch)
    return batch.output_file_id


def retrieve_batch_response(output_file_id, output_file_path):
    openai_key = ''
    model_name = 'gpt-4o'
    azure_endpoint = ''
    api_endversion = '2024-07-01-preview'
    batcher = ChatOpenAIBatcher(openai_key, model_name, azure_endpoint, api_endversion, response_format=None)
    batch = batcher.retrieve_output(output_file_id)
    with open(output_file_path, 'w') as f:
        f.writelines(batch)


def list_batch():
    openai_key = ''
    model_name = 'gpt-4o'
    azure_endpoint = ''
    api_endversion = '2024-07-01-preview'
    batcher = ChatOpenAIBatcher(openai_key, model_name, azure_endpoint, api_endversion, response_format=None)
    batch = batcher.client.batches.list(limit=100)
    return batch.data


def output_gpt_request_jsons_for_instructions(instructions: List, output_path: str, return_json=False, system_prompt=DEFAULT_SYSTEM_PROMPT, model_name='gpt-4o', max_tokens=16384, max_rows_per_file=20000):
    """
    Automatically split the output JSON into lists.
    
    Args:
        instructions: list of instruction either in text or dict
        output_path: output path to store JSONs, will append index for subsplit.
        return_json: whether to request in JSON mode
        system_prompt: system prompt used for all instructions.
        max_rows_per_file: maximum number of rows per request JSONL file.
    """
    openai_key = ''
    azure_endpoint = ''
    api_endversion = '2024-07-01-preview'
    response_format = { "type": "json_object" } if return_json else None
    batcher = ChatOpenAIBatcher(openai_key, model_name, azure_endpoint, api_endversion, max_tokens=max_tokens, response_format=response_format)
    split = 0
    for i in range(len(instructions)):
        inst = instructions[i]
        chat_list = [
            {'role': 'system', 'content': system_prompt}
        ]
        chat_list.append({'role': 'user', 'content': str(inst)})
        batcher.add_chat(chat_list)
        if i > 0 and i % max_rows_per_file == 0:
            batcher.output_jsonl(output_path.replace('.jsonl', '_%d.jsonl' % split))
            batcher.clear_chats()
            split += 1
    if (len(instructions) - 1) % max_rows_per_file != 0:
        batcher.output_jsonl(output_path.replace('.jsonl', '_%d.jsonl' % split))


def worker(gpu_id, prompts_chunk, return_dict, progress_queue, model_name, tensor_parallel_size):
    # Set the environment variable to specify the GPU
    end_gpu_id = gpu_id + tensor_parallel_size
    gpu_id_list = list(range(gpu_id, end_gpu_id))
    gpu_id_str = ','.join(list(map(str, gpu_id_list)))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id_str
    os.environ["HF_ENDPOINT"] = "http://hf-mirror.com"
    # Import inside the worker to ensure the correct CUDA device is used
    from vllm import LLM, SamplingParams
    
    sampling_params = SamplingParams(
        best_of=1,
        temperature=0.0,
        top_p=1,
        top_k=-1,
        use_beam_search=False,
        max_tokens=4096,
        presence_penalty=0,
        frequency_penalty=0,
    )

    # Initialize the LLM instance (will use the specified GPU)
    llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)

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

def generate_with_vllm_multiprocessing(prompts, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", num_dp=8, tensor_parallel_size=1):
    """
    Generates text using the specified language model across multiple GPUs using multiprocessing.
    Displays the generation progress using tqdm.

    Args:
        prompts (List[str]): A list of input prompts for generation.
        model_name (str): The name or path of the language model to use.
        num_dp (int): The number of GPUs to use for data parallelism.

    Returns:
        List[str]: A list of generated texts corresponding to the input prompts.
    """
    # Ensure the number of GPUs does not exceed available prompts
    num_processes = min(num_dp, len(prompts))

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
            args=(gpu_id, prompt_chunks[gpu_id], return_dict, progress_queue, model_name, tensor_parallel_size)
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


def save_jsonl_rows_to_hf_dataset(jsonl_body, hf_path, train_test_split_ratio=0.97):
    dataset = Dataset.from_list(jsonl_body[:int(len(jsonl_body) * train_test_split_ratio)])
    dataset.save_to_disk(os.path.join(hf_path, 'train'))

    dataset = Dataset.from_list(jsonl_body[int(len(jsonl_body) * train_test_split_ratio):])
    dataset.save_to_disk(os.path.join(hf_path, 'test'))
    print('done')


def postprocess_outputs_row(row):
    assistant_response = row['response']['body']['choices'][0]['message'].get('content', '')
    return {'id': int(row['custom_id'][8:]), 'response': assistant_response}


def postprocess_inputs_row(row):
    assistant_response = row['body']['messages'][1]['content']
    return {'id': int(row['custom_id'][8:]), 'response': assistant_response}


def get_values_by_key_ranges(d, low, high):
    return {k: d[k] for k in range(low, high) if k in d}


def extract_digits(utterance):
    # Use a list comprehension to filter only digits
    digits_only = ''.join([char for char in utterance if char.isdigit()])
    return digits_only


def extract_gpt_ranking_score(response):
    response = list(map(lambda x: x.lower().strip(), filter(lambda x: x.strip(), response.split('- '))))
    score = None
    feedback = None
    for r in response:
        if score is None and 'score' in r:
            try:
                score = float(extract_digits(r))
            except:
                score = None
        elif feedback is None and '**feedback**' in r:
            feedback = r.replace('**feedback**', '')
        elif feedback is None and r.startswith('feedback'):
            feedback = r[len('feedback'):]
    if feedback and feedback.startswith(':'):
        feedback = feedback[1:].strip()
    return {'score': score, 'feedback': feedback}
        

def flatten_gpt_inputs_list(path, total_splits, desc='reading dataset'):
    data = [read_jsonl(path % i) for i in tqdm(range(total_splits), desc=desc)]
    flattened = [ee for e in data for ee in e]
    outputs = list(map(postprocess_inputs_row, flattened))
    return outputs


def flatten_gpt_output_list(path, total_splits, desc='reading dataset'):
    data = [read_jsonl(path % i) for i in tqdm(range(total_splits), desc=desc)]
    flattened = [ee for e in data for ee in e]
    outputs = list(map(postprocess_outputs_row, flattened))
    return outputs


def flatten_inputs_to_map(inputs):
    inputs = [ee for e in inputs for ee in e]
    inputs = list(map(postprocess_inputs_row, inputs))
    inputs = {e['id']: eval(e['response']) for e in inputs}
    return inputs


def flatten_outputs_to_map(outputs):
    outputs = [ee for e in outputs for ee in e]
    outputs = list(map(postprocess_outputs_row, outputs))
    final_ans = dict()
    errors = 0
    for e in outputs:
        try:
            final_ans[e['id']] = json.loads(e['response']) 
        except Exception as exc:
            errors += 1
    return final_ans, errors


def download_batches(batch_start, batch_end, input_path_pattern, output_path, num_workers=32):
    openai_key = ''
    model_name = 'gpt-4o'
    azure_endpoint = ''
    api_endversion = '2024-07-01-preview'
    batcher = ChatOpenAIBatcher(openai_key, model_name, azure_endpoint, api_endversion, response_format=None)
    batches = list_batch()[batch_start: batch_end]

    # Define the download function
    def download_batch(b):
        input_file = batcher.client.files.retrieve(b.input_file_id)
        if input_path_pattern in input_file.filename:
            output_filename = f'{output_path}/output_{input_file.filename}'
            if not os.path.exists(output_filename):
                print('Downloading:', input_file.filename)
                retrieve_batch_response(b.output_file_id, output_filename)
            else:
                print(output_filename, 'exists, skipping')

    # Use ThreadPoolExecutor to download batches concurrently
    max_workers = num_workers  # Adjust the number of threads as needed
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks to the executor
        future_to_batch = {executor.submit(download_batch, b): b for b in batches}
        # Use tqdm to display progress
        for future in tqdm(as_completed(future_to_batch), total=len(future_to_batch), desc='Downloading batch outputs'):
            b = future_to_batch[future]
            try:
                future.result()
            except Exception as exc:
                print(f'Batch {b} generated an exception: {exc}')


def submit_batches(max_workers, indices, submit_file):
    import time
    def process_file(i):
        try:
            file_id = submit_file(i)
            print(f"File ID for index {i}: {file_id}")
            time.sleep(8)
            for _ in range(3):
                try:
                    result = create_batch(file_id, model_name='gpt-4o')
                    print(f"Batch created for file ID {file_id}: {result}")
                    break
                except Exception as e:
                    print(f"Retrying for file ID {file_id} due to error: {e}")
                    time.sleep(6)
        except Exception as e:
            print(f"Error processing index {i}: {e}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor
        futures = {executor.submit(process_file, i): i for i in indices}
        
        # Optionally process results as they complete
        for future in as_completed(futures):
            i = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Exception occurred for index {i}: {e}")


# Function to send a request to the vLLM server
def send_request(index, message, is_simple_message, enable_thinking, max_tokens, system_prompt, client, model_name):
    try:
        if is_simple_message:
            messages = [{"role": "user", "content": message}]
        else:
            messages = message
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        # print(messages[0]['content'])
        extra_body = None if enable_thinking is None else {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
        response = client.chat.completions.create(
            model=model_name,  # Replace with your model name
            messages=messages,
            max_completion_tokens=max_tokens,
            presence_penalty=1.5,
            extra_body=extra_body,
            temperature=0.6,
            top_p=0.95,
        )
        if enable_thinking is not None and not enable_thinking:
            return index, response.choices[0].message.reasoning_content, response.usage.prompt_tokens, response.usage.completion_tokens
        else:
            return index, response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens
    except Exception as e:
        print(f"Error: {e}")
        return index, '', 0, 0  # Return zeros to maintain counts
    

def async_request_questions(prompts, is_azure=False, is_gemini=False, enable_thinking=None, max_tokens=6144, is_simple_message=True, model_name='', system_prompt='', max_workers=64*8*2, port=7995):
    total_tokens_lock = threading.Lock()  # To ensure thread-safe updates
    start_time = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    final_ans = []
    if is_azure:
        client = AzureOpenAI(
            api_version="2024-12-01-preview",
            api_key='',
            azure_endpoint="",
        )
    elif is_gemini:
        client = openai.OpenAI(base_url='https://generativelanguage.googleapis.com/v1beta/openai/', api_key='')
    else:
        client = openai.OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:%d/v1" % port,
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(send_request, i, question, is_simple_message, enable_thinking, max_tokens, system_prompt, client, model_name): question for i, question in enumerate(prompts)}
        with tqdm(total=len(futures), desc="Processing") as pbar:
            for future in concurrent.futures.as_completed(futures):
                question = futures[future]
                index, answer, prompt_tokens, completion_tokens = future.result()
                with total_tokens_lock:
                    final_ans.append({'index': index, 'prompt': question, 'answer': answer})
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    total_tokens = total_prompt_tokens + total_completion_tokens
                    elapsed_time = time.time() - start_time
                    throughput_prompt = total_prompt_tokens / elapsed_time if elapsed_time > 0 else 0
                    throughput_completion = total_completion_tokens / elapsed_time if elapsed_time > 0 else 0
                    throughput_total = total_tokens / elapsed_time if elapsed_time > 0 else 0
                pbar.set_postfix({
                    'Total TPS': f"{throughput_total:.2f}",
                    'Prompt TPS': f"{throughput_prompt:.2f}",
                    'Completion TPS': f"{throughput_completion:.2f}"
                })
                pbar.update(1)
    end_time = time.time()
    total_time = end_time - start_time
    total_tokens = total_prompt_tokens + total_completion_tokens

    print(f"\nTotal time taken: {total_time:.2f} seconds")
    print(f"Total prompt tokens processed: {total_prompt_tokens}")
    print(f"Total completion tokens processed: {total_completion_tokens}")
    print(f"Total tokens processed: {total_tokens}")

    if total_time > 0:
        average_prompt_throughput = total_prompt_tokens / total_time
        average_completion_throughput = total_completion_tokens / total_time
        average_total_throughput = total_tokens / total_time
        print(f"Average Prompt Throughput: {average_prompt_throughput:.2f} tokens/second")
        print(f"Average Completion Throughput: {average_completion_throughput:.2f} tokens/second")
        print(f"Average Total Throughput: {average_total_throughput:.2f} tokens/second")
    else:
        print("Total time is zero, cannot compute throughput")

    final_ans.sort(key=lambda x: x['index'])
    return final_ans


def postprocess_thinking_answer(ans):
    ans = ans.strip()
    if '</thinking>' in ans:
        ans = ans[ans.rindex('</thinking>')+len('</thinking>'):]
    elif '</think>' in ans:
        ans = ans[ans.rindex('</think>')+len('</think>'):]
    return ans


def postprocess_mcot_answer(ans):
    if not ans:
        return ''
    ans = ans.strip()
    if ans.startswith('directly generated:\n'):
        ans = ans[len('directly generated:\n'):]
    elif '\n\nThe final answer is' in ans:
        ans = ans[ans.rindex('\n\nThe final answer is') + 22:]
    ans = ans.strip()
    if ans.startswith(':'):
        ans = ans[1:].strip()
    for lang in ['Chinese', 'Indonesian', 'English']:
        key = 'thinking in %s:\n' % lang
        if ans.startswith(key):
            ans = ans[len(key):].strip()
    return ans


def crosslingual_consistency(predictions):
    """
    Compute cross-lingual consistency for any number of languages.
    
    :param predictions: A 2D list (or array) of shape (L, Q), where:
                       - L is the number of languages
                       - Q is the number of questions
                       - predictions[l][q] is the prediction for the l-th language on question q
    :return: A float (0 to 1) representing the fraction of questions
             where all language predictions match.
    """
    # Number of languages
    L = len(predictions)
    if L == 0:
        return 0.0
    
    # Number of questions (assuming at least one language)
    Q = len(predictions[0])
    
    # Check that all languages have the same number of questions
    for l in range(1, L):
        assert len(predictions[l]) == Q, "All languages must have the same number of predictions."
    
    consistent_count = 0
    
    # For each question q
    for q in range(Q):
        # Take the prediction of the first language for this question
        first_pred = predictions[0][q]
        
        # Check if every other language has the same prediction
        if all(predictions[l][q] == first_pred for l in range(1, L)):
            consistent_count += 1
    
    # Fraction of questions where all predictions matched
    return consistent_count / Q


def extract_boxed_answer(line):
    pattern = r"\\boxed\{([^}]+)\}"
    matches = re.findall(pattern, line)
    if matches:
        final_choice = matches[-1]
    else:
        final_choice = ''
    return final_choice.strip()