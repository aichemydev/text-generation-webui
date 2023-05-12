import ast
import logging
import random
import re
import time
import traceback

import numpy as np
import torch
import transformers

import modules.shared as shared
from modules.callbacks import (Iteratorize, Stream,
                               _SentinelTokenStoppingCriteria)
from modules.extensions import apply_extensions
from modules.html_generator import generate_4chan_html, generate_basic_html
from modules.models import clear_torch_cache, local_rank
from modules.text_generation import fix_galactica, fix_gpt4chan, generate_softprompt_input_tensors, get_max_prompt_length, set_manual_seed


def encode(prompts, add_special_tokens=True, add_bos_token=True, truncation_length=None):
    if shared.model_type in ['rwkv', 'llamacpp']:
        encoding = shared.tokenizer(prompts,padding=True)
        input_ids = encoding["input_ids"]
        bs,ids_len = input_ids.size()
        encoding["input_ids"] = np.array(input_ids).reshape(bs, ids_len)
        return encoding
    else:
        encoding = shared.tokenizer(prompts, return_tensors='pt', add_special_tokens=add_special_tokens,padding=True)
        input_ids = encoding["input_ids"]
        # This is a hack for making replies more creative.
        if not add_bos_token and input_ids[0][0] == shared.tokenizer.bos_token_id:
            input_ids = input_ids[:, 1:]

        # Llama adds this extra token when the first character is '\n', and this
        # compromises the stopping criteria, so we just remove it
        if type(shared.tokenizer) is transformers.LlamaTokenizer and input_ids[0][0] == 29871:
            input_ids = input_ids[:, 1:]        

    # Handling truncation
    if truncation_length is not None:
        input_ids = input_ids[:, -truncation_length:]

    encoding["input_ids"] = input_ids
    if shared.model_type in ['rwkv', 'llamacpp'] or shared.args.cpu:
        raise NotImplementedError()
    elif shared.args.flexgen:
        raise NotImplementedError()
    elif shared.args.deepspeed:
        encoding = encoding.to(device=local_rank)
    else:
        if torch.has_mps:
            device = torch.device('mps')
        else:
            device = torch.device('cuda')
        encoding = encoding.to(device)    
   
    return encoding


def decode(output_ids, skip_special_tokens=True):
    return shared.tokenizer.decode(output_ids, skip_special_tokens)

def get_reply_from_output_ids(output_ids, input_ids, state):
    batch_size = output_ids.size(0)
    if shared.model_type == 'HF_seq2seq':
        reply = decode(output_ids, state['skip_special_tokens'])
    else:        
        replies = []
        new_tokens_batched = []
        for o,i in zip(output_ids,input_ids):
            new_tokens = len(o) - len(i)
            new_tokens_batched.append(new_tokens)
            reply = decode(o[-new_tokens:], state['skip_special_tokens']) # Either batch FULL decode or step by step last tokens decode
            replies.append(reply)
        print(replies)

        # We don't really need a space
        # # Prevent LlamaTokenizer from skipping a space
        # if type(shared.tokenizer) is transformers.LlamaTokenizer and len(output_ids) > 0:
        #     for i in range(batch_size):
        #         if shared.tokenizer.convert_ids_to_tokens(int(output_ids[i][-new_tokens_batched[i]])).startswith('▁'):
        #             replies[i] = ' ' + replies[i]

    for i in range(batch_size):
        replies[i] = apply_extensions('output', replies[i])  

    return replies


def formatted_outputs(replies, model_name):
    if shared.model_type == 'galactica':
        replies = [fix_galactica(reply) for reply in replies]
        return replies, replies,[generate_basic_html(reply) for reply in replies]
    elif shared.model_type == 'gpt4chan':
        replies = fix_gpt4chan(replies)
        return replies, 'Only applicable for GALACTICA models.', [generate_4chan_html(reply) for reply in replies]
    else:
        return replies, 'Only applicable for GALACTICA models.',[generate_basic_html(reply) for reply in replies]


def generate_reply_wrapper(question, state, eos_token=None, stopping_strings=None):
    for reply in generate_reply_batched(question, state, eos_token, stopping_strings):        

        yield formatted_outputs(reply, shared.model_name)


def generate_reply_batched(questions, state, eos_token=None, stopping_strings=None):
    state = apply_extensions('state', state)
    generate_func = apply_extensions('custom_generate_reply')
    if generate_func is None:
        if shared.model_name == 'None' or shared.model is None:
            logging.error("No model is loaded! Select one in the Model tab.")
            yield questions
            return

        generate_func = generate_reply_HF

    # Preparing the input
    original_questions = questions
    questions = [apply_extensions('input', q) for q in questions]

    if shared.args.verbose:
        print(f'\n\n{questions}\n--------------------\n')

    shared.stop_everything = False
    clear_torch_cache()
    seed = set_manual_seed(state['seed'])
    for reply in generate_func(questions, original_questions, seed, state, eos_token, stopping_strings):
        yield reply


def generate_reply_HF(questions, original_questions, seed, state, eos_token=None, stopping_strings=None):
    generate_params = {}
    for k in ['max_new_tokens', 'do_sample', 'temperature', 'top_p', 'typical_p', 'repetition_penalty', 'encoder_repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams', 'penalty_alpha', 'length_penalty', 'early_stopping']:
        generate_params[k] = state[k]

    if state['ban_eos_token']:
        generate_params['suppress_tokens'] = [shared.tokenizer.eos_token_id]

    if shared.args.no_cache:
        generate_params.update({'use_cache': False})

    if shared.args.deepspeed:
        generate_params.update({'synced_gpus': True})
    
    if shared.soft_prompt:
        raise NotImplementedError()

    # Encode the input
    encoding = encode(questions, add_bos_token=state['add_bos_token'], truncation_length=get_max_prompt_length(state))
    input_ids = encoding["input_ids"]
    cuda = not any((shared.args.cpu, shared.args.deepspeed))

    # Find the eos tokens
    eos_token_ids = [shared.tokenizer.eos_token_id] if shared.tokenizer.eos_token_id is not None else []
    if eos_token is not None:
        eos_id = encode(eos_token)["input_ids"][0][-1]
        eos_token_ids.append(int(eos_id))

    # Add the encoded tokens to generate_params
    
    questions, input_ids, inputs_embeds = apply_extensions('tokenizer', state, questions, input_ids, None)        
    encoding["input_ids"] = input_ids
    original_input_ids = input_ids
    generate_params.update(**encoding)
    if inputs_embeds is not None:
        generate_params.update({'inputs_embeds': inputs_embeds})

    # Create the StoppingCriteriaList with the stopping strings (needs to be done after tokenizer extensions)
    stopping_criteria_list = transformers.StoppingCriteriaList()
    for st in (stopping_strings, ast.literal_eval(f"[{state['custom_stopping_strings']}]")):
        if type(st) is list and len(st) > 0:
            sentinel_token_ids = [encode(string, add_special_tokens=False)["input_ids"][0] for string in st]            
            stopping_criteria_list.append(_SentinelTokenStoppingCriteria(sentinel_token_ids=sentinel_token_ids, starting_idx=[len(ids) for ids in input_ids]))
            break

    # Update generate_params with the eos token and the stopping strings
    generate_params['eos_token_id'] = eos_token_ids
    generate_params['stopping_criteria'] = stopping_criteria_list
    
    t0 = time.time()
    try:
        if shared.model_type != 'HF_seq2seq':
            yield ''

        # Generate the entire reply at once.        
        with torch.no_grad():
            output = shared.model.generate(**generate_params)
            if cuda:
                output = output.cuda()
 
        yield get_reply_from_output_ids(output, input_ids, state)
            
    except Exception:
        traceback.print_exc()
    finally:
        t1 = time.time()
        original_tokens = original_input_ids.numel()
        new_tokens = output.numel() - (original_tokens if shared.model_type != 'HF_seq2seq' else 0)
        print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
        return