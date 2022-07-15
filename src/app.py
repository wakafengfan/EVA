# coding=utf-8

"""Inference EVA"""

import logging
import math
import os
import torch
import torch.nn.functional as F
from arguments import get_args
from utils import load_checkpoint
from tokenization_eva import EVATokenizer
import mpu
import deepspeed
import torch.distributed as dist
from model import EVAModel, EVAConfig
from fp16 import FP16_Module

from utils import print_rank_0, initialize_distributed, set_random_seed
from generation_utils import generate_beam, generate_no_beam

from model import DistributedDataParallel as DDP

import re
from flask import Flask, request, jsonify
app = Flask(__name__)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
# class log_filter(logging.Filter):
#     def filter(self, record):
#         if record.msg.find('__main__') != -1:
#             return True
#         return False

# logger.addFilter(log_filter())

def get_model(args, vocab_size):
    """Build the model."""

    print_rank_0('building Enc-Dec model ...')
    config = EVAConfig.from_json_file(args.model_config)
    config.vocab_size = vocab_size
    assert not args.checkpoint_activations
    model = EVAModel(
        config,
        parallel_output=True,
        checkpoint_activations=False,
        checkpoint_num_layers=args.checkpoint_num_layers
    )

    print(' > number of parameters on model parallel rank {}: {}'.format(
        mpu.get_model_parallel_rank(),
        sum([p.nelement() for p in model.parameters()])), flush=True)

    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if args.deepspeed and args.fp16:
        model.half()

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    model = DDP(model)

    return model
    

def setup_model(args, vocab_size):
    """Setup model and optimizer."""

    model = get_model(args, vocab_size)

    if args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")

        model, _, _, _ = deepspeed.initialize(
            model=model,
            optimizer=None,
            args=args,
            lr_scheduler=None,
            mpu=mpu,
            dist_init_required=False
        )

    if args.load is not None:
        args.iteration = load_checkpoint(args, model)
    else:
        args.iteration = 0

    return model


def get_attn_masks(args,
                  tokenizer,
                  contexts,
                  targets):
    # Extract batch size and sequence length.
    batch_size, enc_seq_length = contexts.size()

    # Enc Attention mask.
    enc_attn_mask = torch.zeros(
        batch_size, 1, enc_seq_length, enc_seq_length, device=contexts.device)

    ctx_lengths = (contexts != tokenizer.pad_id).sum(1)
    for b in range(batch_size):
        enc_attn_mask[b, 0, :ctx_lengths[b], :ctx_lengths[b]] = 1

    batch_size, dec_seq_length = targets.size()
    # Dec Attention mask
    dec_attn_mask = torch.tril(torch.ones(
        batch_size, 1, dec_seq_length, dec_seq_length, device=targets.device))

    # Cross Attention Mask
    cross_attn_mask = torch.zeros(
        batch_size, 1, dec_seq_length, enc_seq_length, device=contexts.device)

    for b in range(batch_size):
        cross_attn_mask[b, 0, :, :ctx_lengths[b]] = 1

    if args.fp16:
        enc_attn_mask = enc_attn_mask.half()
        dec_attn_mask = dec_attn_mask.half()
        cross_attn_mask = cross_attn_mask.half()

    model_batch = {
        "enc_attention_mask": enc_attn_mask,
        "dec_attention_mask": dec_attn_mask,
        "cross_attention_mask": cross_attn_mask,
    }

    return model_batch


def get_inference_batch(
        context_tokens,
        device,
        batch_size,
        tokenizer,
        args,
    ):
    tokens = context_tokens
    tokens = tokens.view(batch_size, -1).contiguous()
    tokens = tokens.to(device)
    
    targets = torch.zeros(batch_size, 1, dtype=torch.long, device=device) + tokenizer.get_sentinel_id(0)

    # Get the masks and postition ids.
    model_batch = get_attn_masks(
        args,
        tokenizer,
        tokens,
        targets,
    )
    
    model_batch = {
        "enc_input_ids": tokens,
        "dec_input_ids": targets,
        **model_batch
    }

    return model_batch


def generate_samples(model, tokenizer: EVATokenizer, args, device):
    model.eval()

    with torch.no_grad():
        full_context_list = []
        while True:
            if dist.get_rank() == 0:
                input_text = input("Usr >>> ")
                if input_text == "clear":
                    print("Clear Dialog")
                    # set_random_seed(args.seed) # reset rng
                    full_context_list = []
                    length_tensor = torch.tensor([-1], dtype=torch.long).to(device)
                    continue
                if input_text == "seed":
                    seed = int(input("Seed >>> "))
                    print("Clear Dialog")
                    set_random_seed(seed)
                    full_context_list = []
                    length_tensor = torch.tensor([-1], dtype=torch.long).to(device)
                    continue
                else:
                    full_context_list.append(tokenizer.encode(input_text) + [tokenizer.sep_id])
                    full_context = [x for y in full_context_list for x in y]
                    trunc_context = []
                    for utt in full_context_list[:-9:-1]:
                        if len(trunc_context) + len(utt) + 1 <= 128:
                            trunc_context = utt + trunc_context
                    trunc_context.append(tokenizer.get_sentinel_id(0))
                    length_tensor = torch.tensor([len(trunc_context), len(full_context)], dtype=torch.long).to(device)
                    trunc_context = torch.tensor(trunc_context, dtype=torch.long).to(device)
                    full_context = torch.tensor(full_context, dtype=torch.long).to(device)
                    
            else:
                length_tensor = torch.zeros(2, dtype=torch.long).to(device)

            dist.barrier()
            dist.broadcast(length_tensor, 0)
            if length_tensor[0] < 0:
                continue
            if dist.get_rank() != 0:
                trunc_context = torch.zeros(int(length_tensor[0]), dtype=torch.long).to(device)
                full_context = torch.zeros(int(length_tensor[1]), dtype=torch.long).to(device)
            dist.broadcast(trunc_context, 0)
            dist.broadcast(full_context, 0)


            # encoder tensor
            trunc_context = trunc_context.unsqueeze(0).repeat(args.batch_size, 1) # repeat
            full_context = full_context.unsqueeze(0).repeat(args.batch_size, 1) 
            model_batch = get_inference_batch(trunc_context, device, args.batch_size, tokenizer, args)
   
            if args.num_beams == 1:
                generation_str_list, generation_id_list = generate_no_beam(model_batch, full_context, model, tokenizer, args, device)
            else:
                generation_str_list, generation_id_list  = generate_beam(model_batch, full_context, model, tokenizer, args, device)

            full_context_list.append(generation_id_list[0] + [tokenizer.sep_id])

            if dist.get_rank() == 0:
                print("Sys >>> {}".format(generation_str_list[0]))



"""Main serving program."""

print('Loading Model ...')

# Disable CuDNN.
torch.backends.cudnn.enabled = False

# Arguments.
args = get_args()
args.model_config = '/workspace/EVA/src/configs/model/eva2.0_model_config.json'
args.model_parallel_size = 1
args.load = '/workspace/EVA/checkpoints/eva2.0'
args.no_load_strict = True
args.distributed_backend = 'nccl'
args.weight_decay = 1e-2
args.clip_grad = 1.0
args.tokenizer_path = '/workspace/EVA/bpe_dialog_new'
args.temperature = 0.9
args.top_k = 0
args.top_p = 0.9
args.num_beams = 4
args.length_penalty = 1.6
args.repetition_penalty = 1.6
args.rule_path = '/workspace/EVA/rules'
args.fp16 = True
args.deepspeed = True
args.deepspeed_config = '/workspace/EVA/src/configs/deepspeed/eva_ds_config.json'



# Pytorch distributed.
initialize_distributed(args)

# Random seeds for reproducability.
set_random_seed(args.seed)

#get the tokenizer
tokenizer = EVATokenizer(os.path.join(args.tokenizer_path, 'vocab.txt'))

# Model, optimizer, and learning rate.
d_model = setup_model(args, tokenizer.vocab_size)

#setting default batch size to 1
args.batch_size = 4
# os.system('clear')

device = torch.cuda.current_device()
print('Start Inference')


@app.route('/conditional_gen_answer', methods=['post'])
def generate_samples_v2():
    query = request.json['query']
    return_flag = request.json.get('return_flag', '1')
    logger.info(f"query: {query}")

    if isinstance(query, str):
        query = [query]
    
    assert isinstance(query, list) and len(query) > 0
    
    query = [_.lower() for _ in query]
    if query[-1].startswith('player:'):
        query[-1] = query[-1].replace('elf', '你')
        query[-1] = query[-1].replace('吼', '好')
        for n in ['林时', '苏承云', '乔新沐', '楚卫', '简与之', '纪远']:
            query[-1] = query[-1].replace(n, '小哥哥'+n)
    if len(query) > 1 and query[-2].startswith('bot:'):
        query[-2] = query[-2].replace('elf', '我')
        query[-2] = query[-2].replace('主人', '你')
        for ci, c in enumerate(query[-2][::-1]):
            if c == '！' and ci > 2:
                query[-2] = query[-2][::-1][:ci][::-1]
                break


    input_list = [q.replace('player:', '').replace('bot:', '') for q in query][-2:]
    logger.info(f"input_list: {input_list}")
    full_context_list = []

    d_model.eval()

    with torch.no_grad():
        for input_text in input_list:
            full_context_list.append(tokenizer.encode(input_text) + [tokenizer.sep_id])
        full_context = [x for y in full_context_list for x in y]
        trunc_context = []
        for utt in full_context_list[:-9:-1]:
            if len(trunc_context) + len(utt) + 1 <= 128:
                trunc_context = utt + trunc_context
        trunc_context.append(tokenizer.get_sentinel_id(0))
        length_tensor = torch.tensor([len(trunc_context), len(full_context)], dtype=torch.long).to(device)
        trunc_context = torch.tensor(trunc_context, dtype=torch.long).to(device)
        full_context = torch.tensor(full_context, dtype=torch.long).to(device)

        dist.barrier()
        dist.broadcast(length_tensor, 0)
        # if length_tensor[0] < 0:
            # continue
        if dist.get_rank() != 0:
            trunc_context = torch.zeros(int(length_tensor[0]), dtype=torch.long).to(device)
            full_context = torch.zeros(int(length_tensor[1]), dtype=torch.long).to(device)
        dist.broadcast(trunc_context, 0)
        dist.broadcast(full_context, 0)


        # encoder tensor
        trunc_context = trunc_context.unsqueeze(0).repeat(args.batch_size, 1) # repeat
        full_context = full_context.unsqueeze(0).repeat(args.batch_size, 1) 
        model_batch = get_inference_batch(trunc_context, device, args.batch_size, tokenizer, args)

        if args.num_beams == 1:
            generation_str_list, generation_id_list = generate_no_beam(model_batch, full_context, d_model, tokenizer, args, device)
        else:
            generation_str_list, generation_id_list  = generate_beam(model_batch, full_context, d_model, tokenizer, args, device)
        
        generation_str_list = [(t, s, s/(10*len(t))) for t, s in generation_str_list]
        generation_str_list = sorted(generation_str_list, key=lambda x: x[1], reverse=True)
        generation_texts, scores, weighted_scores = zip(*generation_str_list)
        
        res_list = []
        for i, t in enumerate(generation_texts):
            t = t.replace('!', '！')
            t = t.replace('你','主人').replace('您', '主人')
            if not (any(_ in t for _ in ['我的妈呀', '我的天啊', '我的天', '我不是药神']) or len(re.findall('《.*我.*》', t)) > 0):
                t = t.replace('我', 'ELF').replace('我们', 'ELF')
            
            # 吃过-还没吃， 去过-还没去
            if len(input_list)>1 and ('吃过' in input_list[0] and '没吃过' not in input_list[0]):
                if  any(_ in t for _ in ['还没吃', ',没吃呢,']) and not (len(res_list) == 0 and i == len(generation_texts)-1):
                    continue
            
            # elf是主人的
            if re.search('ELF是.*的主人', t):
                t = t.replace('的主人', '的,主人')
            # elf是主人
            for invalid in ['ELF是主人,', 'ELF是主人啊']:
                if invalid in t:
                    t = t.replace('是主人', '就是ELF')

            res_list.append(t)
        
        # special = ["早上好", "晚上好", "上午好", "下午好", "哈喽", "hello", "你好"]
        special = ["哈喽", "hello", "你好", '您好', '你好呀', '你好啊']
        for s in special:
            if s == input_list[-1]:
                res_list[0] = s.replace('你', '主人').replace('您', '主人')
                break

        if return_flag == '1':
            res_json = {
                "response_candidates": res_list
            }
        else:
            res_json = {
                'res': [[t,s,ws] for t, s , ws in zip(res_list, scores, weighted_scores)]
            }
        logger.info(f"res_json: {res_json}")
        logger.info(f"res: {generation_str_list}\n")

        res = jsonify(res_json)

        return res


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=8009, use_reloader=False, threaded=False)




