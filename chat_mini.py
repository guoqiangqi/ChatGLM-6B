from transformers import AutoTokenizer, AutoModel, AutoConfig
import pandas as pd
import torch
import os
import time

MODEL_NAME_OR_PATH = "/root/LLM/chatglm-6b"
PRE_SEQ_LEN = 128
# PTUNING_CHECKPOINT = "/root/LLM/ChatGLM-6B/ptuning/output/openeuler-resultDataFromtWithGPT-all-clean-auto-man-delete-chatglm-6b-pt-128-2e-3-opt/checkpoint-10000"
PTUNING_CHECKPOINT = None
QUANTIZATION_BIT = None

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME_OR_PATH, trust_remote_code=True)
config = AutoConfig.from_pretrained(
    MODEL_NAME_OR_PATH, trust_remote_code=True)

config.pre_seq_len = PRE_SEQ_LEN
config.prefix_projection = False

if PTUNING_CHECKPOINT is not None:
    print(f"Loading prefix_encoder weight from {PTUNING_CHECKPOINT}")
    model = AutoModel.from_pretrained(MODEL_NAME_OR_PATH, config=config, trust_remote_code=True)
    prefix_state_dict = torch.load(os.path.join(PTUNING_CHECKPOINT, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
else:
    model = AutoModel.from_pretrained(MODEL_NAME_OR_PATH, config=config, trust_remote_code=True)

if QUANTIZATION_BIT is not None:
    print(f"Quantized to {QUANTIZATION_BIT} bit")
    model = model.quantize(QUANTIZATION_BIT)

if PRE_SEQ_LEN is not None:
    # P-tuning v2
    model = model.half().cuda()
    model.transformer.prefix_encoder.float().cuda()

model = model.eval()

if __name__ == "__main__":
    response, history = model.chat(tokenizer, "你好", history=[])
    print(response)
    response, history = model.chat(tokenizer, "你是谁", history=[])
    print(response)
    exit()

    timeTotal = 0
    sourceFile = "/root/data/openeuler_corpus.xlsx"

    reader = pd.read_excel(sourceFile, sheet_name="FAQ", engine="openpyxl")
    questiones = reader["prompt"]

    answers = []
    count = 0
    for question in questiones:
        count+=1

        now = time.time()

        response, history = model.chat(tokenizer, question, history=[])

        end = time.time()
        print("Success, handled number {} question.".format(count))
        timeTotal = end - now + timeTotal

        answers.append(response)

    reader["answerChatGLM"] = answers

    print("QA finished: {} questiones have been processed, {} seconds have been spent.".format(count, timeTotal))

    print("Writing back to excel....")
    reader.to_excel("/root/data/openeuler_corpus_chatglm.xlsx", index=False)
    print("Job finished!")