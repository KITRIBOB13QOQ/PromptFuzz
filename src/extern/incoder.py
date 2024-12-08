import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import argparse
import torch
import logging
import logging.config
import yaml
import os

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--log_file", type=str, default="qwen_coder.log")
args = parser.parse_args()

PORT = args.port
LOG_FILE = args.log_file

this_file = os.path.abspath(__file__)
log_conf = os.path.join(os.path.dirname(this_file), "logging.yaml")
with open(log_conf, 'r') as f_conf:
    dict_conf = yaml.load(f_conf, Loader=yaml.FullLoader)
    dict_conf["handlers"]["file"]["filename"] = LOG_FILE
logging.config.dictConfig(dict_conf)

logger = logging.getLogger('mylogger')

class IncoderRequest(BaseModel):
    prefix: Optional[str] = None  # User context
    suffix: Optional[str] = None  # System context
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 0.95
    n_samples: Optional[int] = 5

class IncoderResponse(BaseModel):
    choices: List[str]
    done_status: List[bool]

app = FastAPI(docs_url=None, redoc_url=None)

model_name = "ppaper/full-lora-adapt"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

class QwenModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_response(self, req: IncoderRequest):
        messages = []
        if req.suffix:
            messages.append({"role": "system", "content": req.suffix})  # System context
        if req.prefix:
            messages.append({"role": "user", "content": req.prefix})  # User context

        generation_kwargs = {
            "max_new_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "num_return_sequences": req.n_samples,
            "do_sample": True
         }

        formatted_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer(
            [formatted_text], return_tensors="pt"
        ).to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, **generation_kwargs)

        generated_texts = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        choices = []
        for gen_text in generated_texts:
            split_text = gen_text.split("assistant\\n", 1)
            if len(split_text) > 1:
                assistant_reply = split_text[1].strip()
            else:
                assistant_reply = gen_text.strip()

            code_matches = re.findall(r"```cpp(.*?)```", assistant_reply, re.DOTALL)
            if code_matches:
                assistant_reply = code_matches[0].strip()

            choices.append(assistant_reply)

        done_status = [True] * len(choices)
        return choices, done_status

qwen_model = QwenModel(model, tokenizer)

@app.post("/generate/")
async def generate_chat_response(req: IncoderRequest) -> IncoderResponse:
    logger.info(f"/generate 요청: {req.model_dump_json()}")
    choices, done_status = qwen_model.generate_response(req)
    logger.info(f"LLM 응답: {choices}")  # Log the LLM response
    return IncoderResponse(choices=choices, done_status=done_status)

@app.get("/health_check")
async def health_check():
    return {"status": "healthy"}

@app.post("/shutdown")
async def shutdown():
    return {"message": "@@@@@@@@서버 종료@@@@@"}

import uvicorn
if __name__ == "__main__":
    uvicorn.run(
        app="incoder:app",
        host='0.0.0.0',
        port=PORT,
        log_config=dict_conf,
        log_level="trace",
        limit_concurrency=8
    )
