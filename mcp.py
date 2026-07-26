"""
InternTA MCP Server
==================

English:
--------
This file implements an MCP (Model Context Protocol) server for the InternTA
(Synthetic Biology Teaching Assistant) based on the DeepSeek-R1-Distill-Qwen-7B
model with QLoRA fine-tuning. It exposes a chat completion tool that can be
invoked by MCP clients.

Key components:
- FastMCP for the MCP server and tools
- Hugging Face Transformers for model loading and inference
- PEFT (Parameter-Efficient Fine-Tuning) for loading the QLoRA adapter
- BitsAndBytes for 4-bit quantization

Chinese:
--------
此文件实现了基于 DeepSeek-R1-Distill-Qwen-7B 与 QLoRA 微调的合成生物学助教
InternTA 的 MCP（Model Context Protocol）服务器，通过工具供 MCP 客户端调用。

主要组件：
- FastMCP 提供 MCP 服务器与工具
- Hugging Face Transformers 用于模型加载与推理
- PEFT 用于加载 QLoRA 适配器
- BitsAndBytes 用于 4 位量化
"""

import os
import json
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastmcp import FastMCP
import torch
from transformers.utils import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logger = logging.get_logger(__name__)
load_dotenv()

# Optional: set HF mirror for downloads
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Module-level cache for model and tokenizer (replaces st.cache_resource)
_model_and_tokenizer: Optional[tuple] = None

mcp = FastMCP(name="InternTA")


def init():
    """Download or prepare model assets if needed."""
    from modelscope import snapshot_download
    snapshot_download("Kongfoo-ai/internTAv2.0_test", cache_dir="./")
    os.system(
        "huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B "
        "--local-dir DeepSeek-R1-Distill-Qwen-7B --cache-dir DeepSeek-R1-Distill-Qwen-7B"
    )


@dataclass
class GenerationConfig:
    max_length: int = 32768
    max_new_tokens: Optional[int] = 8000
    top_p: float = 0.8
    temperature: float = 0.8
    do_sample: bool = True
    repetition_penalty: float = 1.005

    def update(self, **kwargs):
        config = asdict(self).copy()
        config.update(kwargs)
        config.pop("cache_position", None)
        return config


def load_model() -> tuple:
    """Load base model with 4-bit quantization and QLoRA adapter. Cached at module level."""
    global _model_and_tokenizer
    if _model_and_tokenizer is not None:
        return _model_and_tokenizer

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model_path = "DeepSeek-R1-Distill-Qwen-7B"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    lora_adapter_path = "internTAv2.0_test"
    lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)
    _model_and_tokenizer = (lora_model, tokenizer)
    return _model_and_tokenizer


@torch.inference_mode()
def generate_response(
    model,
    tokenizer,
    prompt: str,
    generation_config: Optional[GenerationConfig] = None,
    **kwargs,
) -> str:
    """Run non-streaming generation and return the full assistant text."""
    inputs = tokenizer([prompt], return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()

    if generation_config is None:
        generation_config = GenerationConfig()
    gen_kwargs = generation_config.update(**kwargs)
    if gen_kwargs.get("pad_token_id") is None:
        gen_kwargs["pad_token_id"] = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )

    outputs = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt) :]
    return generated_text.strip()


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Build chat prompt from list of message dicts (role, content)."""
    total_prompt = "<s>"
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            total_prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            total_prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            total_prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    total_prompt += "<|im_start|>assistant\n"
    return total_prompt


@mcp.tool
def chat_completion(
    messages: str,
    temperature: float = 0.8,
    max_tokens: int = 8000,
    top_p: float = 0.8,
) -> Dict[str, Any]:
    """
    Get a chat completion from the InternTA (synthetic biology teaching assistant) model.
    :param messages: JSON array of message objects, each with "role" ("system"|"user"|"assistant") and "content".
    :param temperature: Sampling temperature (default 0.8).
    :param max_tokens: Maximum new tokens to generate (default 8000).
    :param top_p: Top-p nucleus sampling (default 0.8).
    :return: Dict with "content" (assistant reply) and "role" ("assistant").
    """
    try:
        msg_list = json.loads(messages)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid messages JSON: {e}", "content": "", "role": "assistant"}

    if not isinstance(msg_list, list):
        return {"error": "messages must be a JSON array", "content": "", "role": "assistant"}

    for i, m in enumerate(msg_list):
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            return {
                "error": f"messages[{i}] must be an object with 'role' and 'content'",
                "content": "",
                "role": "assistant",
            }
        m["content"] = str(m["content"])

    try:
        model, tokenizer = load_model()
    except Exception as e:
        return {"error": f"Model load failed: {e}", "content": "", "role": "assistant"}

    prompt = _messages_to_prompt(msg_list)
    config = GenerationConfig(
        temperature=temperature,
        max_new_tokens=max_tokens,
        top_p=top_p,
    )
    try:
        content = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            generation_config=config,
            use_cache=False,
        )
    except Exception as e:
        return {"error": str(e), "content": "", "role": "assistant"}

    return {"role": "assistant", "content": content}


@mcp.tool
def internta_health() -> Dict[str, Any]:
    """
    Check whether the InternTA model is loaded and ready.
    :return: Dict with "loaded" (bool) and optional "error" if load failed.
    """
    global _model_and_tokenizer
    if _model_and_tokenizer is not None:
        return {"loaded": True}
    try:
        load_model()
        return {"loaded": True}
    except Exception as e:
        return {"loaded": False, "error": str(e)}


if __name__ == "__main__":
    mcp.run()
