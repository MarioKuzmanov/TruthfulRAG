import os
import asyncio
import json
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, APIError

@lru_cache(maxsize=1)
def initialize_openai_client(api_key=None, base_url=None):
    return OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError, APIError)
    ),
)
async def openai_chat_completion(
    model_name: str,
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    **kwargs
):
    client = initialize_openai_client()
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    api_params = {
        "model": model_name,
        "messages": messages,
        "max_tokens": kwargs.get("max_tokens", 1000),
        "temperature": kwargs.get("temperature", 1e-6),
        "top_p": kwargs.get("top_p", 1.0),
    }
    if "logprobs" in kwargs:
        api_params["logprobs"] = kwargs["logprobs"]
    if "top_logprobs" in kwargs:
        api_params["top_logprobs"] = kwargs["top_logprobs"]

    
    kwargs.pop("hashing_kv", None)    
    api_params.update(kwargs)
    # print(f"api_params: {api_params}\n\n")
    response = client.chat.completions.create(**api_params)
    choice = response.choices[0]
    # print(f"response: {response}\n\n")
    # 如果请求了 logprobs，返回 tokens 的 logprob 列表
    if "logprobs" in api_params and choice.logprobs:
        logit_list = []
        for c in choice.logprobs.content:
            logit_list.append(
                [
                    {
                        "token": tok.token,
                        "logprob": tok.logprob,
                        "bytes": tok.bytes
                    }
                    for tok in c.top_logprobs
                ]
            )
        return logit_list
    else:
        return choice.message.content.strip()


async def openai_complete(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    keyword_extraction: bool = False,
    **kwargs
):
    model_name = kwargs.pop("model_name", "gpt-4o-mini")
    result = await openai_chat_completion(
        model_name=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )
    
    return result

# response.choices[0].logprobs.content[0]