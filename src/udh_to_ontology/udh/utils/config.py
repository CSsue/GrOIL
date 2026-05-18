import os
import dspy
import openai
from dotenv import load_dotenv

def configure_dspy(temperature=0.1, max_tokens=None):
    load_dotenv()
    api_base = os.getenv("API_BASE", "http://localhost:8000/v1")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    if max_tokens is None:
        max_tokens = int(os.getenv("MAX_TOKENS", "4096"))

    # Dummy key: vLLM ignores it, but LiteLLM/OpenAI requires non-empty
    dummy_key = "sk-no-key-needed"

    # Set global OpenAI client config
    openai.api_base = api_base
    openai.api_key = dummy_key

    # Create DSPy LM - pass dummy_key explicitly
    model = dspy.LM(
        model=f"openai/{model_name}",  # 'openai/' prefix signals compatible API
        api_base=api_base,
        api_key=dummy_key,  # Non-empty!
        model_type="chat",
        temperature=temperature,
        max_tokens=max_tokens,
        cache=True,
        extra_body={
            "repetition_penalty": 1.15
        }
    )

    dspy.settings.configure(lm=model, temperature=temperature)  # Reduce from default ~0.7
