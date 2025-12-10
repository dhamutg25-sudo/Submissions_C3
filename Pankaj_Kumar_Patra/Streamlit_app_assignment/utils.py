from openrouter_client import ask_openrouter

def summarize_conversation(messages):
    summary_prompt = [
        {"role": "system", "content": "Summarize the following conversation briefly."},
        {"role": "user", "content": str(messages)}
    ]
    return ask_openrouter(summary_prompt, reasoning=False)["content"]
