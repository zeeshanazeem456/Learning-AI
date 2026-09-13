import os
import json
import time
import argparse
from dotenv import load_dotenv

def estimate_tokens(text):
    """Rough estimate: 1 token ~= 4 characters."""
    return len(text) // 4

def load_client():
    """Load .env and initialize a Groq client. Raise a clear error if the key is missing."""
    load_dotenv("API.env")
    api_key = os.getenv("Groq_Key")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found -- get a free key at console.groq.com "
            "and add it to a .env file as GROQ_API_KEY=your_key_here"
        )
    from groq import Groq
    return Groq(api_key=api_key)

def get_system_prompt(mode):
    """Return an appropriate system prompt string based on the active mode."""
    if mode == "summarize":
        return "You are a summarization assistant that produces exactly 3 concise bullet points."
    elif mode == "qa":
        return "You are a concise, helpful assistant. Keep answers under 3 sentences."
    else:
        return "You are a helpful assistant."

def call_model_with_retry(client, model, system_prompt, history, max_retries=3, base_delay=1):
    """
    Call the LLM with the system prompt and message history.
    Includes exponential backoff for robust error handling.
    """
    if client is None:
        return f"[SIMULATED -- no live client] Reply to: {history[-1]['content'][:50]}..."
        
    full_messages = [{"role": "system", "content": system_prompt}] + history
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model, 
                max_tokens=300, 
                messages=full_messages
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            print(f"Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** (attempt - 1)))
                
    return f"[Error calling the model after {max_retries} attempts: {last_error}]"

def save_session(history, filepath="session_log.json"):
    """Write the full history list to a JSON file, nicely indented."""
    if not history:
        return
    with open(filepath, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSession saved successfully to {filepath}")

def main():
    # Stretch Goal: Add a --mode toggle
    parser = argparse.ArgumentParser(description="LLM-Powered CLI Tool")
    parser.add_argument(
        "--mode", 
        choices=["qa", "summarize"], 
        default="qa", 
        help="Choose assistant mode: 'qa' or 'summarize'"
    )
    import sys
    # If running in Jupyter, ignore the system args and use defaults
    if 'ipykernel' in sys.modules:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    try:
        client = load_client()
        print("Client loaded successfully.")
    except ValueError as e:
        client = None
        print(f"(Notice: {e})\nRunning in simulated mode.")

    system_prompt = get_system_prompt(args.mode)
    history = []
    model = "openai/gpt-oss-20b"
    total_tokens = 0

    print(f"\n=== Active Mode: {args.mode.upper()} ===")
    print("Type 'quit' to exit and save the session.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() in ["quit", "exit"]:
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        total_tokens += estimate_tokens(user_input)

        reply = call_model_with_retry(client, model, system_prompt, history)
        
        history.append({"role": "assistant", "content": reply})
        total_tokens += estimate_tokens(reply)

        print(f"Assistant: {reply}")
        # Stretch Goal: Running token-count estimate
        print(f"[Estimated Session Tokens: {total_tokens}]")

    save_session(history, filepath=f"{args.mode}_session.json")

if __name__ == "__main__":
    main()