import sys
import time
import torch
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

MODEL_NAME     = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 4096

CYAN   = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
GRAY   = "\033[90m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def load_model():
    print(f"{BOLD}Loading: {MODEL_NAME}{RESET}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"{GREEN}Model ready.{RESET}\n")
    return tokenizer, model


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def stream_response(tokenizer, model, question: str):
    # ── build prompt ───────────────────────────────────────────────
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    prompt_token_count = inputs["input_ids"].shape[-1]

    # ── set up streamer ────────────────────────────────────────────
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,   # must keep <think> / </think>
    )

    thread = Thread(
        target=model.generate,
        kwargs=dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            do_sample=True,
        ),
    )
    thread.start()

    # ── streaming loop ─────────────────────────────────────────────
    print(f"\n{GRAY}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}[Thinking]{RESET}")

    in_thinking  = True
    buffer       = ""   # accumulates text to detect multi-char tokens like </think>
    thinking_buf = ""   # full thinking text (for token count)
    answer_buf   = ""   # full answer text   (for token count)
    start_time   = time.time()

    for chunk in streamer:
        buffer += chunk

        if in_thinking:
            if "</think>" in buffer:
                before, after = buffer.split("</think>", 1)
                thinking_tail = before.replace("<think>", "")

                # flush remaining thinking
                sys.stdout.write(CYAN + thinking_tail + RESET)
                sys.stdout.flush()
                thinking_buf += thinking_tail

                # switch to answer
                print(f"\n{GRAY}{'─' * 60}{RESET}")
                print(f"{BOLD}{YELLOW}[Answer]{RESET}")

                if after:
                    sys.stdout.write(YELLOW + after + RESET)
                    sys.stdout.flush()
                    answer_buf += after

                buffer      = ""
                in_thinking = False

            else:
                # still in thinking — flush everything except the opening <think>
                display = buffer.replace("<think>", "")
                if display:
                    sys.stdout.write(CYAN + display + RESET)
                    sys.stdout.flush()
                    thinking_buf += display
                buffer = ""

        else:
            # answer mode
            display = buffer.replace("</think>", "")
            if display:
                sys.stdout.write(YELLOW + display + RESET)
                sys.stdout.flush()
                answer_buf += display
            buffer = ""

    # flush any leftover buffer
    if buffer:
        display = buffer.replace("<think>", "").replace("</think>", "")
        if display:
            sys.stdout.write((CYAN if in_thinking else YELLOW) + display + RESET)
            sys.stdout.flush()
            if in_thinking:
                thinking_buf += display
            else:
                answer_buf += display

    thread.join()
    elapsed = time.time() - start_time

    # ── token stats ────────────────────────────────────────────────
    thinking_tokens = count_tokens(tokenizer, thinking_buf)
    answer_tokens   = count_tokens(tokenizer, answer_buf)
    total_output    = thinking_tokens + answer_tokens

    print(f"\n{GRAY}{'─' * 60}")
    print(f"{BOLD}[Token stats]{RESET}")
    print(f"{GRAY}  Prompt tokens   : {prompt_token_count:>6}")
    print(f"  Thinking tokens  : {thinking_tokens:>6}")
    print(f"  Answer tokens    : {answer_tokens:>6}")
    print(f"  Total output     : {total_output:>6}")
    print(f"  Time elapsed     : {elapsed:>5.1f}s")
    tps = total_output / elapsed if elapsed > 0 else 0
    print(f"  Throughput       : {tps:>5.1f} tok/s")
    print(f"{'─' * 60}{RESET}\n")


def main():
    tokenizer, model = load_model()
    print(f"Type your question. {BOLD}exit{RESET} to quit.\n")

    while True:
        try:
            question = input(f"{BOLD}You: {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break

        stream_response(tokenizer, model, question)
        
if __name__ == "__main__":
    main()