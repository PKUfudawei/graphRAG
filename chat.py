#!/usr/bin/env python3
import requests, time, argparse, readline
from colorama import init, Fore, Style

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", default="8000", help="Access LLM with the local port")
    parser.add_argument("-m", "--model", default="Qwen/Qwen3.5-9B", help="Choose the LLM", choices=["Qwen/Qwen3.5-9B"])
    parser.add_argument("-s", "--system_prompt", default="", help="The system prompt for LLM")
    parser.add_argument("-t", "--enable_thinking", action="store_true", help="Enable thinking with -t")
    return parser.parse_args()


def chat(query, args):
    response = requests.post(
        f"http://localhost:{args.port}/v1/chat/completions",
        json={
            "model": args.model,
            "messages": [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": query}
            ],
            "chat_template_kwargs": {"enable_thinking": args.enable_think},
        }
    )
    return response.json()


def main():
    args = parse_arguments()
    init(autoreset=True)
    
    count = 0
    try:
        while True:
            query = input(f"{Fore.BLUE}{Style.BRIGHT}[{count}] Input: {Style.RESET_ALL}") or "你是谁"
            
            start = time.time()
            result = chat(query=query, args=args)
            end = time.time()

            print(f"{Fore.CYAN}{args.model}{Style.RESET_ALL}{Fore.YELLOW} in {(end-start):.1f} s:{Style.RESET_ALL} ")
            print(f"{Fore.LIGHTGREEN_EX}{result['choices'][0]['message']['content']}{Style.RESET_ALL}\n")
            count += 1
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Exited{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
