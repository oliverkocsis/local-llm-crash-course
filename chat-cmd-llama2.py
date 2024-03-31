from typing import List
from ctransformers import AutoModelForCausalLM


llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q8_0.gguf")
history = []


def get_prompt(instruction: str, history: List[str] = []) -> str:
    system = "You are a straightforward knowledge base machine designed to provide short, clear and concise answers."
    memory = f"Here is the conversation history: {''.join(history)}\nNow, answer the questions: {instruction}"
    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{memory} [/INST]"
    # print(prompt)
    return prompt


def process_input(user_input):
    history.append(user_input)
    print("AI: ", end="")
    for word in llm(get_prompt(user_input, history=history), stream=True):
        print(word, end="", flush=True)
        history.append(word)
    print()


def main():
    print("AI: Hi. What would you like to know?")

    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        process_input(user_input)


if __name__ == "__main__":
    main()
