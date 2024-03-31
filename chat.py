from typing import List
from ctransformers import AutoModelForCausalLM


llm = AutoModelForCausalLM.from_pretrained("zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf")
history = []


def get_prompt(instruction: str, history: List[str] = []) -> str:
    system = "You are a straightforward knowledge base machine designed to provide short, clear and concise answers."
    memory = f"Here is the conversation history: {''.join(history)}\nNow, answer the questions: {instruction}"
    prompt = f"### System:\n{system}\n\n### User:\n{memory}\n\n### Response:\n"
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
