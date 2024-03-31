import chainlit as cl
from typing import List
from ctransformers import AutoModelForCausalLM

USER_SESSION_CONVERSATION_HISTORY = "conversation_history"


def get_prompt(instruction: str, history: List[str] = []) -> str:
    system = "You are a straightforward knowledge base machine designed to provide short, clear and concise answers."
    memory = f"Here is the conversation history: {''.join(history)}\nNow, answer the questions: {instruction}"
    prompt = f"### System:\n{system}\n\n### User:\n{memory}\n\n### Response:\n"
    print(prompt)
    return prompt


@cl.on_chat_start
def on_chat_start():
    print("A new chat session has started!")
    global llm, history
    llm = AutoModelForCausalLM.from_pretrained(
        "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
    )
    cl.user_session.set(USER_SESSION_CONVERSATION_HISTORY, [])


@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get(USER_SESSION_CONVERSATION_HISTORY)
    answer_complete = ""
    answer = cl.Message(content="")
    await answer.send()
    for word in llm(get_prompt(message.content, history=history), stream=True):
        answer_complete += word
        await answer.stream_token(word)

    history.append(message.content)
    history.append(answer_complete)

    await answer.update()
