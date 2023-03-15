from pathlib import Path
from operator import itemgetter

import openai
import gradio as gr
from ruamel.yaml import YAML


default_message = {
    "role": "system",
    "content": "You are a helpful assistant.",
}

config_file = Path("config.yml")
if not config_file.exists():
    raise FileNotFoundError("Config file not found.")

with config_file.open() as fp:
    yaml = YAML(typ="safe")
    config = yaml.load(fp)

openai.api_key = config["openai"]["api_key"]


with gr.Blocks() as ui:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    with gr.Row() as row:
        with gr.Column(scale=1):
            clear = gr.Button("Clear")
            model_select = gr.Dropdown(
                list(
                    map(
                        itemgetter("id"),
                        openai.Model.list().data,
                    )
                ),
                label="Model select",
                value="gpt-3.5-turbo",
            )

        with gr.Column(scale=3):
            history = gr.Dropdown(
                list(
                    map(
                        itemgetter("content"),
                        filter(
                            lambda msg: msg["role"] == "user",
                            [],
                        ),
                    )
                ),
                label="History",
            )

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        chat_completion = openai.ChatCompletion.create(
            model=model_select.value,
            messages=[
                default_message,
                *[
                    {
                        "role": "user",
                        "content": message[0],
                    }
                    for message in history
                ]
            ],
        )

        history[-1][1] = chat_completion.choices[0].message.content

        return history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)


if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=7862)
