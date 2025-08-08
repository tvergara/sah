class LimaFormatter:
    def __init__(self):
        pass

    def __call__(self, example):
        conversations = example['conversations']

        # Format conversation as alternating human/assistant turns
        # First message is human, second is assistant, etc.
        formatted_text = ""
        for i, message in enumerate(conversations):
            if i % 2 == 0:  # Human message
                formatted_text += f"Human: {message}\n\n"
            else:  # Assistant message
                formatted_text += f"Assistant: {message}\n\n"

        return {"text": formatted_text.strip()}
