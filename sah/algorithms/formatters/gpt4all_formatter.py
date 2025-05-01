class Gpt4allFormatter:
    def __init__(self):
        self.template = "<user>{prompt}</user>\n<assistant>{response}</assistant>"
    def __call__(self, example):
        return {"text": self.template.format(**example)}
