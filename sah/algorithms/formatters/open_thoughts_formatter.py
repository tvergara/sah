class OpenThoughtsFormatter:
    def __init__(self):
        pass

    def __call__(self, example):
        conversations = example['conversations']
        question = conversations[0]['value']
        answer = conversations[1]['value']
        return {"question": question, "answer": answer}
