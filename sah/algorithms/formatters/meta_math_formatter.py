class MetaMathFormatter:
    def __init__(self):
        pass

    def __call__(self, example):
        query = example['query']
        answer = example['response']

        text = f"Question: {query}\nResponse: {answer}"

        return {"text": text}
