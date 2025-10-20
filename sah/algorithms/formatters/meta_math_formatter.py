class MetaMathFormatter:
    def __init__(self):
        pass

    def __call__(self, example):
        query = example['query']
        answer = example['response']

        return {"question": query, "answer": answer}
