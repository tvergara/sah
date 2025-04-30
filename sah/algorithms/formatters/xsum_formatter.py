class XsumFormatter:
    def __init__(self):
        self.template = "{document}\nSummary: {summary}"
    def __call__(self, example):
        return {"text": self.template.format(**example)}
