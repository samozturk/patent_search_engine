service = PatentRetrievalService('patent_abstracts.txt')
results = service.retrieve_patents(
    keywords=["artificial intelligence"],
    precision_recall_balance=0.7
)

pick any model from the (list of models)[https://huggingface.co/models?library=sentence-transformers&language=multilingual&sort=trending]