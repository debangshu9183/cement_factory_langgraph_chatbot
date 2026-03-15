def query_analyzer(state):

    question = state["question"]

    return {"query": question.lower()}