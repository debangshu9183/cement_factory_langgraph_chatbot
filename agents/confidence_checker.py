def confidence_checker(state):

    answer = state["answer"]

    if len(answer) < 80:
        return {"confidence": 0.3}

    return {"confidence": 0.9}