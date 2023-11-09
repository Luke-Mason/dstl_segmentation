restricted = True
def dataset_gateway(arr):
    return arr[:5] if restricted else arr

def epoch_gateway(epochs):
    return 2 if restricted else epochs