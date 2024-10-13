
def get_accuracy(models, requests):
    accuracy_dict = {}
    import random
    for j in requests.keys():
        model_accuracies = {}
        for k in models.keys():
            model_accuracies[k] = random.uniform(0.4, 1) # todo: change the accuracy to a random number between 0.4 and 1
        accuracy_dict[j] = model_accuracies
    return accuracy_dict