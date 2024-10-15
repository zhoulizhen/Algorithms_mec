
def print_metrics(*metrics):
    """
    Prints the delay, accuracy, cost, time, and weight sum metrics for each algorithm.
    """

    algorithms = ['LP', 'Appro', 'NBS', 'NonShare', 'SC', 'OL', 'MAB', 'ADMS']

    metrics_to_print = {
        'delay': 'Delay',
        'accuracy': 'Accuracy',
        'cost': 'Cost',
        'time': 'Time',
        'weight': 'Weight Sum'
    }

    def print_metrics_by_category(metrics_list, algorithms_list, metrics_keys):
        for metric_key, display_name in metrics_keys.items():
            print(f"-----------------------{display_name}---------------------")
            for i, alg in enumerate(algorithms_list):
                if i >= len(metrics_list):
                    print(f"Warning: No metrics available for {alg}.")
                    continue

                metric = metrics_list[i]

                if metric_key == 'weight' and alg not in ['OL', 'MAB', 'ADMS']:
                    continue

                if metric_key in metric:
                    print(f"'{alg}': {metric[metric_key]}")
                else:
                    print(f"'{alg}':{metric_key}= N/A")

    print_metrics_by_category(metrics, algorithms, metrics_to_print)

