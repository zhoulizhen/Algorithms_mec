import matplotlib.pyplot as plt

def plot_results(iterations, *metrics):
    """
    Plots the accuracy, delay, cost, and time results.
    """
    algorithm_names = ['LP', 'Appro', 'NBS', 'NonShare', 'SC', 'OL', 'MAB', 'ADMS']
    plt.figure(figsize=(12, 9), dpi=80)

    # Accuracy plot
    ax1 = plt.subplot(231)
    for alg_name, metric in zip(algorithm_names, metrics):
        plt.plot(iterations, metric['accuracy'], marker="v", label=f"{alg_name}")
    plt.xlabel('Number of iterations')
    plt.ylabel('Average accuracy')
    plt.legend()

    # Delay plot
    ax2 = plt.subplot(232)
    for alg_name, metric in zip(algorithm_names, metrics):
        plt.plot(iterations, metric['delay'], marker="v", label=f"{alg_name}")
    plt.xlabel('Number of iterations')
    plt.ylabel('Total delay')
    plt.legend()

    # Cost plot
    ax3 = plt.subplot(233)
    for alg_name, metric in zip(algorithm_names, metrics):
        plt.plot(iterations, metric['cost'], marker="v", label=f"{alg_name}")
    plt.xlabel('Number of iterations')
    plt.ylabel('Average cost')
    plt.legend()

    # Time plot
    ax4 = plt.subplot(234)
    for alg_name, metric in zip(algorithm_names, metrics):
        plt.plot(iterations, metric['time'], marker="v", label=f"{alg_name}")
    plt.xlabel('Number of iterations')
    plt.ylabel('Running time')
    plt.legend()


    # Weight sum plot
    ax5 = plt.subplot(235)
    for alg_name, metric in zip(algorithm_names[5:], metrics[5:]):
        plt.plot(iterations, metric['weight'], marker="v", label=f"{alg_name}")
    plt.xlabel('Number of iterations')
    plt.ylabel('Total penalty')
    plt.legend()

    plt.tight_layout()
    plt.show()