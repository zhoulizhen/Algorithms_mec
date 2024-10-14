import matplotlib.pyplot as plt

def plot_results(iterations, *metrics):
    """
    Plots the accuracy, delay, cost, and time results.
    """
    algorithm_names = ['relaxILP', 'app', 'nbs', 'nonshare', 'sc', 'ol', 'mab', 'adms']
    plt.figure(figsize=(12, 9), dpi=80)

    # Accuracy plot
    ax1 = plt.subplot(221)
    for alg_name, metric in zip(algorithm_names, metrics):
        plt.plot(iterations, metric['accuracy'], marker="v", label=f"{alg_name}")
    plt.xlabel('Number of Cloudlets')
    plt.ylabel('Accuracy')
    plt.legend()

    # Delay plot
    ax2 = plt.subplot(222)
    for alg_name, metric in zip(algorithm_names, metrics):
        plt.plot(iterations, metric['delay'], marker="v", label=f"{alg_name}")
    plt.xlabel('Number of Cloudlets')
    plt.ylabel('Delay')
    plt.legend()

    # Cost plot
    ax3 = plt.subplot(223)
    for alg_name, metric in zip(algorithm_names, metrics):
        plt.plot(iterations, metric['cost'], marker="v", label=f"{alg_name}")
    plt.xlabel('Number of Cloudlets')
    plt.ylabel('Cost')
    plt.legend()

    # Time plot
    ax4 = plt.subplot(224)
    for alg_name, metric in zip(algorithm_names, metrics):
        plt.plot(iterations, metric['time'], marker="v", label=f"{alg_name}")
    plt.xlabel('Number of Cloudlets')
    plt.ylabel('Time')
    plt.legend()


    # Weight sum plot
    ax5 = plt.subplot(224)
    for alg_name, metric in zip(algorithm_names[5:], metrics[5:]):
        plt.plot(iterations, metric['weight'], marker="v", label=f"{alg_name}")
    plt.xlabel('Number of Cloudlets')
    plt.ylabel('Weight Sum')
    plt.legend()

    plt.tight_layout()
    plt.show()