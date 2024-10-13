import random

from Graph import FileReader as FR
from Graph import ConnectMEC as GC
from Compute import Accuracy
from Algorithm import OL, MAB, ADMS, NBSRound, NBSILP, RelaxILP, App, SC, NonShare
import matplotlib.pyplot as plt
import context

def initialize_metrics():
    """
    Initializes a dictionary to store metrics for each algorithm.
    """
    return {
        'accuracy': [],
        'delay': [],
        'cost': [],
        'time': [],
        'weight': []
    }

def run_experiment():
    # Set random seed for reproducibility

    # Initialize metrics for different algorithms
    relax_ilp_metrics = initialize_metrics()
    app_metrics = initialize_metrics()
    nbs_ilp_metrics = initialize_metrics()
    nbs_metrics = initialize_metrics()
    nonshare_metrics = initialize_metrics()
    sc_metrics = initialize_metrics()
    ol_metrics = initialize_metrics()
    mab_metrics = initialize_metrics()
    adms_metrics = initialize_metrics()

    random.seed(123)

    # Define parameters
    iterations = [30,50,100,300,500]
    num_models = 10
    request_num_usr = 1
    num_features = 100
    feature_limit = num_features // 5

    for clsnum in iterations:
        print("Iteration:", clsnum)

        usrnum = 100
        num_requests = usrnum * request_num_usr
        services = list(range(num_models // 10))

        # Generate users and cloudlets
        cls, ues_old = FR.read_file(usrnum, clsnum)
        ues = GC.fineHomeCloudlet(cls, ues_old)
        models = GC.generateModels(num_models, services)

        t2 = None
        cloudlets, locations = GC.generateCloudlets(cls, ues, t2)
        graph = GC.connectNode(locations, ues)

        t1 = None
        tcost = None
        requests = GC.generateRequests(cls, ues, request_num_usr, services, t1, tcost, cloudlets)

        num_locations = len(locations)
        num_cloudlets = len(cloudlets)

        accuracy_dict = Accuracy.get_accuracy(models, requests)
        contexts = context.generate_context(num_requests, num_features)

        xi = 0.98
        # xi = random.uniform(0, 1) # todo: change the value of xi
        alpha = random.uniform(0, 1)

        # Initialize location dictionaries
        locations_dict = initialize_model_locations(num_models, num_cloudlets, clsnum, models)

        # ----------------------------- Offline Algorithms --------------------------------- #
        print("compute relaxilp")
        relaxilp_results = RelaxILP.relaxILP(num_requests, num_models, num_locations,
                                             requests, models, cloudlets,
                                             locations, graph, accuracy_dict,
                                             xi, alpha, locations_dict['lsh'],
                                             locations_dict['lre'])

        relax_ilp_metrics['accuracy'].append(relaxilp_results[0])
        relax_ilp_metrics['delay'].append(relaxilp_results[1])
        relax_ilp_metrics['cost'].append(relaxilp_results[2])
        relax_ilp_metrics['time'].append(relaxilp_results[3])

        app_results = App.app(num_requests, num_models, num_locations, *relaxilp_results[4:],
                              requests, models, cloudlets, locations,
                              graph, accuracy_dict,
                              locations_dict['lsh'], locations_dict['lre'], alpha)

        app_metrics['accuracy'].append(app_results[0])
        app_metrics['delay'].append(app_results[1])
        app_metrics['cost'].append(app_results[2])
        app_metrics['time'].append(app_results[3])


        nbsilp_results = NBSILP.relaxILP(num_requests, num_models, num_locations,
                                             requests, models, cloudlets,
                                             locations, graph, accuracy_dict,
                                             xi, alpha, locations_dict['lsh'],
                                             locations_dict['lre'])

        nbs_ilp_metrics['accuracy'].append(nbsilp_results[0])
        nbs_ilp_metrics['delay'].append(nbsilp_results[1])
        nbs_ilp_metrics['cost'].append(nbsilp_results[2])
        nbs_ilp_metrics['time'].append(nbsilp_results[3])

        nbs_results = NBSRound.nbs(num_requests, num_models, num_locations, *nbsilp_results[4:],
                              requests, models, cloudlets, locations,
                              graph, accuracy_dict,
                              locations_dict['lsh'], locations_dict['lre'], nbsilp_results[3], alpha)

        nbs_metrics['accuracy'].append(nbs_results[0])
        nbs_metrics['delay'].append(nbs_results[1])
        nbs_metrics['cost'].append(nbs_results[2])
        nbs_metrics['time'].append(nbs_results[3])

        nonshare_results = NonShare.nonshare(num_requests, num_models, num_features,
                                             requests, models, cloudlets,
                                             locations, graph, accuracy_dict,
                                             locations_dict['lshnon'], locations_dict['lre'], alpha)

        nonshare_metrics['accuracy'].append(nonshare_results[0])
        nonshare_metrics['delay'].append(nonshare_results[1])
        nonshare_metrics['cost'].append(nonshare_results[2])
        nonshare_metrics['time'].append(nonshare_results[3])

        sc_results= SC.sc(num_requests, num_models, num_features, requests, models, cloudlets,locations, graph, accuracy_dict,locations_dict['lshnon'], locations_dict['lre'], alpha)
        sc_metrics['accuracy'].append(sc_results[0])
        sc_metrics['delay'].append(sc_results[1])
        sc_metrics['cost'].append(sc_results[2])
        sc_metrics['time'].append(sc_results[3])

        # ----------------------------- Online Algorithms --------------------------------- #
        ol_results = OL.OL(num_requests, num_models, num_features, requests,
                           models, cloudlets, locations,
                           graph, accuracy_dict, xi, contexts,
                           feature_limit, alpha)

        ol_metrics['accuracy'].append(ol_results[0])
        ol_metrics['delay'].append(ol_results[1])
        ol_metrics['cost'].append(ol_results[2])
        ol_metrics['time'].append(ol_results[3])
        ol_metrics['weight'].append(ol_results[6])

        mab_results = MAB.mab(num_requests, num_models, num_features, requests,
                              models, cloudlets, locations,
                              graph, accuracy_dict, xi,
                              feature_limit, alpha)

        mab_metrics['accuracy'].append(mab_results[0])
        mab_metrics['delay'].append(mab_results[1])
        mab_metrics['cost'].append(mab_results[2])
        mab_metrics['time'].append(mab_results[3])
        mab_metrics['weight'].append(mab_results[6])

        adms_results = ADMS.adms(num_requests, num_models, num_features, requests,
                                 models, cloudlets, locations,
                                 graph, accuracy_dict, xi, contexts,
                                 feature_limit, alpha)

        adms_metrics['accuracy'].append(adms_results[0])
        adms_metrics['delay'].append(adms_results[1])
        adms_metrics['cost'].append(adms_results[2])
        adms_metrics['time'].append(adms_results[3])
        adms_metrics['weight'].append(adms_results[6])

    print_metrics(relax_ilp_metrics, app_metrics, nbs_metrics, nonshare_metrics, sc_metrics, ol_metrics, mab_metrics, adms_metrics)

    plot_results(iterations, relax_ilp_metrics, app_metrics, nbs_metrics, nonshare_metrics, sc_metrics, ol_metrics, mab_metrics, adms_metrics)

def initialize_model_locations(num_models, num_cloudlets, clsnum, models):
    """
    Initializes location dictionaries for each model, including parameter sharing and non-sharing.
    """
    lsh = {}
    lre = {}
    lshnon = {}

    for k in range(1, num_models + 1):
        num_po_locations = random.randint(1, clsnum // 20)  # Number of locations for model k
        lk = random.sample(range(1, num_cloudlets + 1), num_po_locations)
        lsh[k] = list(lk)  # Shared locations
        lre[k] = list(lk)  # Remaining locations
        lshnon[k] = list(lk)  # Non-shared locations

    # Update shared locations based on service types
    for k1 in range(1, num_models + 1):
        lsh_copy = set(lsh[k1])  # Use set to avoid duplicates
        for k2 in range(1, num_models + 1):
            if models[k1]['service_type'] == models[k2]['service_type']:
                lsh_copy.update(lsh[k2])
        lsh[k1] = list(lsh_copy)
    return {'lsh': lsh, 'lre': lre, 'lshnon': lshnon}

def print_metrics(*metrics):
    """
    Prints the delay, accuracy, cost, time, and weight sum metrics for each algorithm.
    """
    #todo: update the algorithm names and print format

    # algorithms = ['relaxILP', 'app', 'nbs', 'nonshare', 'sc', 'ol', 'mab', 'adms']
    # for metric in metrics:
    #     for alg in algorithms:
    #         print(f"{alg} - Delay: {metric['delay'][-1]}, "
    #               f"{alg} - Accuracy: {metric['accuracy'][-1]}, "
    #               f"{alg} - Cost: {metric['cost'][-1]}, "
    #               f"{alg} - Time: {metric['time'][-1]}")


    algorithms = ['relaxilp', 'app', 'nbs', 'nonshare', 'sc', 'ol', 'mab', 'adms']

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

                if metric_key == 'weight' and alg not in ['ol', 'mab', 'adms']:
                    continue

                if metric_key in metric:
                    print(f"{alg}{metric_key}= {metric[metric_key]}")
                else:
                    print(f"{alg}{metric_key}= N/A")

    print_metrics_by_category(metrics, algorithms, metrics_to_print)


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

if __name__ == '__main__':
    run_experiment()