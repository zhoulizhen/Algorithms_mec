import random
import numpy as np
import matplotlib.pyplot as plt
from Graph import FileReader as FR
from Graph import ConnectMEC as GC
from Compute import Accuracy
from Algorithm import (OL, MAB, ADMS, NBSRound, NBSILP, RelaxILP,
                       App, SC, NonShare, RelaxNonlinear, Nonlinear, ILP)


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
    random.seed(123)

    # Initialize metrics for different algorithms
    ilp_metrics = initialize_metrics()
    nonlinear_metrics = initialize_metrics()
    relax_nonlinear_metrics = initialize_metrics()
    relax_ilp_metrics = initialize_metrics()
    app_metrics = initialize_metrics()
    ol_metrics = initialize_metrics()
    nonshare_metrics = initialize_metrics()
    sc_metrics = initialize_metrics()
    mab_metrics = initialize_metrics()
    adms_metrics = initialize_metrics()
    nbs_metrics = initialize_metrics()

    # Define parameters
    iterations = [30, 50, 100, 300, 500]
    num_models = 10  # Total number of models
    request_num_usr = 1  # Number of requests per user
    num_features = 100
    feature_limit = num_features // 5

    for clsnum in iterations:
        print("Iteration:", clsnum)

        usrnum = 100
        num_requests = usrnum * request_num_usr  # Total number of requests
        services = list(range(num_models // 10))  # Service types

        contexts = context.generate_context(num_requests, num_features)

        # Generate users and cloudlets
        cls, ues_old = FR.read_file(usrnum, clsnum)
        ues = GC.fineHomeCloudlet(cls, ues_old)  # Add home cloudlet
        models = GC.generateModels(num_models, services)  # Generate models

        t2 = None
        cloudlets, locations = GC.generateCloudlets(cls, ues, t2)  # Generate cloudlets
        graph = GC.connectNode(locations, ues)  # Connect cloudlets

        t1 = None
        tcost = None
        requests = GC.generateRequests(cls, ues, request_num_usr, services, t1, tcost, cloudlets)  # Generate requests

        num_locations = len(locations)
        num_cloudlets = len(cloudlets)
        print("Number of cloudlets:", num_cloudlets)

        accuracy_dict = Accuracy.get_accuracy(models, requests)

        xi = 0.98
        alpha = random.uniform(0, 1)

        # Initialize location dictionaries
        locations_dict = initialize_model_locations(num_models, num_cloudlets, clsnum)

        # ----------------------------- Offline Algorithms --------------------------------- #
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

        # Non-parameter sharing algorithms
        nonshare_results = NonShare.nonshare(num_requests, num_models, num_features,
                                             requests, models, cloudlets,
                                             locations, graph, accuracy_dict,
                                             locations_dict['lshnon'], locations_dict['lre'], alpha)

        nonshare_metrics['accuracy'].append(nonshare_results[0])
        nonshare_metrics['delay'].append(nonshare_results[1])
        nonshare_metrics['cost'].append(nonshare_results[2])
        nonshare_metrics['time'].append(nonshare_results[3])

        # Add additional algorithms as necessary...

        # ----------------------------- Online Algorithms --------------------------------- #
        ol_results = OL.OL(num_requests, num_models, num_features, requests,
                           models, cloudlets, locations,
                           graph, accuracy_dict, xi, contexts,
                           feature_limit, alpha)

        ol_metrics['accuracy'].append(ol_results[0])
        ol_metrics['delay'].append(ol_results[1])
        ol_metrics['cost'].append(ol_results[2])
        ol_metrics['time'].append(ol_results[3])
        ol_metrics['weight'].append(ol_results[7])  # Weight sum

        mab_results = MAB.mab(num_requests, num_models, num_features, requests,
                              models, cloudlets, locations,
                              graph, accuracy_dict, xi,
                              feature_limit, alpha)

        mab_metrics['accuracy'].append(mab_results[0])
        mab_metrics['delay'].append(mab_results[1])
        mab_metrics['cost'].append(mab_results[2])
        mab_metrics['time'].append(mab_results[3])
        mab_metrics['weight'].append(mab_results[7])  # Weight sum

        adms_results = ADMS.adms(num_requests, num_models, num_features, requests,
                                 models, cloudlets, locations,
                                 graph, accuracy_dict, xi, contexts,
                                 feature_limit, alpha)

        adms_metrics['accuracy'].append(adms_results[0])
        adms_metrics['delay'].append(adms_results[1])
        adms_metrics['cost'].append(adms_results[2])
        adms_metrics['time'].append(adms_results[3])
        adms_metrics['weight'].append(adms_results[7])  # Weight sum

    print_metrics(relax_ilp_metrics, app_metrics, nonshare_metrics, ol_metrics, mab_metrics, adms_metrics)

    plot_results(iterations, relax_ilp_metrics, app_metrics, nonshare_metrics,
                 ol_metrics, mab_metrics, adms_metrics)


def initialize_model_locations(num_models, num_cloudlets, clsnum):
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
    algorithms = ['relaxILP', 'app', 'nbs', 'nonshare', 'sc', 'ol', 'mab', 'adms']
    for metric in metrics:
        for alg in algorithms:
            if alg in metric:
                print(f"{alg} - Delay: {metric['delay'][-1]}, "
                      f"Accuracy: {metric['accuracy'][-1]}, "
                      f"Cost: {metric['cost'][-1]}, "
                      f"Time: {metric['time'][-1]}")


def plot_results(iterations, *metrics):
    """
    Plots the accuracy, delay, cost, and time results.
    """
    plt.figure(figsize=(12, 9), dpi=80)

    # Accuracy plot
    ax1 = plt.subplot(221)
    for metric in metrics:
        plt.plot(iterations, metric['accuracy'], marker="v", label=f"{metric.__name__} Accuracy")
    plt.xlabel('Number of Cloudlets')
    plt.ylabel('Accuracy')
    plt.legend()

    # Delay plot
    ax2 = plt.subplot(222)
    for metric in metrics:
        plt.plot(iterations, metric['delay'], marker="v", label=f"{metric.__name__} Delay")
    plt.xlabel('Number of Cloudlets')
    plt.ylabel('Delay')
    plt.legend()

    # Cost plot
    ax3 = plt.subplot(223)

