import random

from Graph import FileReader as FR
from Graph import ConnectMEC as GC
from Compute import Accuracy
from Offline import SC, RelaxILP, NonShare, NBSRound, NBSILP, App
from Oline import OL, MAB, ADMS
import context
from metrics import initialize_metrics
from plot import plot_results
from results import print_metrics
from preheat import initialize_model_locations


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
    iterations = [10, 30, 50, 70, 100]
    request_num_usr = 1
    num_features = 100
    feature_limit = num_features // 5
    clsnum = 300
    usrnum = 300

    for num_models in iterations:

        num_requests = usrnum * request_num_usr
        services = list(range(num_models // 5))

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
        relaxilp_results = RelaxILP.relaxILP(num_requests, num_models, num_locations, requests, models, cloudlets, locations, graph, accuracy_dict, xi, alpha, locations_dict['lsh'], locations_dict['lre'])

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

        sc_results= SC.sc(num_requests, num_models, num_features, requests, models, cloudlets, locations, graph, accuracy_dict, locations_dict['lshnon'], locations_dict['lre'], alpha)
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




if __name__ == '__main__':
    run_experiment()