import random

from Graph import FileReader as FR
from Graph import ConnectMEC as GC
from Compute import Accuracy
from Online import OL, MAB, ADMS

import context
from metrics import initialize_metrics
from preheat import initialize_model_locations

def run_experiment():

    # Initialize metrics for different algorithms
    ol_metrics = initialize_metrics()
    mab_metrics = initialize_metrics()
    adms_metrics = initialize_metrics()

    random.seed(123)

    clsnum = 100
    num_models = 50
    request_num_usr = 1
    num_features = 50
    feature_limit = num_features // 2

    usrnum = 400
    num_requests = usrnum * request_num_usr  # The number of requests
    services = [i for i in range(num_models // 5)]  # The number of service types

    contexts = context.generate_context(num_requests, num_features)

    xi = random.uniform(0, 1)
    alpha = random.uniform(0, 1)

    cls, uesOld = FR.read_file(usrnum, clsnum)

    ues = GC.fineHomeCloudlet(cls, uesOld)

    models = GC.generateModels(num_models, services)

    t2 = None
    cloudlets, locations = GC.generateCloudlets(cls, ues, t2)
    graph = GC.connectNode(locations, ues)

    t1 = None
    tcost = None
    requests = GC.generateRequests(cls, ues, request_num_usr, services, t1, tcost, cloudlets)

    num_locations = len(locations)
    num_cloudlets = len(cloudlets)

    # homerequest = {}
    # for l in range(1, num_cloudlets + 1):
    #     homerequest[l] = []
    #     for j in range(1, num_requests + 1):
    #         if requests[j]['home_cloudlet_id'] == l:
    #             homerequest[l].append(j)

    accuracy_dict = Accuracy.get_accuracy(models, requests)

    xi = 0.98

    # Initialize location dictionaries
    locations_dict = initialize_model_locations(num_models, num_cloudlets, clsnum, models)

    # -----------------------------online---------------------------------#

    ol_results = OL.OL(num_requests, num_models, num_features, requests,
                       models, cloudlets, locations,
                       graph, accuracy_dict, xi, contexts,
                       feature_limit, alpha)

    ol_metrics['convergence'].append(ol_results[4])
    ol_metrics['penalty_con'].append(ol_results[5])

    mab_results = MAB.mab(num_requests, num_models, num_features, requests,
                          models, cloudlets, locations,
                          graph, accuracy_dict, xi,
                          feature_limit, alpha)

    mab_metrics['convergence'].append(mab_results[4])
    mab_metrics['penalty_con'].append(mab_results[5])


    adms_results = ADMS.adms(num_requests, num_models, num_features, requests,
                             models, cloudlets, locations,
                             graph, accuracy_dict, xi, contexts,
                             feature_limit, alpha)

    adms_metrics['convergence'].append(adms_results[4])
    adms_metrics['penalty_con'].append(adms_results[5])

    print("#-----------------------Convergence---------------------")
    print("ol=", ol_metrics['convergence'][0]) # regret
    print("mab=", mab_metrics['convergence'][0])
    print("adms=", adms_metrics['convergence'][0])

    print("#-----------------------Convergence_penalty---------------------")
    print("ol=", ol_metrics['penalty_con'][0])
    print("mab=", mab_metrics['penalty_con'][0])
    print("adms=", adms_metrics['penalty_con'][0])

    from Plot import regret as R
    from Plot import penalty as P
    R.regret_plot(ol_metrics['convergence'][0], mab_metrics['convergence'][0], adms_metrics['convergence'][0])
    P.penalty_plot(ol_metrics['penalty_con'][0], mab_metrics['penalty_con'][0], adms_metrics['penalty_con'][0])

if __name__ == '__main__':
    run_experiment()