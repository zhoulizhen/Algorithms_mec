from Compute import Delay,Cost
import time

def nonshare(num_requests, num_models, num_features, requests, models, cloudlets,locations, Graph, accuracy_dict,lsh,lre,alpha):

    start = time.time()
    sumaccuracy = 0
    sumdelay = 0
    sumcost = 0

    for j in range(1, num_requests+1):
        modellist = []
        for k in range(1, num_models+1):
            acc = accuracy_dict[j][k]
            modellist.append(acc)
        model_id = modellist.index(max(modellist)) + 1
        acc = accuracy_dict[j][model_id]
        sumaccuracy += acc

        if model_id:

            k = model_id

            if lsh[k]:
                pullshdelay = []
                for l in lsh[k]:
                    pulling_delay_sh = Delay.get_pulling_delay_sh(models[k], requests[j], cloudlets[l], Graph)
                    pullshdelay.append(pulling_delay_sh)
                pullsh = min(pullshdelay)
            else:
                location_key = list(locations.keys())[-1]
                pullsh = Delay.get_pull_cloud_sh(models[k], requests[j], location_key, Graph)

            if lre[k]:
                pullredelay = []
                for l in lre[k]:
                    pulling_delay_re = Delay.get_pulling_delay_re(models[k], requests[j], cloudlets[l], Graph)
                    pullredelay.append(pulling_delay_re)
                pullre = min(pullredelay)
            else:
                location_key = list(locations.keys())[-1]
                pullre = Delay.get_pull_cloud_re(models[k], requests[j], location_key, Graph)

            inference_delay = Delay.get_inference_delay(models[k], requests[j], cloudlets)
            pull_delay = max(pullsh, pullre)

            queue_delay = Delay.get_queue_delay( requests[j], cloudlets)

            trans_delay = Delay.get_trans_delay(models[k], requests[j])

            delay = inference_delay + pull_delay + queue_delay + trans_delay

            cost = Cost.get_cost(models[k], requests[j], cloudlets, pull_delay, inference_delay, trans_delay)

            sumcost += cost
            sumdelay += delay

    end = time.time()
    sumtime = end-start

    return sumaccuracy / num_requests, sumdelay, sumcost / num_requests, sumtime