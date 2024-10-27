import random
from Compute import Delay, Cost
import time

def rand_pick(seq, probabilities):
    x = random.uniform(0,1)
    cumprob = 0.0
    item = 0
    for item, item_pro in zip(seq, probabilities):
        cumprob += item_pro
        if x < cumprob:
            break
    return item

def nbs(num_requests,num_models,num_locations,x,y_sh,y_re,requests, models, cloudlets,locations,Graph,accuracy_dict,lsh,lre,timenbsilp,alpha):
    start = time.time()
    group = {}

    for j in range(1, num_requests+1):
        group[j] = {}
        group[j]['x'] = []
        for k in range(1, num_models+1):
            group[j]['x'].append(x[(j,k)] * 1/2)
            group[j][k] = {}
            group[j][k]['y_sh'] = []
            group[j][k]['y_re'] = []
            for l in lsh[k]:
                group[j][k]['y_sh'].append(y_sh[(j, k, l)] * 1 / 2)
            for l in lre[k]:
                group[j][k]['y_re'].append(y_re[(j, k, l)] * 1 / 2)

    sumdelay = 0
    sumaccuracy = 0
    sumcost = 0
    # Randomized rounding for model selection and pulling decisions
    for j in range(1, num_requests + 1):
        model_id = None
        pull_sh_id = None
        pull_re_id = None

        model_id = rand_pick(range(1, num_models + 1), group[j]['x'])

        if not model_id:
            print("Warning: No model selected for request", j)

        else:
            pull_sh_id = rand_pick(lsh[model_id], group[j][model_id]['y_sh'])
            pull_re_id = rand_pick(lre[model_id], group[j][model_id]['y_re'])

        acc = accuracy_dict[j][model_id]
        sumaccuracy += acc

        pulling_delay_sh = Delay.get_pulling_delay_sh(models[model_id], requests[j], locations[pull_sh_id], Graph)
        pulling_delay_re = Delay.get_pulling_delay_re(models[model_id], requests[j], locations[pull_re_id], Graph)
        pull_delay = max(pulling_delay_sh, pulling_delay_re)
        inference_delay = Delay.get_inference_delay(models[model_id], requests[j], cloudlets)
        queue_delay = Delay.get_queue_delay( requests[j], cloudlets)
        trans_delay = Delay.get_trans_delay(models[model_id], requests[j])
        sumdelay += inference_delay + pull_delay + queue_delay + trans_delay
        cost = Cost.get_cost(models[model_id], requests[j], cloudlets, pull_delay, inference_delay, trans_delay)
        sumcost += cost

    end = time.time()
    sumtime = end-start

    return sumaccuracy/num_requests, sumdelay, sumcost/num_requests,sumtime+timenbsilp