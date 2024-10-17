import numpy as np
from Compute import Delay,Cost
import time
import random
class EpsilonGreedyBandit:

    def __init__(self, num_arms, epsilon):
        self.arms = num_arms
        self.epsilon = epsilon
        self.action_counts = {arm: 0 for arm in range(1, num_arms + 1)}
        self.q_values = {arm: 0.0 for arm in range(1, num_arms + 1)}

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(list(self.action_counts.keys()))
            return action
        else:
            max_q_value = max(self.q_values.values())
            best_actions = [arm for arm, q_value in self.q_values.items() if q_value == max_q_value]
            return random.choice(best_actions)

    def update(self, action, reward):
        self.action_counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]

def mab(num_requests, num_models, num_features, requests, models, cloudlets,locations, Graph, accuracy_dict,xi,feature_limit,alpha):

    lsh = {}
    lre = {}
    for k in range(1, num_models + 1):
        lsh[k] = []
        lre[k] = []

    epsilon = 0.5
    bandit = EpsilonGreedyBandit(num_models, epsilon)

    sumcost = 0
    sumdelay = 0
    sumaccuracy = 0
    convergence = []
    penalty_con = []
    weightsum = 0
    start1 = 0
    end1 = 0
    start = time.time()
    featurelist = {}
    for j in range(1, num_requests + 1):
        optimal = {}
        start1 = time.time()
        for k in range(1, num_models + 1):
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
            queue_delay = Delay.get_queue_delay(requests[j], cloudlets)
            trans_delay = Delay.get_trans_delay(models[k], requests[j])
            delay = inference_delay + pull_delay + queue_delay + trans_delay
            accuracy = accuracy_dict[j][k]
            penalty = xi * (-np.log(accuracy)) + (1 - xi) * delay
            optimal[k] = penalty
        optimal_penalty = min(optimal.values())
        end1 = time.time()
        model_id = bandit.choose_action()
        if model_id:
            pulling_delay_sh = 0
            pulling_delay_re = 0
            k = model_id

            if lsh[k]:
                for l in lsh[k]:
                    pulling_delay_sh = Delay.get_pulling_delay_sh(models[k], requests[j], cloudlets[l], Graph)
            else:
                location_key = list(locations.keys())[-1]
                pulling_delay_sh = Delay.get_pull_cloud_sh(models[k], requests[j], location_key, Graph)

            if lre[k]:
                for l in lre[k]:
                    pulling_delay_re = Delay.get_pulling_delay_re(models[k], requests[j], cloudlets[l], Graph)
            else:
                location_key = list(locations.keys())[-1]
                pulling_delay_re = Delay.get_pull_cloud_re(models[k], requests[j], location_key, Graph)

            lsh[k].append(requests[j]['home_cloudlet_id'])
            lre[k].append(requests[j]['home_cloudlet_id'])

            inference_delay = Delay.get_inference_delay(models[k], requests[j], cloudlets)
            pull_delay = max(pulling_delay_sh, pulling_delay_re)
            queue_delay = Delay.get_queue_delay( requests[j], cloudlets)
            trans_delay = Delay.get_trans_delay(models[k], requests[j])
            delay = inference_delay + pull_delay + queue_delay + trans_delay

            accuracy = accuracy_dict[j][k]
            cost = Cost.get_cost(models[k], requests[j], cloudlets, pull_delay, inference_delay, trans_delay)
            sumcost += cost
            sumdelay += delay
            sumaccuracy += accuracy
            penalty = xi * (-np.log(accuracy)) + (1-xi) * delay
            reward = 1/penalty
            bandit.update(model_id, reward)
            regret = penalty - optimal_penalty
            convergence.append(regret)
            penalty_con.append(penalty)
            weightsum += penalty

    end = time.time()
    timeo = end1-start1
    sumtime = end-start- timeo

    return sumaccuracy/num_requests, sumdelay, sumcost/num_requests, sumtime, convergence, penalty_con,weightsum, featurelist