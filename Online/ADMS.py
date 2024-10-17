import random
import numpy as np
from Compute import Delay, Cost
import time

class LinearUCB:
    def __init__(self, n_features, n_arms, alpha, delta):
        self.n_features = n_features
        self.n_arms = n_arms
        self.alpha = alpha
        self.delta = delta

        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros((n_features, 1)) for _ in range(n_arms)]
        self.theta_hat = [np.zeros((n_features, 1)) for _ in range(n_arms)]
        self.C_theta_hat = [np.eye((n_features)) for _ in range(n_arms)]

    def choose_action(self, context_vector,alpha):
        UCB_values = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            x_t_a = context_vector
            theta_hat_a = self.theta_hat[a].reshape(-1, 1)
            C_theta_hat_a = self.C_theta_hat[a]
            UCB_values[a] = float(x_t_a.T @ theta_hat_a) + self.alpha * np.sqrt(float(x_t_a.T @ C_theta_hat_a @ x_t_a))

        if random.random() > alpha:
            chosen_action = random.randint(1, self.n_arms)
        else:
            chosen_action = np.argmax(UCB_values) + 1
        return chosen_action

    def update_parameters(self, chosen_model, context_vector, reward):
        chosen_model = chosen_model - 1
        self.A[chosen_model] += np.outer(context_vector, context_vector)
        self.b[chosen_model] += reward * context_vector.reshape(-1, 1)
        self.theta_hat[chosen_model] = np.linalg.inv(self.A[chosen_model]) @ self.b[chosen_model]
        self.C_theta_hat[chosen_model] = np.linalg.inv(self.delta * np.eye(self.n_features) + self.A[chosen_model])

def adms(num_requests, num_models, num_features, requests, models, cloudlets,locations, Graph, accuracy_dict, xi,context,feature_limit,alpha):

    alpha = 0.3
    delta = 1
    linear_ucb = LinearUCB(feature_limit, num_models, alpha, delta)

    lsh = {}
    lre = {}
    for k in range(1, num_models + 1):
        lsh[k] = []
        lre[k] = []

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
    for j in range(1,num_requests+1):
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
            queue_delay = Delay.get_queue_delay( requests[j], cloudlets)
            trans_delay = Delay.get_trans_delay(models[k], requests[j])
            delay = inference_delay + pull_delay + queue_delay + trans_delay
            accuracy = accuracy_dict[j][k]
            penalty = xi * (-np.log(accuracy)) + (1 - xi) * delay
            optimal[k] = penalty
        optimal_penalty = min(optimal.values())
        end1 = time.time()
        selected_features = random.sample(range(0, num_features), feature_limit)
        featurelist[j] = []
        for f in selected_features:
            featurelist[j].append(f)
        context_vector = context[j-1][selected_features]
        context_vector = context_vector.reshape(-1, 1)
        model_id = linear_ucb.choose_action(context_vector,alpha)

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
            penalty = xi * (-np.log(accuracy)) + (1 - xi) * delay
            reward = 1 / penalty
            linear_ucb.update_parameters(model_id, context_vector, reward)
            regret = penalty - optimal_penalty
            convergence.append(regret)
            penalty_con.append(penalty)
            weightsum += penalty

    end = time.time()
    timeo = end1 - start1
    sumtime = end - start - timeo

    return sumaccuracy / num_requests, sumdelay, sumcost / num_requests, sumtime,convergence, penalty_con, weightsum, featurelist
