import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm
from Compute import Delay, Accuracy,Cost
from Graph import ParameterSetting as PS
import time

class BetaDistribution:
    def __init__(self, S, F):
        self.S = S
        self.F = F

    def update(self, r, n):
        self.S = self.S + r
        self.F = self.F + n - r

    def sample(self):
        return np.random.beta(self.S, self.F)

class GassuianDistribution:

    def __init__(self, selected_features, B, mu_hat, v):
        self.mean = np.dot(selected_features.T, mu_hat)
        self.variance = v**2 * np.dot(np.dot(selected_features.T, np.linalg.inv(B)), selected_features)

    def sample(self):
        return np.random.normal(self.mean, np.sqrt(self.variance))

def OL(num_requests, num_models, num_features, requests, models, cloudlets, locations, Graph,accuracy_dict,xi,context,feature_limit,alpha):

    penalty_baseline = random.uniform(100,300) # Todo: computing the penalty
    threshold = 20

    # beta initialization
    S = {}
    F = {}
    n = {}
    r = {}

    beta_dict = {}
    for l in range(1, num_features + 1):
        S[l] = 1
        F[l] = 1
        n[l] = 0
        r[l] = 0
        beta_dict[l] = BetaDistribution(S[l], F[l])

    # normal initialization
    B = {}
    mu_hat = {}
    g = {}
    v = 3  # Todo: update v
    lsh = {}
    lre = {}
    for k in range(1, num_models + 1):

        B[k] = np.eye(feature_limit)
        mu_hat[k] = np.zeros(feature_limit)
        g[k] = np.zeros(feature_limit)
        lsh[k] = []
        lre[k] = []

    # gaussian_dict ={}
    sumcost = 0
    sumdelay = 0
    sumaccuracy = 0
    sumpenalty = 0
    convergence = []
    penalty_con = []
    weightsum = 0
    model_loc_list = {}
    start1  = 0
    end1  = 0
    start = time.time()
    featurelist = {}
    for j in range(1, num_requests + 1):

        "feature selection"
        sampled_mus = {}  # store the mu value
        start1 = time.time()
        for l in range(1, num_features + 1):
            beta_dict[l].update(r[l], n[l])
            mu = beta_dict[l].sample()
            sampled_mus[l] = mu
        sorted_features = sorted(sampled_mus.keys(), key=lambda x: sampled_mus[x], reverse=True)
        selected_features = sorted_features[:feature_limit]

        selected_features_copy = np.array(selected_features).reshape(-1, 1)  # 5*1
        featurelist[j] = []
        b_d = []
        for f_index in selected_features_copy:
            b_d.append(context[j-1][f_index-1])
            featurelist[j].append(f_index.tolist()[0])
        b_d = np.array(b_d).reshape(-1, 1)


        "arm selection function"
        sampled_theta = {}  # store the theta value
        gaussian_dict = {}
        optimal = {}

        for k in range(1, num_models + 1):

            gaussian_dict[k] = GassuianDistribution(b_d, B[k], mu_hat[k], v)

            theta = gaussian_dict[k].sample()
            sampled_theta[k] = theta

            #------------------compute optimal penalty-------------------
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

            # Obtain the penalty
            inference_delay = Delay.get_inference_delay(models[k], requests[j], cloudlets)
            pull_delay = max(pullsh, pullre)
            queue_delay = Delay.get_queue_delay( requests[j], cloudlets)
            trans_delay = Delay.get_trans_delay(models[k], requests[j])
            delay = inference_delay + pull_delay + queue_delay + trans_delay

            # accuracy = Accuracy.get_accuracy(models[k], requests[j])
            accuracy = accuracy_dict[j][k]

            cost = Cost.get_cost(models[k], requests[j], cloudlets, pull_delay, inference_delay, trans_delay)

            penalty = xi * (-np.log(accuracy)) + (1 - xi) * delay
            reward = 1 / penalty
            optimal[k] = penalty
        optimal_penalty = min(optimal.values())
        end1 = time.time()

        selected_model = min(sampled_theta, key=sampled_theta.get)

        if selected_model:
            pulling_delay_sh = {}
            pulling_delay_re = {}
            k = selected_model
            if lsh[k]:
                pullshdelay = []
                for l in lsh[k]:
                    pulling_delay_sh = Delay.get_pulling_delay_sh(models[k], requests[j], cloudlets[l],Graph)
                    pullshdelay.append(pulling_delay_sh)
                pullsh = min(pullshdelay)
            else:
                location_key = list(locations.keys())[-1]
                pullsh = Delay.get_pull_cloud_sh(models[k], requests[j], location_key, Graph)

            if lre[k]:
                pullredelay = []
                for l in lre[k]:
                    pulling_delay_re = Delay.get_pulling_delay_re(models[k], requests[j], cloudlets[l],Graph)
                    pullredelay.append(pulling_delay_re)
                pullre = min(pullredelay)
            else:
                location_key = list(locations.keys())[-1]
                pullre = Delay.get_pull_cloud_re(models[k], requests[j],location_key, Graph)

            lsh[k].append(requests[j]['home_cloudlet_id'])
            lre[k].append(requests[j]['home_cloudlet_id'])

            for k1 in range(1, num_models + 1):
                if models[k1]['service_type'] == models[k]['service_type']:
                    if k1 != k:
                        lsh[k1].append(requests[j]['home_cloudlet_id'])
                        lsh[k1] = list(set(lsh[k1]))

            inference_delay = Delay.get_inference_delay(models[k], requests[j], cloudlets)
            pull_delay = max(pullsh, pullre)
            queue_delay = Delay.get_queue_delay( requests[j], cloudlets)
            trans_delay = Delay.get_trans_delay(models[k], requests[j])
            # print("pull_delay",pull_delay)
            # print("queue_delay",queue_delay)
            # print('inference_delay',inference_delay)
            # print('trans_delay',trans_delay)
            delay = inference_delay + pull_delay + queue_delay + trans_delay

            accuracy = accuracy_dict[j][k]

            cost = Cost.get_cost(models[k], requests[j], cloudlets, pull_delay, inference_delay,trans_delay)

            sumcost += cost
            sumdelay += delay
            sumaccuracy += accuracy
            penalty = xi * (-np.log(accuracy)) + (1-xi) * delay
            reward = 1/ penalty

            sumpenalty += penalty

            weightsum += penalty

            B[k] += np.outer(b_d,b_d)
            g[k] += b_d.flatten() * penalty
            mu_hat[k] = np.linalg.solve(B[k], g[k])

            if penalty <= penalty_baseline + threshold:
                for l in selected_features:
                    n[l] = n[l] + 1
                    r[l] = r[l] + 1
            regret = penalty - optimal_penalty
            convergence.append(regret)
            penalty_con.append(penalty)

    end = time.time()
    timeo = end1-start1
    sumtime = end - start - timeo

    return sumaccuracy/num_requests, sumdelay, sumcost/num_requests,sumtime,convergence,penalty_con,weightsum,featurelist


def draw(num_features, num_models, beta_dict, gaussian_dict):
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 30,
             }
    plt.figure(figsize=(12,6))

    for l in range(1, num_features + 1):
        plt.subplot(2, 2, l)
        alpha1 = beta_dict[l].S
        beta1 = beta_dict[l].F
        # x = sampled_mus[l]
        x = np.linspace(0,1,1000)
        pdf = beta.pdf(x,alpha1,beta1)
        # plt.plot(x,pdf,'r-',lw=2, label=f'Feature {l},alpha={alpha1},beta={beta1}')
        plt.plot(x, pdf, 'r-', lw=2, label=f'Feature {l}')
        # plt.xlabel('Sampled values')
        # plt.ylabel('Probability density')
        plt.title('Beta Distribution', font1)
        plt.legend(prop=font1)
        # legend = plt.legend(handles=[A], prop=font1, ncol=3, labelspacing=0.02, handlelength=1, handleheight=1, handletextpad=0.5, columnspacing=0.5, borderaxespad=0.1, borderpad=0.2)
        plt.grid(True)

    plt.figure(figsize=(12, 6))

    for k in range(1, num_models + 1):
        plt.subplot(2, 2, k)
        mean1 =  gaussian_dict[k].mean
        variance1 =  gaussian_dict[k].variance
        # print("Gaussian distribution ~({},{})".format(mean1,variance1))
        x = np.linspace(-10, 500, 100)
        pdf = 1 / np.sqrt(2 * np.pi * variance1) * np.exp(
            -0.5 * ((x - mean1)** 2 / variance1) )
        pdf = np.squeeze(pdf)
        # plt.plot(x, pdf, 'r-', lw=2, label=f'Feature {k},mean={mean1},variance={variance1}')
        plt.plot(x, pdf, 'r-', lw=2, label=f'Model {k}')

        # plt.xlabel('Sampled values')
        # plt.ylabel('Probability density')
        plt.title('Gaussian Distribution', font1)
        plt.legend(prop=font1)
        plt.grid(True)

    plt.tight_layout()
    plt.show()
