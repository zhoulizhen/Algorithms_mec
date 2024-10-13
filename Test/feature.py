import random

from Graph import FileReader as FR
from Graph import ConnectMEC as GC
from Compute import Accuracy
from Algorithm import OL, MAB, ADMS, NBSRound, NBSILP, RelaxILP, App, SC, NonShare, RelaxNonlinear, Nonlinear, ILP
import matplotlib.pyplot as plt
import numpy as np


def test1():
    random.seed(123)


    ol = {}
    # ol['convergence'] = []

    mab = {}
    # mab['convergence'] = []

    adms = {}
    # adms['convergence'] = []


    clsnum = 100
    num_models = 50  # The number of models
    request_num_usr = 1  # The number of requests per user
    num_features = 50
    feature_limit = num_features // 2

    usrnum = 100
    num_requests = usrnum * request_num_usr  # The number of requests
    services = [i for i in range(num_models // 5)]  # The number of service types

    import context
    contexts = context.generate_context(num_requests, num_features)
    # context = {j: np.random.rand(num_features) for j in range(1, num_requests + 1)}
    # context = np.random.rand(num_features)
    # print("context", contexts)

    xi = random.uniform(0, 1)
    alpha = random.uniform(0, 1)

    cls, uesOld = FR.read_file(usrnum, clsnum)
    # GCM.constructMap(cls, uesOld)
    ues = GC.fineHomeCloudlet(cls, uesOld)  # Add home cloudlet

    models = GC.generateModels(num_models, services)  # Generate models

    t2 = None
    cloudlets, locations = GC.generateCloudlets(cls, ues, t2)  # Generate cloudlets
    Graph = GC.connectNode(locations, ues)  # Connect cloudlets

    t1 = None
    tcost = None
    requests = GC.generateRequests(cls, ues, request_num_usr, services, t1, tcost, cloudlets)  # Generate requests

    num_locations = len(locations)
    num_cloudlets = len(cloudlets)

    homerequest = {}
    for l in range(1, num_cloudlets + 1):
        homerequest[l] = []
        for j in range(1, num_requests + 1):
            if requests[j]['home_cloudlet_id'] == l:
                homerequest[l].append(j)
    # print("homerequest", homerequest)

    accuracy_dict = Accuracy.get_accuracy(models, requests)

    xi = 0.98

    lsh = {}
    lre = {}
    lshcopy = {}
    lk = {}
    lshnon = {}
    for k in range(1, num_models + 1):
        # todo: notice the setting of service type and the number of locations
        num_po_locations = random.randint(1, clsnum // 10)  # how many locations the model k is stored
        lk[k] = random.sample(range(1, num_cloudlets + 1), num_po_locations)
        lre[k] = lk[k]  # the locations where the remaining subset of model k is stored
        lshnon[k] = lk[k]  # this is for algorithms without parameter sharing

    for k1 in range(1, num_models + 1):
        lshcopy[k1] = []
        for l in lk[k1]:  # add itself locations of model k1
            lshcopy[k1].append(l)
        for k2 in range(1, num_models + 1):  # find other locations
            if models[k1]['service_type'] == models[k2]['service_type']:
                for l in lk[k2]:
                    lshcopy[k1].append(l)
        lsh[k1] = list(set(lshcopy[k1]))

    # -----------------------------online---------------------------------#

    acco, delayo, costo, timeo,convergence,penalty_con,wightsum,featurelistol = OL.OL(num_requests, num_models, num_features, requests, models, cloudlets,locations, Graph, accuracy_dict, xi, contexts, feature_limit,alpha)
    # ol['convergence']=convergence
    # ol['penalty_con']=penalty_con

    # accm, delaym, costm, timem,convergencem,penalty_conm,wightsum = MAB.mab(num_requests, num_models, num_features, requests, models, cloudlets, locations, Graph, accuracy_dict, xi, feature_limit,homerequest,alpha)
    # mab['convergence']=convergencem
    # mab['penalty_con']=penalty_conm

    accad, delayad, costad, timead,convergencead,penalty_conad,wightsum,featurelistad = ADMS.adms(num_requests, num_models, num_features, requests, models, cloudlets, locations, Graph, accuracy_dict, xi, contexts, feature_limit,alpha)

    # adms['convergence']=convergencead
    # adms['penalty_con']=penalty_conad

    # from Plot import regret as R
    # from Plot import penalty as P
    # R.regret_plot(ol['convergence'], mab['convergence'], adms['convergence'])
    # P.penalty_plot(ol['penalty_con'], mab['penalty_con'], adms['penalty_con'])

    for j in range(1, num_requests + 1):
        print("Feature List OL: ", featurelistol[j])
        # print("Feature List ADMS: ", featurelistad[j])



    import matplotlib.pyplot as plt
    import numpy as np

    # feature_listol = [[i for i in range(num_requests)], [featurelistol[j] for j in range(1, num_requests+1)]]
    # feature_listad = [[i for i in range(num_requests)], [featurelistad[j] for j in range(1, num_requests + 1)]]

    x = []
    for j in range(1, num_requests+1):
        x.append(featurelistol[j])
    # plt.imshow(x, cmap='Blues')

    import numpy as np

    # 创建一个二维数组，表示是否选择了对应的特征
    data_array = np.zeros((len(x), num_features))
    # print("data_array", data_array)

    for i, features in enumerate(x):
        print("i, features", i, features)
        for feature in features:
            print("feature", feature)
            data_array[i][feature - 1] = 1

    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.imshow(data_array, cmap='Blues', aspect='auto')

    # 添加标题和坐标轴标签
    plt.title('Selected Features by Requests')
    plt.xlabel('Request Index')
    plt.ylabel('Feature Index')

    # 设置横坐标刻度
    # plt.xticks(np.arange(num_requests), np.arange(1, 11))

    # 设置纵坐标刻度
    # plt.yticks(np.arange(num_features), np.arange(1, num_features + 1))

    # 显示图表
    plt.colorbar(label='Selected')  # 添加颜色条
    plt.show()


if __name__ == '__main__':
    test1()