import random

from Graph import FileReader as FR
from Graph import ConnectMEC as GC
from Compute import Accuracy
from Algorithm import OL, MAB, ADMS, NBSRound, NBSILP, RelaxILP, App, SC, NonShare, RelaxNonlinear, Nonlinear, ILP
import matplotlib.pyplot as plt
import numpy as np


def test1():
    random.seed(123)

    ilp = {}
    ilp['accuracy'] = []
    ilp['delay'] = []
    ilp['cost'] = []
    ilp['time'] = []

    nonlinear = {}
    nonlinear['accuracy'] = []
    nonlinear['delay'] = []
    nonlinear['cost'] = []
    nonlinear['time'] = []

    relaxnonlinear = {}
    relaxnonlinear['accuracy'] = []
    relaxnonlinear['delay'] = []
    relaxnonlinear['cost'] = []
    relaxnonlinear['time'] = []

    relaxilp = {}
    relaxilp['accuracy'] = []
    relaxilp['delay'] = []
    relaxilp['cost'] = []
    relaxilp['time'] = []

    app = {}
    app['accuracy'] = []
    app['delay'] = []
    app['cost'] = []
    app['time'] = []

    ol = {}
    ol['accuracy'] = []
    ol['delay'] = []
    ol['cost'] = []
    ol['time'] = []
    ol['weight'] = []

    nonshare = {}
    nonshare['accuracy'] = []
    nonshare['delay'] = []
    nonshare['cost'] = []
    nonshare['time'] = []


    sc = {}
    sc['accuracy'] = []
    sc['delay'] = []
    sc['cost'] = []
    sc['time'] = []

    mab = {}
    mab['accuracy'] = []
    mab['delay'] = []
    mab['cost'] = []
    mab['time'] = []
    mab['weight'] = []

    adms = {}
    adms['accuracy'] = []
    adms['delay'] = []
    adms['cost'] = []
    adms['time'] = []
    adms['weight'] = []

    nbs = {}
    nbs['accuracy'] = []
    nbs['delay'] = []
    nbs['cost'] = []
    nbs['time'] = []

    iteration = [10,30,50,70,90]
    num_models = 100  # The number of models
    request_num_usr = 1  # The number of requests per user
    num_features = 100
    feature_limit = num_features // 5
    usrnum = 100
    clsnum = 100
    for t in iteration:

        print("iteration", t)

        # usrnum = 100
        # clsnum = t
        num_requests = usrnum * request_num_usr  # The number of requests
        services = [i for i in range(num_models // 2)]  # The number of service types

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

        cloudlets, locations = GC.generateCloudlets(cls, ues, t)  # Generate cloudlets
        Graph = GC.connectNode(locations, ues)  # Connect cloudlets

        t1 = None
        tcost = None
        requests = GC.generateRequests(cls, ues, request_num_usr, services,t1,tcost,cloudlets)


        num_locations = len(locations)
        num_cloudlets = len(cloudlets)

        # homerequest = {}
        # for l in range(1, num_cloudlets + 1):
        #     homerequest[l] = []
        #     for j in range(1, num_requests + 1):
        #         if requests[j]['home_cloudlet_id'] == l:
        #             homerequest[l].append(j)
        # # print("homerequest", homerequest)

        accuracy_dict = Accuracy.get_accuracy(models, requests)

        xi = 0.98

        lsh = {}
        lre = {}
        lshcopy = {}
        lk = {}
        lshnon = {}
        for k in range(1, num_models + 1):
            # todo: notice the setting of service type and the number of locations
            num_po_locations = random.randint(1, clsnum // 20)  # how many locations the model k is stored
            lk[k] = random.sample(range(1, num_cloudlets + 1), num_po_locations)
            lre[k] = lk[k]  # the locations where the remaining subset of model k is stored
            lshnon[k] = lk[k] # this is for algorithms without parameter sharing

        for k1 in range(1, num_models + 1):
            lshcopy[k1] = []
            for l in lk[k1]:  # add itself locations of model k1
                lshcopy[k1].append(l)
            for k2 in range(1, num_models + 1):  # find other locations
                if models[k1]['service_type'] == models[k2]['service_type']:
                    for l in lk[k2]:
                        lshcopy[k1].append(l)
            lsh[k1] = list(set(lshcopy[k1]))

        # for k in range(1,num_models+1):
        #     print("model",k)
        #     print("lsh",lsh[k])
        #     print("lre",lre[k])
            # print("lshnon",lshnon[k])

        # accn, delayn, costn,timen= Nonlinear.nonlinear(num_requests, num_models, num_locations, requests, models, cloudlets, locations, Graph, accuracy_dict, xi, alpha)
        # nonlinear['accuracy'].append(accn)
        # nonlinear['delay'].append(delayn)
        # nonlinear['cost'].append(costn)
        # nonlinear['time'].append(timen)
        #
        #
        # accrn, delayrn, costrn, timern = RelaxNonlinear.relaxnonlinear(num_requests, num_models, num_locations, requests, models, cloudlets,locations, Graph, accuracy_dict, xi, alpha)
        # relaxnonlinear['accuracy'].append(accrn)
        # relaxnonlinear['delay'].append(delayrn)
        # relaxnonlinear['cost'].append(costrn)
        # relaxnonlinear['time'].append(timern)

        # acci, delayi, costi, timei = ILP.ilp(num_requests, num_models, num_locations, requests, models, cloudlets,locations, Graph, accuracy_dict, xi, alpha)
        # ilp['accuracy'].append(acci)
        # ilp['delay'].append(delayi)
        # ilp['cost'].append(costi)
        # ilp['time'].append(timei)

        # -----------------------------offline---------------------------------#
        # parameter sharing
        accr, delayr, costr, timer,x_values, y_sh_values, y_re_values = RelaxILP.relaxILP(num_requests, num_models, num_locations, requests, models, cloudlets, locations, Graph, accuracy_dict, xi,alpha, lsh,lre)
        relaxilp['accuracy'].append(accr)
        relaxilp['delay'].append(delayr)
        relaxilp['cost'].append(costr)
        relaxilp['time'].append(timer)

        # parameter sharing
        acca, delaya, costa, timea = App.app(num_requests,num_models,num_locations,x_values, y_sh_values, y_re_values,requests, models, cloudlets,locations,Graph,accuracy_dict,lsh,lre,alpha)
        app['accuracy'].append(acca)
        app['delay'].append(delaya)
        app['cost'].append(costa)
        app['time'].append(timea)

        # NON parameter sharing
        accnbsilp, delaynbsilp, costnbsilp, timenbsilp, x_valuesnbs, y_sh_valuesnbs, y_re_valuesnbs = NBSILP.relaxILP(num_requests, num_models, num_locations, requests, models, cloudlets, locations, Graph, accuracy_dict, xi,alpha,lshnon,lre)

        # non parameter sharing
        accanb, delayanb, costanb, timeanb = NBSRound.nbs(num_requests,num_models,num_locations,x_valuesnbs, y_sh_valuesnbs, y_re_valuesnbs,requests, models, cloudlets,locations,Graph,accuracy_dict,lshnon,lre,timenbsilp,alpha)

        nbs['accuracy'].append(accanb)
        nbs['delay'].append(delayanb)
        nbs['cost'].append(costanb)
        nbs['time'].append(timeanb)

        # non parameter sharing
        accn, delayn, costn, timen = NonShare.nonshare(num_requests, num_models, num_features, requests, models, cloudlets,locations, Graph, accuracy_dict,lshnon,lre,alpha)
        nonshare['accuracy'].append(accn)
        nonshare['delay'].append(delayn)
        nonshare['cost'].append(costn)
        nonshare['time'].append(timen)

        # non parameter sharing
        accs, delays, costs, times = SC.sc(num_requests, num_models, num_features, requests, models, cloudlets,locations, Graph, accuracy_dict,lshnon,lre,alpha)
        sc['accuracy'].append(accs)
        sc['delay'].append(delays)
        sc['cost'].append(costs)
        sc['time'].append(times)

        # -----------------------------online---------------------------------#
        # todo: add parameter sharing
        acco, delayo, costo, timeo,convergence,penalty,wightsum,featurelist = OL.OL(num_requests, num_models, num_features, requests, models, cloudlets,
                                           locations, Graph, accuracy_dict, xi, contexts, feature_limit,alpha)
        ol['accuracy'].append(acco)
        ol['delay'].append(delayo)
        ol['cost'].append(costo)
        ol['time'].append(timeo)
        ol['weight'].append(wightsum)

        accm, delaym, costm, timem,convergencem,penaltym,wightsum ,featurelist= MAB.mab(num_requests, num_models, num_features, requests, models, cloudlets,
                                             locations, Graph, accuracy_dict, xi, feature_limit,alpha)
        mab['accuracy'].append(accm)
        mab['delay'].append(delaym)
        mab['cost'].append(costm)
        mab['time'].append(timem)
        mab['weight'].append(wightsum)

        accad, delayad, costad, timead,convergencead,penaltyad,wightsum ,featurelist= ADMS.adms(num_requests, num_models, num_features, requests, models, cloudlets,
                                                   locations, Graph, accuracy_dict, xi, contexts, feature_limit,alpha)
        adms['accuracy'].append(accad)
        adms['delay'].append(delayad)
        adms['cost'].append(costad)
        adms['time'].append(timead)
        adms['weight'].append(wightsum)

    print("#-----------------------Delay---------------------")
    print("relaxilpdelay=", relaxilp['delay'])
    print("appdelay=", app['delay'])
    print("nbsdelay=", nbs['delay'])
    print("nonsharedelay=", nonshare['delay'])
    print("scdelay=", sc['delay'])


    print("oldelay=", ol['delay'])
    print("mabdelay=", mab['delay'])
    print("admsdelay=", adms['delay'])

    print("#-----------------------Accuracy---------------------")
    print("relaxilpacc=", relaxilp['accuracy'])
    print("appacc=", app['accuracy'])
    print("nbsacc=", nbs['accuracy'])
    print("nonshareacc=", nonshare['accuracy'])
    print("scacc=", sc['accuracy'])

    print("olacc=", ol['accuracy'])
    print("mabacc=", mab['accuracy'])
    print("admsacc=", adms['accuracy'])

    print("#-----------------------Cost---------------------")
    print("relaxilpcost=", relaxilp['cost'])
    print("appcost=", app['cost'])
    print("nbscost=", nbs['cost'])
    print("nonsharecost=", nonshare['cost'])
    print("sccost=", sc['cost'])

    print("olcost=", ol['cost'])
    print("mabcost=", mab['cost'])
    print("admscost=", adms['cost'])

    print("#-----------------------Time---------------------")
    print("relaxilptime=", relaxilp['time'])
    print("apptime=", app['time'])
    print("nbstime=", nbs['time'])
    print("nonsharetime=", nonshare['time'])
    print("sctime=", sc['time'])

    print("oltime=", ol['time'])
    print("mabtime=", mab['time'])
    print("admstime=", adms['time'])

    print("#-----------------------Weightsum---------------------")
    print("olsum=", ol['weight'])
    print("mabsum=", mab['weight'])
    print("admssum=", adms['weight'])

    plt.figure(figsize=(12, 9), dpi=80)
    ax1 = plt.subplot(221)
    # plt.plot(iteration, relaxnonlinear['accuracy'], marker="v", color="b",label='RelaxNonLinear')
    # plt.plot(iteration, nonlinear['accuracy'], marker="v", color="m",label='Nonlinear')
    # plt.plot(iteration, ilp['accuracy'], marker="v", color="r",label='ILP')

    plt.plot(iteration, relaxilp['accuracy'], marker="v", color="y",label='RelaxILP')
    plt.plot(iteration, app['accuracy'], marker="v", color="b", label='Appro')
    plt.plot(iteration, nbs['accuracy'], marker="v", color="c", label='NBS')
    plt.plot(iteration, nonshare['accuracy'], marker="v", color="m", label='Nonshare')
    plt.plot(iteration, sc['accuracy'], marker="v", color="r", label='SC')

    plt.plot(iteration, ol['accuracy'], marker="v", color="k", label='OL')
    plt.plot(iteration, mab['accuracy'], marker="v", color="y", label='MAB')
    plt.plot(iteration, adms['accuracy'], marker="v", color="g", label='ADMS')
    plt.xlabel('Number of cloudlets')
    plt.ylabel('Accuracy')
    plt.legend()

    ax2 = plt.subplot(222)
    # plt.plot(iteration, relaxnonlinear['delay'], marker="v", color="b",label='RelaxNonLinear')
    # plt.plot(iteration, nonlinear['delay'], marker="v", color="m",label='Nonlinear')
    # plt.plot(iteration, ilp['delay'], marker="v", color="r",label='ILP')

    plt.plot(iteration, relaxilp['delay'], marker="v", color="y",label='RelaxILP')
    plt.plot(iteration, app['delay'], marker="v", color="b", label='Appro')
    plt.plot(iteration, nbs['delay'], marker="v", color="c", label='NBS')
    plt.plot(iteration, nonshare['delay'], marker="v", color="m", label='Nonshare')
    plt.plot(iteration, sc['delay'], marker="v", color="r", label='SC')

    plt.plot(iteration, ol['delay'], marker="v", color="k", label='OL')
    plt.plot(iteration, mab['delay'], marker="v", color="y", label='MAB')
    plt.plot(iteration, adms['delay'], marker="v", color="g", label='ADMS')
    plt.xlabel('Number of cloudlets')
    plt.ylabel('Delay')
    plt.legend()

    ax3 = plt.subplot(223)
    # plt.plot(iteration, relaxnonlinear['cost'], marker="v", color="b",label='RelaxNonLinear')
    # plt.plot(iteration, nonlinear['cost'], marker="v", color="m",label='Nonlinear')
    # plt.plot(iteration, ilp['cost'], marker="v", color="r",label='ILP')

    plt.plot(iteration, app['cost'], marker="v", color="b", label='Appro')
    plt.plot(iteration, relaxilp['cost'], marker="v", color="y",label='RelaxILP')
    plt.plot(iteration, nbs['cost'], marker="v", color="c", label='NBS')
    plt.plot(iteration, nonshare['cost'], marker="v", color="m", label='Nonshare')
    plt.plot(iteration, sc['cost'], marker="v", color="r", label='SC')

    plt.plot(iteration, ol['cost'], marker="v", color="k", label='OL')
    plt.plot(iteration, mab['cost'], marker="v", color="y", label='MAB')
    plt.plot(iteration, adms['cost'], marker="v", color="g", label='ADMS')
    plt.xlabel('Number of cloudlets')
    plt.ylabel('Cost')
    plt.legend()

    ax4 = plt.subplot(224)
    # plt.plot(iteration, relaxnonlinear['time'], marker="v", color="b",label='RelaxNonLinear')
    # plt.plot(iteration, nonlinear['time'], marker="v", color="m",label='Nonlinear')
    # plt.plot(iteration, ilp['time'], marker="v", color="r",label='ILP')

    plt.plot(iteration, relaxilp['time'], marker="v", color="y", label='RelaxILP')
    plt.plot(iteration, app['time'], marker="v", color="b", label='Appro')
    plt.plot(iteration, nbs['time'], marker="v", color="c", label='NBS')
    plt.plot(iteration, nonshare['time'], marker="v", color="m", label='Nonshare')
    plt.plot(iteration, sc['time'], marker="v", color="r", label='SC')

    plt.plot(iteration, ol['time'], marker="v", color="k", label='OL')
    plt.plot(iteration, mab['time'], marker="v", color="y", label='MAB')
    plt.plot(iteration, adms['time'], marker="v", color="g", label='ADMS')
    plt.xlabel('Number of cloudlets')
    plt.ylabel('Time')
    plt.legend()


    #-------------------weighsum--------------------------------

    plt.figure(figsize=(12, 9), dpi=80)
    ax5 = plt.subplot(221)

    plt.plot(iteration, ol['weight'], marker="v", color="k", label='OL')
    plt.plot(iteration, mab['weight'], marker="v", color="y", label='MAB')
    plt.plot(iteration, adms['weight'], marker="v", color="g", label='ADMS')

    plt.xlabel('Number of cloudlets')
    plt.ylabel('Weightsum')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test1()