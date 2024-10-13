import random
from itertools import product
from math import sqrt

import gurobipy as gp
from gurobipy import GRB
from Compute import Delay, Cost
from math import log
import time

def ilp(num_requests, num_models, num_locations, requests, models, cloudlets, locations, Graph, accuracy_dict, xi, alpha):
    start = time.time()
    # construct a list consisting locations besides the home cloudlet of j
    # Todo: construct the lcoations for specific subset
    Loc_j = {}
    # m_j = {}
    for j in range(1, num_requests+1):
        # m_j[j] = []
        Loc_j[j] = [i for i in range(1, num_locations+1)]
        Loc_j[j].remove(requests[j]['home_cloudlet_id'])
        # for m in range(1, num_models+1):
        #     if models[m]['service_type'] == requests[j]['service_type']:
        #         m_j[j].append(m)
        # if not len(m_j[j]):
        #     m_j[j] = random.sample(range(1, num_models+1), 2)

    # print("Loc_j", Loc_j)
    # print("m_j", m_j)


    x_jk = [(j,k) for j in range(1, num_requests+1) for k in range(1, num_models + 1)]
    # print("x_jk", x_jk)
    # y_jkl = list(product(range(1, num_requests + 1), range(1, num_models + 1), range(1, num_locations + 1)))
    y_jkl = [(j,k,l) for j in range(1, num_requests+1) for k in range(1, num_models + 1) for l in Loc_j[j]]
    # print("y_jkl",y_jkl)

    'accuracy'
    a_jk = {(j, k): accuracy_dict[j][k] for j, k in x_jk}
    # for j,k in x_jk:
    #     print(f"accuracy of request {j} and model {k} is {a_jk[j,k]}", requests[j]['accuracy_requirement'])

    # a_jk = {(j, k): -log(accuracy_dict[j][k]) for j, k in x_jk}
    # # print("accuracy", a_jk)

    # inference delay
    d_jk_inf = {(j, k): Delay.get_inference_delay(models[k], requests[j], cloudlets) for j, k in x_jk}
    # print("inference delay", d_jk_inf)

    # inference cost
    c_jk_inf = {(j, k): Cost.get_inference_cost(models[k], requests[j], cloudlets, Delay.get_inference_delay(models[k], requests[j], cloudlets)) for j, k in x_jk}
    # print("inference cost", c_jk_inf)

    c_jk_inf = {(j, k): Cost.get_inference_cost(models[k], requests[j], cloudlets,
                                                Delay.get_inference_delay(models[k], requests[j], cloudlets)) for j, k in x_jk}
    # print("inference cost", c_jk_inf)
    # get_cost(model, request, cloudlets, pulling_delay, inference_delay)


    # pulling delay
    d_jkl_sh = {(j, k, l): Delay.get_pulling_delay_sh(models[k], requests[j], locations[l], Graph) for j, k, l in y_jkl}
    d_jkl_re = {(j, k, l): Delay.get_pulling_delay_re(models[k], requests[j], locations[l], Graph) for j, k, l in y_jkl}
    # print("pulling delay_sh", d_jkl_sh)
    # print("pulling delay_re", d_jkl_sh)

    # Create a new model
    m = gp.Model("ILP")

    # Decision variables
    x = m.addVars(((j,k) for j in range(1, num_requests + 1) for k in range(1, num_models + 1)), vtype=GRB.BINARY, name="ModelSelection")

    y_sh = m.addVars(((j, k, l) for j in range(1, num_requests + 1) for k in range(1, num_models + 1) for l in Loc_j[j]), vtype=GRB.BINARY, name="PullShareable")

    y_re = m.addVars(((j, k, l) for j in range(1, num_requests + 1) for k in range(1, num_models + 1) for l in Loc_j[j]),
                     vtype=GRB.BINARY, name="PullRemaining")

    z = m.addVars(((j, k, l) for j in range(1, num_requests + 1) for k in range(1, num_models + 1) for l in Loc_j[j]),
                     vtype=GRB.BINARY, name="AuxiliaryZ")

    q = m.addVars(((j, k, l) for j in range(1, num_requests + 1) for k in range(1, num_models + 1) for l in Loc_j[j]),
                  vtype=GRB.BINARY, name="AuxiliaryQ")

    # y_sh = m.addVars(range(1, num_requests + 1), m_j[j], Loc_j[j], vtype=GRB.BINARY, name="PullShareable")
    # y_re = m.addVars(range(1, num_requests + 1), range(1, num_models + 1), range(1, num_locations + 1),  vtype=GRB.BINARY, name="PullRemaining")
    # z = m.addVars(range(1, num_requests + 1), range(1, num_models + 1), range(1, num_locations + 1), vtype=GRB.BINARY, name="AuxiliaryZ")
    # q = m.addVars(range(1, num_requests + 1), range(1, num_models + 1), range(1, num_locations + 1), vtype=GRB.BINARY, name="AuxiliaryQ")

    max_d = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="max_d")
    m.addConstr(max_d >= z.prod(d_jkl_sh), name="max_d_sh")
    m.addConstr(max_d >= q.prod(d_jkl_re), name="max_d_re")
    # print("max_d is ", max_d)

    'each request select one model'
    m.addConstrs((gp.quicksum(x[(j, k)] for k in range(1, num_models + 1)) == 1 for j in range(1, num_requests + 1)), name='model_selection')

    'each request pull the selected model from one location' # no need
    # m.addConstrs((gp.quicksum(y_sh[(j, k, l)] for l in range(1, num_locations + 1)) == 1 for j in range(1, num_requests + 1) for k in range (1, num_models+1)),  name='model_pull_sh')
    #
    # m.addConstrs((gp.quicksum(y_re[(j, k, l)] for l in range(1, num_locations + 1)) == 1 for j in range(1, num_requests + 1) for k in range(1, num_models + 1)), name='model_pull_re')

    'only the model is selected, it can pull the model from one location'
    m.addConstrs((gp.quicksum(y_sh[(j, k, l)] for l in Loc_j[j]) == x[(j, k)] for j in range(1, num_requests + 1) for k in range(1, num_models + 1)), name='model_pull_sh')

    m.addConstrs((gp.quicksum(y_re[(j, k, l)] for l in Loc_j[j]) == x[(j, k)] for j in
                  range(1, num_requests + 1) for k in range(1, num_models + 1)), name='model_pull_re')

    # 'each request should pull two subsets simultaneously' # no need
    # m.addConstrs((gp.quicksum(y_sh[(j, k, l)] for l in range(1, num_locations + 1)) + gp.quicksum(y_re[(j, k, l)] for l in range(1, num_locations + 1)) == 2 * x[(j, k)] for j in range(1, num_requests + 1) for k in range(1, num_models + 1)), name='pull_both')

    'z >= x + y - 1'
    m.addConstrs((z[(j,k,l)] >= x[(j,k)] + y_sh[(j,k,l)]- 1 for j in range(1, num_requests+1) for k in range(1, num_models + 1) for l in Loc_j[j]), name='variable_z')

    m.addConstrs((q[(j,k,l)] >= x[(j,k)] + y_re[(j,k,l)]- 1 for j in range(1, num_requests+1) for k in range(1, num_models + 1) for l in Loc_j[j]), name='variable_q')

    m.addConstrs(
        (z[(j, k, l)] >= y_sh[(j, k, l)] for j in range(1, num_requests + 1) for k in range(1, num_models + 1) for l in
         Loc_j[j]), name='variable_z')

    m.addConstrs(
        (q[(j, k, l)] >= y_re[(j, k, l)] for j in range(1, num_requests + 1) for k in range(1, num_models + 1) for l in
         Loc_j[j]), name='variable_q')

    # 'q <= 0.5 (x + y)'
    # m.addConstrs((q[(j, k, l)] == 0.5 * (x[(j, k)] + y_re[(j, k, l)]) for j in range(1, num_requests + 1) for k in
    #               range(1, num_models + 1) for l in range(1, num_locations + 1)), name='variable_q')

    # 'each request should pull two subsets simultaneously'
    # m.addConstrs((gp.quicksum(z[(j, k, l)] for l in range(1, num_locations + 1)) + gp.quicksum(q[(j, k, l)] for l in range(1, num_locations + 1)) == 2 for j in range(1, num_requests + 1) for k in range(1, num_models + 1)), name='pull_both')
    # Todo: modify the potential lcoations

    'cost budget'
    m.addConstrs((gp.quicksum(x[(j, k)] * c_jk_inf[(j, k)] for k in range(1, num_models + 1))
                  + cloudlets[requests[j]['home_cloudlet_id']]['trans_power'] * max_d <= requests[j]['cost_budget'] for j in range(1, num_requests+1)), name='budget')


    'capacity constraint'
    m.addConstrs((gp.quicksum(x[(j, k)] * alpha * (models[k]['shareable'] + models[k]['remain']) for k in
                  range(1, num_models + 1)) <= cloudlets[requests[j]['home_cloudlet_id']]['computing_capacity'] for j in range(1, num_requests + 1)), name='capacity')

    'accuracy constraint'
    m.addConstrs((gp.quicksum(x[(j, k)] * a_jk[(j, k)] for k in range(1, num_models + 1)) >= requests[j]['accuracy_requirement'] for j in range(1, num_requests + 1)), name='accuracy')

    # 'binary variables'
    # m.addConstrs((x[(j, k)] >= 0 for j in range(1, num_requests + 1) for k in range(1, num_models + 1)),name='Constraint_x1')
    # m.addConstrs((x[(j, k)] <= 1 for j in range(1, num_requests + 1) for k in range(1, num_models + 1)),
    #     name='Constraint_x2')
    # m.addConstrs((y_sh[(j, k, l)] >= 0 for j in range(1, num_requests + 1) for k in
    #               range(1, num_models + 1) for l in range(1, num_locations + 1)), name='Constraint_y_sh1')
    # m.addConstrs((y_sh[(j, k, l)] <= 1 for j in range(1, num_requests + 1) for k in
    #               range(1, num_models + 1) for l in range(1, num_locations + 1)), name='Constraint_y_sh2')
    # m.addConstrs((y_re[(j, k, l)] >= 0 for j in range(1, num_requests + 1) for k in
    #               range(1, num_models + 1) for l in range(1, num_locations + 1)), name='Constraint_y_re1')
    # m.addConstrs((y_re[(j, k, l)] <= 1 for j in range(1, num_requests + 1) for k in
    #               range(1, num_models + 1) for l in range(1, num_locations + 1)), name='Constraint_y_re2')
    # m.addConstrs((z[(j, k, l)] >= 0 for j in range(1, num_requests + 1) for k in
    #               range(1, num_models + 1) for l in range(1, num_locations + 1)), name='Constraint_z1')
    # m.addConstrs((z[(j, k, l)] <= 1 for j in range(1, num_requests + 1) for k in
    #               range(1, num_models + 1) for l in range(1, num_locations + 1)), name='Constraint_z2')
    # m.addConstrs((q[(j, k, l)] >= 0 for j in range(1, num_requests + 1) for k in
    #               range(1, num_models + 1) for l in range(1, num_locations + 1)), name='Constraint_q1')
    # m.addConstrs((q[(j, k, l)] <= 1 for j in range(1, num_requests + 1) for k in
    #               range(1, num_models + 1) for l in range(1, num_locations + 1)), name='Constraint_q2')

    m.setObjective(max_d + x.prod(d_jk_inf), GRB.MINIMIZE)

    m.optimize()

    # print("status", m.status)

    # for j, k in x.keys():
    #     # if (abs(x[j, k].x) > 1e-6):
    #     print(f"\n variable x: request {j} select model {k} {round(100*x[j, k].X, 2)} %")
    #
    # for j, k, l in y_sh.keys():
    #     # if (abs(x[j, k].x) > 1e-6):
    #     print(f"\n variable y_sh: request {j} select model {k} from location {l} {round(100 * y_sh[j, k, l].X, 2)} %")
    #
    # for j, k, l in y_re.keys():
    #     # if (abs(x[j, k].x) > 1e-6):
    #     print(f"\n variable y_re: request {j} select model {k} from location {l} {round(100 * y_re[j, k, l].X, 2)} %")
    #
    # for j, k, l in z.keys():
    #     # if (abs(x[j, k].x) > 1e-6):
    #     print(f"\n variable z: request {j} select model {k} from location {l} {round(100 * z[j, k, l].X, 2)} %")
    #
    # for j, k, l in q.keys():
    #     # if (abs(x[j, k].x) > 1e-6):
    #     print(f"\n variable q: request {j} select model {k} from location {l} {round(100 * q[j, k, l].X, 2)} %")


    # print("max_d", max_d)
    #
    accuracy = 0
    delay = 0
    cost = 0
    infdelay = 0

    # print("iteration ILP......................")

    for j in range(1, num_requests + 1):
        for k in range(1, num_models + 1):
            # print(f"\n x: request {j} select {k} {x[j, k].X}")
            inf_decision = x[j, k].X
            if inf_decision:
                # print("computing inf.................")
                infdelay = d_jk_inf[(j, k)] * inf_decision
                # print("inference delay", infdelay)
                accuracy += a_jk[(j, k)] * inf_decision
                # print("accuracy", accuracy)
                pullshdelay = 0
                pullredelay = 0

                for l in Loc_j[j]:
                    pullsh_decision = y_sh[j, k, l].X
                    pullre_decision = y_re[j, k, l].X
                    # print(f"\n y_sh: request {j} select {k} from {l} {y_sh[j, k, l].X}")
                    # print(f"\n y_re: request {j} select {k} from {l} {y_re[j, k, l].X}")
                    if pullsh_decision:
                        # print("computing pullsh.................")
                        pullshdelay += d_jkl_sh[(j, k, l)] * pullsh_decision

                    if pullre_decision:
                        # print("computing pullre.................")
                        pullredelay += d_jkl_re[(j, k, l)] * pullre_decision

                pulldelay = max(pullshdelay, pullredelay)
                # print("pull delay", pulldelay)

                delay += pulldelay + infdelay

                cost += Cost.get_cost(models[k], requests[j], cloudlets, pulldelay, infdelay)
    # print("cost--ILP", cost)
    # print("delay--ILP", delay)
    # print("Objective value: ", m.ObjVal)
    end = time.time()
    sumtime = end-start
    return accuracy / num_requests, delay, cost / num_requests,sumtime

    # print("-----------------------ILP---------------------")
    # print("accuracy {}, delay {}, cost {}".format(accuracy / num_requests, delay, cost / num_requests))




