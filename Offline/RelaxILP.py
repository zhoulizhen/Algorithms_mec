
import gurobipy as gp
from gurobipy import GRB
from Compute import Delay, Cost
from math import log
import time

def relaxILP(num_requests, num_models, num_locations, requests, models, cloudlets, locations, Graph, accuracy_dict, xi, alpha,lsh,lre):
    start = time.time()

    x_jk = [(j,k) for j in range(1, num_requests+1) for k in range(1, num_models + 1)]
    y_jkl_sh = [(j,k,l) for j in range(1, num_requests+1) for k in range(1, num_models + 1) for l in lsh[k]]
    y_jkl_re = [(j, k, l) for j in range(1, num_requests + 1) for k in range(1, num_models + 1) for l in lre[k]]

    'accuracy'
    a_jk = {(j, k): accuracy_dict[j][k] for j, k in x_jk}

    'inference delay'
    d_jk_inf = {(j, k): Delay.get_inference_delay(models[k], requests[j], cloudlets) for j, k in x_jk}

    'queue delay'

    d_jk_queue = {(j, k): Delay.get_queue_delay(requests[j], cloudlets) for j, k in x_jk}

    'inference cost'
    c_jk_inf = {(j, k): Cost.get_inference_cost(models[k], requests[j], cloudlets, Delay.get_inference_delay(models[k], requests[j], cloudlets)) for j, k in x_jk}

    'pulling delay'
    d_jkl_sh = {(j, k, l): Delay.get_pulling_delay_sh(models[k], requests[j], locations[l], Graph) for j, k, l in y_jkl_sh}
    d_jkl_re = {(j, k, l): Delay.get_pulling_delay_re(models[k], requests[j], locations[l], Graph) for j, k, l in y_jkl_re}

    'transmission delay'
    d_jk_trans = {(j, k): Delay.get_trans_delay(models[k], requests[j]) for j, k in x_jk}

    'Create a new model'
    m = gp.Model("ILP")

    'Decision variables'
    x = m.addVars(((j,k) for j in range(1, num_requests + 1) for k in range(1, num_models + 1)), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="ModelSelection")

    y_sh = m.addVars(((j, k, l) for j in range(1, num_requests + 1) for k in range(1, num_models + 1) for l in lsh[k]), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="PullShareable")

    y_re = m.addVars(((j, k, l) for j in range(1, num_requests + 1) for k in range(1, num_models + 1) for l in lre[k]), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="PullRemaining")

    z = m.addVars(((j, k, l) for j in range(1, num_requests + 1) for k in range(1, num_models + 1) for l in lsh[k]), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="AuxiliaryZ")

    q = m.addVars(((j, k, l) for j in range(1, num_requests + 1) for k in range(1, num_models + 1) for l in lre[k]), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="AuxiliaryQ")

    max_d = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="max_d")
    m.addConstr(max_d >= z.prod(d_jkl_sh), name="max_d_sh")
    m.addConstr(max_d >= q.prod(d_jkl_re), name="max_d_re")

    'each request need to select one model'
    m.addConstrs((gp.quicksum(x[(j, k)] for k in range(1, num_models + 1)) == 1 for j in range(1, num_requests + 1)), name='model_selection')

    'once the model is selected, it must be pulled from one location'
    m.addConstrs((gp.quicksum(y_sh[(j, k, l)] for l in lsh[k]) == x[(j, k)] for j in range(1, num_requests + 1) for k in range(1, num_models + 1)), name='model_pull_sh')

    m.addConstrs((gp.quicksum(y_re[(j, k, l)] for l in lre[k]) == x[(j, k)] for j in range(1, num_requests + 1) for k in range(1, num_models + 1)), name='model_pull_re')

    'z >= (x + y)-1'
    m.addConstrs((z[(j,k,l)] >= x[(j,k)] + y_sh[(j,k,l)]- 1 for j in range(1, num_requests+1) for k in range(1, num_models + 1) for l in lsh[k]), name='variable_z')

    m.addConstrs((q[(j,k,l)] >= x[(j,k)] + y_re[(j,k,l)]- 1 for j in range(1, num_requests+1) for k in range(1, num_models + 1) for l in lre[k]), name='variable_q')

    'z >= y'
    m.addConstrs((z[(j, k, l)] >= y_sh[(j, k, l)] for j in range(1, num_requests + 1) for k in range(1, num_models + 1) for l in lsh[k]), name='variable_z')

    m.addConstrs((q[(j, k, l)] >= y_re[(j, k, l)] for j in range(1, num_requests + 1) for k in range(1, num_models + 1) for l in lre[k]), name='variable_q')

    'cost budget'
    m.addConstrs((gp.quicksum(x[(j, k)] * c_jk_inf[(j, k)] for k in range(1, num_models + 1)) + cloudlets[requests[j]['home_cloudlet_id']]['trans_power'] * (max_d + gp.quicksum(x[(j,k)] * d_jk_trans[(j,k)] for k in range(1, num_models + 1))) <= requests[j]['cost_budget'] for j in range(1, num_requests+1)), name='budget')

    'capacity constraint'
    m.addConstrs((gp.quicksum(x[(j, k)] * alpha * (models[k]['shareable'] + models[k]['remain']) for k in range(1, num_models + 1)) <= cloudlets[requests[j]['home_cloudlet_id']]['computing_capacity'] for j in range(1, num_requests + 1)), name='capacity')

    'accuracy constraint'
    m.addConstrs((gp.quicksum(x[(j, k)] * a_jk[(j, k)] for k in range(1, num_models + 1)) >= requests[j]['accuracy_requirement'] for j in range(1, num_requests + 1)), name='accuracy')

    m.setObjective(max_d + x.prod(d_jk_inf) + x.prod(d_jk_queue) + x.prod(d_jk_trans), GRB.MINIMIZE)

    m.optimize()

    print("Objective value--RelaxILP: ", m.ObjVal)

    accuracy = 0
    delay = 0
    cost = 0
    for j in range(1, num_requests + 1):
        for k in range(1, num_models + 1):
            inf_decision = x[j, k].X
            if inf_decision:
                infdelay = d_jk_inf[(j, k)] * inf_decision
                queue_delay = d_jk_queue[(j, k)] * inf_decision
                trans_delay = d_jk_trans[(j, k)] * inf_decision
                accuracy += a_jk[(j, k)] * inf_decision
                pullshdelay = 0
                pullredelay = 0

                for l in lsh[k]:
                    pullsh_decision = y_sh[j,k,l].X
                    if pullsh_decision:
                        pullshdelay += d_jkl_sh[(j, k, l)] * pullsh_decision

                for l in lre[k]:
                    pullre_decision = y_re[j, k, l].X
                    if pullre_decision:
                        pullredelay += d_jkl_re[(j, k, l)] * pullre_decision
                pulldelay = max(pullshdelay, pullredelay)
                delay += pulldelay + infdelay + queue_delay + trans_delay
                cost += Cost.get_cost(models[k], requests[j], cloudlets, pulldelay, infdelay, trans_delay)

    end = time.time()
    sumtime = end - start

    x_values = {(j, k): x[j, k].X for j in range(1, num_requests + 1) for k in range(1, num_models + 1)}
    y_sh_values = {(j, k, l): y_sh[j, k, l].X for j in range(1, num_requests + 1) for k in range(1, num_models + 1) for l in lsh[k]}
    y_re_values = {(j, k, l): y_re[j, k, l].X for j in range(1, num_requests + 1) for k in range(1, num_models + 1) for l in lre[k]}

    return accuracy / num_requests, delay, cost / num_requests, sumtime, x_values, y_sh_values, y_re_values

