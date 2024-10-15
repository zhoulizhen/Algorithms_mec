import random
from itertools import product
from math import sqrt, log
import gurobipy as gp
from gurobipy import GRB
from Compute import Delay, Cost
import numpy as np
import time


def relaxILP(num_requests, num_models, requests, models, cloudlets, accuracy_dict, alpha):
    start = time.time()

    # Initialize parameters
    accuracy_matrix = np.array(
        [[accuracy_dict[j][k] for k in range(1, num_models + 1)] for j in range(1, num_requests + 1)])
    costs = np.array([Cost.get_cost(models[k], requests[j], cloudlets) for k in range(1, num_models + 1) for j in
                      range(1, num_requests + 1)])

    # Create Gurobi model
    m = gp.Model("NBS")

    # Decision variables
    x = m.addVars(num_requests, num_models, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="ModelSelection")
    y = m.addVars(num_models, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="ResourceProvisioning")

    # Objective function
    def objective():
        app_accuracy = gp.quicksum(
            gp.quicksum(x[j, k] * accuracy_matrix[j - 1, k - 1] for k in range(1, num_models + 1)) for j in
            range(1, num_requests + 1))
        total_cost = gp.quicksum(y[k] * costs[k - 1] for k in range(1, num_models + 1))
        return gp.quicksum(log(app_accuracy[j] - requests[j]['min_accuracy'] * requests[j]['num_requests']) for j in range(num_requests)) + log(C - total_cost)

    # Constraints
    m.addConstrs((gp.quicksum(x[j, k] for k in range(1, num_models + 1)) == requests[j]['num_requests'] for j in range(1, num_requests + 1)), name='model_selection')
    m.addConstrs((gp.quicksum(y[k] * costs[k - 1] for k in range(1, num_models + 1)) <= C), name='cost_budget')

    # Set objective
    m.setObjective(objective(), GRB.MAXIMIZE)

    # Optimize
    m.optimize()

    # Extract the solution
    x_solution = np.array([[x[j, k].X for k in range(1, num_models + 1)] for j in range(1, num_requests + 1)])
    y_solution = np.array([y[k].X for k in range(1, num_models + 1)])

    end = time.time()


# Example usage
num_requests = 10  # Number of applications
num_models = 5  # Number of models
requests = [{'num_requests': random.randint(10, 100), 'min_accuracy': random.random() * 0.8} for _ in
            range(num_requests)]
models = [{} for _ in range(num_models)]  # Fill in model details as needed
cloudlets = [{}]  # Fill in cloudlet details as needed
accuracy_dict = np.random.rand(num_requests, num_models).tolist()  # Random accuracy data
alpha = 1.0  # Example alpha value

relaxILP(num_requests, num_models, requests, models, cloudlets, accuracy_dict, alpha)
