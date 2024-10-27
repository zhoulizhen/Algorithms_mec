
import random

def number_of_instructions(): return random.randint(50,100)

def cost_budget(cost_constraint):
    if cost_constraint:
        return cost_constraint
    else:
        return random.uniform(30000,50000)

def service_type(services): return random.choice(services)

def accuracy_requirement(accuracy_constraint):
    if accuracy_constraint:
        return accuracy_constraint
    else:
        return random.uniform(0.1,0.4)

def input_data_size(): return random.uniform(0.09,0.5)

def computing_capacity(computing_constraint):
    if computing_constraint:
        return computing_constraint
    else:
        return random.randint(8, 24)

def peak_power(): return random.uniform(0.8,2.5)

def idle_power(): return random.uniform(0.1,0.3)

def leak_power(): return random.uniform(0.1,0.3)

def trans_power(): return random.uniform(0.8,2.5)

def service_rate(): return random.uniform(0.5,0.9)

def arrival_rate(): return random.uniform(0.1,0.5)

def shareable_subset(size): return size*2/3

def remaining_subset(size, shareble_size): return size-shareble_size

def model_size(): return random.uniform(5, 20)

def float_operations(): return random.uniform(100, 300)

def accessing_rate(): return random.uniform(0.1,0.5)

def processing_rate(): return random.uniform(107,312)

def link_delay(): return random.uniform(5,10)

def cloud_delay(): return random.uniform(10,20)