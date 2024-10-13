#----------request---------------#
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

def input_data_size(): return random.uniform(0.09,0.5) # GB #todo: reference

#----------cloudlet---------------#
import random

def computing_capacity(computing_constraint):
    if computing_constraint:
        return computing_constraint
    else:
        return random.randint(8, 24)  # GB

def peak_power(): return random.uniform(0.8,2.5) # Watt

def idle_power(): return random.uniform(0.1,0.3) # Watt

def leak_power(): return random.uniform(0.1,0.3) # Watt

def trans_power(): return random.uniform(0.8,2.5) # Watt

def service_rate(): return random.uniform(0.5,0.9) # GB

def arrival_rate(): return random.uniform(0.1,0.5) # GB

#---------------models-------------------#


def shareable_subset(size): return size*2/3 # Set the shareble subset takes 2/3 of model size

def remaining_subset(size, shareble_size): return size-shareble_size

def model_size(): return random.uniform(5, 20) # GB
def float_operations(): return random.uniform(100, 300) # TFLOPS
def accessing_rate(): return random.uniform(0.1,0.5)

def processing_rate(): return random.uniform(107,312) # TFLOPS

# inference delay 计算出来的是1Gb --> second，
#---------------edge-------------------#
def link_delay(): return random.uniform(5,10) # s
#todo: actually, this has to be ms, but really situation is more larger than this, so we just use s

# todo: the real value should be 30-50ms, but we just use 5-10s to balance the inference delay and pulling delay

def cloud_delay(): return random.uniform(10,20) # s

# todo: second \ TFLOPS \GB \Watt