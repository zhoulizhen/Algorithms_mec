import random
import numpy as np

def get_pulling_delay_sh(model, request, location_sh, Graph):
    length_sh = shortest_path(request['home_cloudlet_id'], location_sh['id'], Graph)
    pulling_delay = model['shareable'] * length_sh
    return pulling_delay

def get_pulling_delay_re(model, request, location_re, Graph):
    length_re = shortest_path(request['home_cloudlet_id'], location_re['id'], Graph)
    pulling_delay = model['remain'] * length_re
    return pulling_delay

def get_pull_cloud_sh(model, request, location,Graph):
    length_cloud = shortest_path(request['home_cloudlet_id'], location, Graph)
    pulling_delay = model['shareable'] * length_cloud
    return pulling_delay

def get_pull_cloud_re(model, request, location, Graph):
    length_cloud = shortest_path(request['home_cloudlet_id'], location, Graph)
    pulling_delay = model['remain'] * length_cloud
    return pulling_delay

def get_trans_delay(model, request):
    rate = request['Transmission_rate']
    input_size = request['input_size']
    out_size = input_size * model['size'] * random.uniform(0.1, 0.3)
    trans_delay = input_size/rate + out_size/rate
    return trans_delay

import networkx as nx
def shortest_path(source_node, target_node, Graph):
    length = None
    if nx.has_path(Graph, source_node, target_node):
        length = nx.shortest_path_length(Graph, source=source_node, target=target_node, weight='weight')
    else:
        print(source_node,target_node)
        print("There are no edges")
    return length

def get_inference_delay(model, request, cloudlets):
    home_cloudlet_id = request['home_cloudlet_id']
    inference_delay = model['float_operations'] * model['size'] / cloudlets[home_cloudlet_id]['processing_rate']
    return inference_delay

def get_queue_delay(request, cloudlets):
    home_cloudlet_id = request['home_cloudlet_id']
    num_request = len(cloudlets[home_cloudlet_id]['home_requests'])
    if num_request == 0:
        return 0
    service_rate = cloudlets[home_cloudlet_id]['service_rate']
    arrival_rate = cloudlets[home_cloudlet_id]['arrival_rate']
    service_times = 1/ service_rate
    if service_times:
        std_deviation = np.std(service_times)
    else:
        std_deviation = 0.1
    if arrival_rate >= service_rate:
        return float('inf')
    queue_delay = (arrival_rate * (std_deviation ** 2 + (1/service_rate) ** 2)) / (
                2 * (1 - arrival_rate / service_rate))
    return queue_delay
