
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx
import random
import matplotlib.pyplot as plt
from Graph import ParameterSetting as PS
import copy
import pandas as pd
from scipy.spatial.distance import cdist
import math

def shannon_rate(bandwidth, signal_power, noise_power):
    return bandwidth * math.log2(1 + (signal_power / noise_power))

def calculate_signal_power(distance, p_tx=30, path_loss_exponent=2):
    return p_tx / (distance ** path_loss_exponent)

def calculate_transmission_rate(distance, bandwidth=20, noise_power=10, p_tx=30):
    signal_power = calculate_signal_power(distance, p_tx)
    return shannon_rate(bandwidth, signal_power, noise_power)

def fineHomeCloudlet(cls, ues):
    distances = cdist(ues[['Latitude', 'Longitude']], cls[['LATITUDE', 'LONGITUDE']], metric='euclidean')
    nearest_cls_index = np.argmin(distances, axis=1)
    nearest_cls_ids = cls.iloc[nearest_cls_index]['CL_ID'].values
    ues['Home_cloudlet'] = nearest_cls_ids

    transmission_rates = []
    for i, distance in enumerate(distances[np.arange(len(distances)), nearest_cls_index]):
        rate = calculate_transmission_rate(distance)
        transmission_rates.append(rate)
    ues['Transmission_rate'] = transmission_rates
    return ues

def connectNode(locations, ues):
    graph = nx.Graph()
    for l in locations.keys():
        graph.add_node(l)
    cloud = len(locations)

    while not all(nx.has_path(graph, source, target) for source in graph.nodes() for target in graph.nodes() if source != target):
        node1 = random.choice(list(graph.nodes))
        node2 = random.choice(list(graph.nodes))
        if node1 != node2 and not graph.has_edge(node1, node2) and node1 != cloud and node2 != cloud:
            link_delay = PS.link_delay()
            graph.add_edge(node1, node2, weight=link_delay)
        if node1 != node2 and not graph.has_edge(node1, node2) and (node1 == cloud or node2 == cloud):
            cloud_delay = PS.cloud_delay()
            graph.add_edge(node1, node2, weight=cloud_delay)

    connected_nodes = nx.number_connected_components(graph)
    connected = nx.is_connected(graph)

    if connected:
        print("There are {} connected nodes".format(connected_nodes))
    else:
        print("The graph is not connected")

    return graph

def generateRequests(cls,ues,num, services,accuracy_constraint,cost_constraint,cloudlets):
    ues_count = len(ues)
    cl_count = len(cls)

    requests = {}
    count = 0
    for i in range(1, ues_count+1):
        user = ues.iloc[0]
        for n in range(1, num+1):
            count += 1
            user_id = user['User_ID']
            home_cloudlet_id = random.choice(range(1, cl_count + 1))
            info = {
                'id': count,
                'user_id': user_id,
                'home_cloudlet_id': home_cloudlet_id,
                'number_of_instructions': PS.number_of_instructions(),
                'cost_budget': PS.cost_budget(cost_constraint),
                'service_type': PS.service_type(services),
                'accuracy_requirement': PS.accuracy_requirement(accuracy_constraint),
                'input_size': PS.input_data_size(),
                'Transmission_rate': user['Transmission_rate']
            }
            requests[count] = info
            if home_cloudlet_id in cloudlets.keys():
                cloudlets[home_cloudlet_id]['home_requests'].append(count)
    return requests

def generateCloudlets(cls, ues,computing_constraint):
    ues_count = len(ues)
    cl_count = len(cls)

    cloudlets = {}
    locations = {}
    for i in range(1, cl_count+1):
        cls_ids = cls['CL_ID'].unique()
        info = {
            'id': cls_ids[i-1],
            'computing_capacity': PS.computing_capacity(computing_constraint),
            'processing_rate': PS.processing_rate(),
            'accessing_rate': PS.accessing_rate(),
            'peak_power': PS.peak_power(),
            'idle_power': PS.idle_power(),
            'leak_power': PS.leak_power(),
            'trans_power': PS.trans_power(),
            'home_requests': [],
            'service_rate': PS.service_rate(),
            'arrival_rate': PS.arrival_rate()
        }
        cloudlets[i] = info
    locations = copy.copy(cloudlets)
    locations[cl_count+1] = {
            'id': cl_count+1,
            'computing_capacity': PS.computing_capacity(computing_constraint),
            'processing_rate': PS.processing_rate(),
            'accessing_rate': PS.accessing_rate(),
            'peak_power': PS.peak_power(),
            'idle_power': PS.idle_power(),
            'leak_power': PS.leak_power(),
            'trans_power': PS.trans_power(),
            'home_requests': [],
            'service_times': []
        }
    # print("Thre are {} locations".format(len(locations)))
    return cloudlets,locations

def generateModels(num_models, services):

    models = {}
    for i in range(1, num_models+1):
        info = {
            'id': i,
            'size': PS.model_size(),
            'float_operations': PS.float_operations(),
            'service_type': PS.service_type(services)
        }
        info['shareable'] = PS.shareable_subset(info['size'])
        info['remain'] = PS.remaining_subset(info['size'], info['shareable'])
        models[i] = info
    # print("There are {} models.".format(len(models)))
    return models