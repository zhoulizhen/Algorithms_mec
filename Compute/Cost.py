
def get_pulling_cost(model, request, cloudlets, pulling_delay,trans_delay):
    home_cloudlet_id = request['home_cloudlet_id']
    pulling_cost = cloudlets[home_cloudlet_id]['trans_power'] * (pulling_delay + trans_delay)
    return pulling_cost

def get_inference_cost(model, request, cloudlets, inference_delay):
    home_cloudlet_id = request['home_cloudlet_id']
    inference_cost = inference_delay * (cloudlets[home_cloudlet_id]['accessing_rate'] * request['number_of_instructions'] / inference_delay * cloudlets[home_cloudlet_id]['peak_power'] + cloudlets[home_cloudlet_id]['idle_power'] + cloudlets[home_cloudlet_id]['leak_power'] )
    return inference_cost

def get_cost(model, request, cloudlets, pulling_delay, inference_delay, trans_delay):
    pulling_cost = get_pulling_cost(model, request, cloudlets, pulling_delay, trans_delay)
    inference_cost = get_inference_cost(model, request, cloudlets, inference_delay)
    cost = pulling_cost + inference_cost
    return cost