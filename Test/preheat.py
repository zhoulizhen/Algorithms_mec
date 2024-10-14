import random
def initialize_model_locations(num_models, num_cloudlets, clsnum, models):
    """
    Initializes location dictionaries for each model, including parameter sharing and non-sharing.
    """
    lsh = {}
    lre = {}
    lshnon = {}

    for k in range(1, num_models + 1):
        num_po_locations = random.randint(1, clsnum // 10)  # Number of locations for model k
        lk = random.sample(range(1, num_cloudlets + 1), num_po_locations)
        lsh[k] = list(lk)  # Shared locations
        lre[k] = list(lk)  # Remaining locations
        lshnon[k] = list(lk)  # Non-shared locations

    # Update shared locations based on service types
    for k1 in range(1, num_models + 1):
        lsh_copy = set(lsh[k1])  # Use set to avoid duplicates
        for k2 in range(1, num_models + 1):
            if models[k1]['service_type'] == models[k2]['service_type']:
                lsh_copy.update(lsh[k2])
        lsh[k1] = list(lsh_copy)
    return {'lsh': lsh, 'lre': lre, 'lshnon': lshnon}