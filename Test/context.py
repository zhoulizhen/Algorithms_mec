
import numpy as np

def generate_context(num_requests: int, num_features: int) -> list:
    """
    Generate a series of contexts for multiple requests,
    where each context is a scaled version of a base context.
    """
    # Generate a base context vector with random values scaled by 100
    base_context = np.random.rand(num_features) * 100

    # Create contexts for subsequent requests by scaling the base context
    contexts = [base_context * (i + 1) for i in range(num_requests)]

    return contexts
