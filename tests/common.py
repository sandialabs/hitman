import numpy as np


def generate_random_vectors():
    # Generate two random 3-vectors
    theta = np.random.normal(size=3)
    wt = np.random.uniform(0, np.pi)
    theta_dot = np.random.normal(size=3)

    # Scale initial rotation
    theta = theta / np.linalg.norm(theta) * wt
    return (theta, theta_dot)


def print_comparison(val, ref):
    print(val)
    print(ref)
    print(f"atol:  {np.max(np.abs(ref-val))}")
    print(f"rtol:  {np.max(np.abs(ref-val)/np.abs(ref))}")
    print(f"mutol: {np.mean(np.abs(ref-val))}")
    return
