import numpy as np
import scipy as sp

from tqdm import tqdm


def get_interval(point, dist, u_prime, w):

    r = np.random.uniform(0, 1)
    xl = point - r * w
    xr = point + (1 - r) * w

    while dist.pdf(xl) > u_prime:
        xl = xl - w

    while dist.pdf(xr) > u_prime:
        xr = xr + w

    return xl, xr


def modify_interval(point, proposed_point, xl, xr):

    if proposed_point > point:
        xr = proposed_point
    else:
        xl = proposed_point

    return xl, xr


class SliceSampler(object):

    def __init__(self, dims=1, w=0.1):
        self.dims = 1
        self.w = w

    def sample(self, dist, n_samples=1000, max_iters=1000, init=[]):
        
        if len(init) == 0:
            point = np.zeros(self.dims) 
        dim = self.dims
        samples = np.zeros((n_samples, dim))
        p_star = dist.pdf(point)
        
        sample_counter = 0
        for sample_counter in tqdm(range(n_samples)):
            u_prime = np.random.uniform(0, p_star)
            xl, xr = get_interval(point, dist, u_prime, self.w)
            n_iters = 0
            while n_iters < max_iters:
                proposed_point = np.random.uniform(xl, xr)
                p_star = dist.pdf(proposed_point)
                if p_star > u_prime:
                    samples[sample_counter] = proposed_point
                    point = proposed_point
                    break
                else:
                    xl, xr = modify_interval(point, proposed_point, xl, xr)
                    n_iters += 1

            if n_iters == max_iters:
                raise RuntimeError(("maximum iterations for modify interval"
                                    "reached"))
        
        return samples



