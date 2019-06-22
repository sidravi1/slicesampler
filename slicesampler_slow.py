import numpy as np
import scipy.stats as st
from tqdm import tqdm


def get_interval(point, dist, u_prime, w):

    r = np.random.uniform(0, 1)
    xl = point - r * w
    xr = point + (1 - r) * w

    while dist.pdf(xl) > u_prime:
        xl = xl - w
    while dist.pdf(xr) > u_prime:
        xr = xr + w

    xs = [(xl, xr)]

    return xs


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
        all_proposed_points = []
        u_primes = []
        all_xs = []
        sample_counter = 0
        for sample_counter in tqdm(range(n_samples)):
            proposed_points = []
            u_prime = np.random.uniform(0, p_star)
            u_primes.append(u_prime)
            xs = get_interval(point, dist, u_prime, self.w)
            xl, xr = xs[-1]
            n_iters = 0
            while n_iters < max_iters:
                proposed_point = np.random.uniform(xl, xr)
                proposed_points.append(proposed_point.squeeze())
                p_star = dist.pdf(proposed_point)
                if p_star > u_prime:
                    samples[sample_counter] = proposed_point
                    point = proposed_point
                    all_proposed_points.append(np.hstack(proposed_points))
                    break
                else:
                    xl, xr = modify_interval(point, proposed_point, xl, xr)
                    xs.append((xl, xr)) 
                    n_iters += 1
            all_xs.append(np.array(xs).reshape(-1, 2))
            if n_iters == max_iters:
                raise RuntimeError(("maximum iterations for modify interval"
                                    "reached"))
        return {'samples': samples, 'xs': all_xs, 
                'proposed_points': np.array(all_proposed_points),
                'uprimes': np.hstack(u_primes)
               }

class MixtureGaussians(object):

    def __init__(self, 
                 mu = np.linspace(-2, 2, 5), 
                 sd = [0.15] * 5, 
                 w = [0.1, 0.2, 0.3, 0.2, 0.1]):

        self.dist = [st.norm(m, s) for m,s in zip(mu, sd)]
        self.w = w / np.sum(w)

    def pdf(self, x):

        all_pds =  [rv.pdf(x) * wt for rv, wt in zip(self.dist, self.w)]
        return np.sum(all_pds)

    def rvs(self, n_samples):
        mn_draws = st.multinomial(n_samples, self.w).rvs(1).squeeze()
        samples = [dist.rvs(draws) for draws, dist in zip(mn_draws, self.dist)]
        return np.hstack(samples)
