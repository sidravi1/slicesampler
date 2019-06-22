import numpy as np
import scipy.stats as st
import scipy as sp
from tqdm import tqdm


def get_interval(point, dist, u_prime, w, dim=1):

    # random line to draw from
    delta_x = np.random.uniform(-1, 1, size=point.shape[0])

    r = np.random.uniform(0, 1)

    xl = point - r * w * delta_x
    xr = point + (1 - r) * w * delta_x

    while dist.pdf(xl) > u_prime:
        xl = xl - w * delta_x
    while dist.pdf(xr) > u_prime:
        xr = xr + w * delta_x

    xs = [(xl, xr, delta_x)]
    # print("get interval:", xl, xr,thetas, w)
    return xs


def modify_interval(point, proposed_point, xl, xr):

    xl_to_prop = sp.spatial.distance.euclidean(proposed_point, xl)
    xl_to_pt = sp.spatial.distance.euclidean(point, xl)

    if xl_to_prop > xl_to_pt:
        xr = proposed_point
    else:
        xl = proposed_point

    return xl, xr


class SliceSampler(object):
    def __init__(self, dims=1, w=0.1):
        self.dims = dims
        self.w = w

    def sample(self, dist, n_samples=1000, max_iters=100, init=[]):
        if len(init) == 0:
            point = np.zeros(self.dims)
        else:
            point = init
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
            xs_pt = get_interval(point, dist, u_prime, self.w, dim)
            xl, xr, delta = xs_pt[-1]
            xs = [(xl, xr)]
            # print(xs[-1])
            n_iters = 0
            while n_iters < max_iters:
                proposed_point = xl + (xr - xl) * np.random.uniform(0, 1)
                proposed_points.append(proposed_point)
                p_star = dist.pdf(proposed_point)
                # print("probs:", p_star, u_prime)
                # print("points:", xl, xr, proposed_point, point)
                if p_star > u_prime:
                    samples[sample_counter] = proposed_point
                    point = proposed_point
                    all_proposed_points.append(np.stack(proposed_points))
                    break
                else:
                    xl, xr = modify_interval(point, proposed_point, xl, xr)
                    xs.append((xl, xr))
                    n_iters += 1
            all_xs.append(np.array(xs).reshape(-1, 2, dim))
            if n_iters == max_iters:
                raise RuntimeError(
                    ("maximum iterations for modify interval " "reached")
                )
        return {
            "samples": samples,
            "xs": all_xs,
            "proposed_points": np.array(all_proposed_points),
            "uprimes": np.hstack(u_primes),
        }


class MixtureGaussians(object):
    def __init__(
        self, mu=np.linspace(-2, 2, 5), sd=[0.15] * 5, w=[0.1, 0.2, 0.3, 0.2, 0.1]
    ):

        self.dist = [st.norm(m, s) for m, s in zip(mu, sd)]
        self.w = w / np.sum(w)

    def pdf(self, x):

        all_pds = [rv.pdf(x) * wt for rv, wt in zip(self.dist, self.w)]
        return np.sum(all_pds)

    def rvs(self, n_samples):
        mn_draws = st.multinomial(n_samples, self.w).rvs(1).squeeze()
        samples = [dist.rvs(draws) for draws, dist in zip(mn_draws, self.dist)]
        return np.hstack(samples)
