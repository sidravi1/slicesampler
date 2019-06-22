from slicesampler_slow import SliceSampler
from slicesampler_slow import MixtureGaussians

import matplotlib.pyplot as plt
import seaborn as sns

FIGSIZE = (10, 7)
MU = 0.4
SD = 1

dist = MixtureGaussians()

ss = SliceSampler(1, 2)
samples = ss.sample(dist)

xs = samples["xs"]
proposed_points = samples["proposed_points"]
uprime = samples["uprimes"]

xlim = (MU - 3 * SD, MU + 3 * SD)
ylim_max = dist.pdf(MU)
vals = dist.rvs(10000)

with plt.style.context("Solarize_Light2"):
    f, ax = plt.subplots(figsize=(10, 7))
    sns.distplot(vals, color="gray", bins=100)
    curr_point = 0
    stepout, = plt.plot([], [], "-g")
    proposal, = plt.plot([], [], ".k", ms=15, fillstyle="none")
    currline = plt.vlines([2], 0, 0, linestyle=":", color="dodgerblue")
    draw_pt = plt.text(
        MU + SD, 0.8, "> Draw u' ~ U(0, p(x))", fontsize=10, fontfamily="serif"
    )
    shrink_window = plt.text(
        MU + SD, 0.75, "> Get/shrink window", fontsize=10, fontfamily="serif"
    )
    propose_point = plt.text(
        MU + SD, 0.7, "> Propose point", fontsize=10, fontfamily="serif"
    )
    get_sample = plt.text(
        MU + SD, 0.65, "> Get sample!", fontsize=10, fontfamily="serif"
    )
    all_texts = [draw_pt, shrink_window, propose_point, get_sample]
    ax.set_xlabel("x")
    ax.set_ylabel("P(x)")
    ax.set_title("Slice Sampler", fontsize=14)
x_samples = []
u_samples = []

sample_points, = plt.plot(x_samples, u_samples, "*r")


for i, (x, u, p) in enumerate(zip(xs, uprime[:500], proposed_points)):
    currline.set_verts([[[curr_point, 0], [curr_point, dist.pdf(curr_point)]]])
    [x.set_alpha(0.2) for x in all_texts]
    draw_pt.set_alpha(1.0)

    f.canvas.draw()
    f.canvas.flush_events()
    if i < 4:
        delay = 1
    elif (i > 4) and (i < 10):
        delay = 0.1
    else:
        delay = 0.01

    plt.pause(delay)
    for j, (x1, p1) in enumerate(zip(x, p)):
        [x.set_alpha(0.2) for x in all_texts]
        shrink_window.set_alpha(1.0)
        curr_point = p1
        stepout.set_data(x1, [u, u])
        f.canvas.draw()
        f.canvas.flush_events()
        plt.pause(delay)
        proposal.set_data(p1, u)
        [x.set_alpha(0.2) for x in all_texts]
        propose_point.set_alpha(1.0)
        f.canvas.draw()
        f.canvas.flush_events()
        plt.pause(delay)

    [x.set_alpha(0.2) for x in all_texts]
    get_sample.set_alpha(1.0)
    sample_points.set_data(x_samples, u_samples)
    x_samples.append(curr_point)
    u_samples.append(u)
    f.canvas.draw()
    f.canvas.flush_events()
    plt.pause(delay)


plt.show()
