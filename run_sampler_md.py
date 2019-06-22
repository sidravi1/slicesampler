from slicesampler_md_slow import SliceSampler

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy.stats as st
from mpl_toolkits import mplot3d
import numpy as np

FIGSIZE = (10, 7)
MU = 0.4
SD = 1

dist = st.multivariate_normal([0.0, 0.0], [[1, 0.0], [0.0, 1.0]])

ss = SliceSampler(2, 2)
samples = ss.sample(dist)

xs = samples["xs"]
proposed_points = samples["proposed_points"]
uprime = samples["uprimes"]

xlim = (MU - 3 * SD, MU + 3 * SD)
ylim_max = dist.pdf(MU)
vals = dist.rvs(10000)

X = np.linspace(-3, 3, 50)
Y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(X, Y)

xr = X.ravel()
yr = Y.ravel()
z = dist.pdf(np.stack((xr, yr)).reshape(-1, order="F").reshape(-1, 2))
z = z.reshape(50, 50)

with plt.style.context("Solarize_Light2"):
    f = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, z, antialiased=False, alpha=0.2)
    curr_point = [0, 0]
    stepout, = ax.plot3D([], [], [], "-g")
    marker = mpl.markers.MarkerStyle("o", fillstyle="none")
    proposal = ax.scatter3D([], [], [], ".k", s=15, marker=marker)
    currline, = ax.plot3D([], [], [], linestyle=":", color="dodgerblue")
    draw_pt = ax.text2D(
        0.75,
        0.9,
        "> Draw u' ~ U(0, p(x))",
        fontsize=10,
        fontfamily="serif",
        transform=ax.transAxes,
    )
    shrink_window = ax.text2D(
        0.75,
        0.85,
        "> Get/shrink window",
        fontsize=10,
        fontfamily="serif",
        transform=ax.transAxes,
    )
    propose_point = ax.text2D(
        0.75,
        0.80,
        "> Propose point",
        fontsize=10,
        fontfamily="serif",
        transform=ax.transAxes,
    )
    get_sample = ax.text2D(
        0.75,
        0.75,
        "> Get sample!",
        fontsize=10,
        fontfamily="serif",
        transform=ax.transAxes,
    )
    all_texts = [draw_pt, shrink_window, propose_point, get_sample]
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Slice Sampler", fontsize=14)
    ax.w_zaxis.line.set_lw(0.0)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.set_zticks([])
x1_samples = []
x2_samples = []
u_samples = []

sample_points = ax.scatter3D(x1_samples, x2_samples, u_samples, marker="*", c="r")

angle = 0
for i, (x, u, p) in enumerate(zip(xs, uprime[:500], proposed_points)):
    currline.set_data([curr_point[0], curr_point[0]], [curr_point[1], curr_point[1]])
    currline.set_3d_properties(zs=[0, dist.pdf(curr_point)])
    [x.set_alpha(0.2) for x in all_texts]
    draw_pt.set_alpha(1.0)
    ax.view_init(30, angle)
    angle += 1
    f.canvas.draw()
    f.canvas.flush_events()
    if i < 4:
        delay = 0.1
    elif (i > 4) and (i < 10):
        delay = 0.01
    else:
        delay = 0.01

    plt.pause(delay)
    for j, (x1, p1) in enumerate(zip(x, p)):
        [x.set_alpha(0.2) for x in all_texts]
        shrink_window.set_alpha(1.0)
        curr_point = p1
        stepout.set_data(x1[:, 0], x1[:, 1])
        stepout.set_3d_properties([u, u])
        ax.view_init(30, angle)
        angle += 1
        f.canvas.draw()
        f.canvas.flush_events()
        plt.pause(delay)
        proposal._offsets3d = ([p1[0]], [p1[1]], [u])
        # proposal.set_3d_properties(u)
        [x.set_alpha(0.2) for x in all_texts]
        propose_point.set_alpha(1.0)
        ax.view_init(30, angle)
        angle += 1
        f.canvas.draw()
        f.canvas.flush_events()
        plt.pause(delay)

    [x.set_alpha(0.2) for x in all_texts]
    get_sample.set_alpha(1.0)
    sample_points._offsets3d = (x1_samples, x2_samples, u_samples)
    #     sample_points.set_data(x1_samples, x2_samples)
    #     sample_points.set_3d_properties(u_samples)
    x1_samples.append(curr_point[0])
    x2_samples.append(curr_point[1])
    u_samples.append(u)
    ax.view_init(30, angle)
    angle += 1
    f.canvas.draw()
    f.canvas.flush_events()
    plt.pause(delay)


plt.show()
