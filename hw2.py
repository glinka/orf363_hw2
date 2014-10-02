>import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gs
from mpl_toolkits.mplot3d import Axes3D

def do_hw():
    einstein_embedding()
    fn_min_max()

def einstein_embedding():
    k_vals = np.array((25, 50, 100, 200))
    image_filename = "albert.jpg"
    recons, recon_errs = svd_embed_image(image_filename, k_vals)
    gspec = gs.GridSpec(3,2)
    fig = plt.figure()
    ax_full = fig.add_subplot(gspec[0,:])
    image = mpimg.imread(image_filename)
    n = image.shape[0]
    m = image.shape[1]
    ax_full.imshow(rgb_to_grayscale(image), cmap="gray")
    ax_full.set_title("Original image")
    ax_recons = [[fig.add_subplot(gspec[i,j]) for j in range(2)] for i in range(1,3)]
    ax_recons = [ax for sublist in ax_recons for ax in sublist]
    for i in range(4):
        ax_recons[i].imshow(recons[i], cmap="gray")
        ax_recons[i].set_title("k=" + str(k_vals[i]))
    plt.show(fig)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(k_vals, recon_errs, zorder=1)
    ax.scatter(k_vals, recon_errs, s=50, c='r', zorder=2)
    ax.set_xlabel("k", fontsize=24)
    ax.set_ylabel(r"$\parallel A - A_{(k)} \parallel_F$", fontsize=24)
    plt.show(fig)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(k_vals, n*m-(n+m+1)*k_vals, zorder=1)
    ax.scatter(k_vals, n*m-(n+m+1)*k_vals, s=50, c='r', zorder=2)
    ax.set_xlabel("k", fontsize=24)
    ax.set_ylabel("Pixel savings", fontsize=24)
    plt.show(fig)


def rgb_to_grayscale(rgb):
    return np.dot(rgb, [0.299, 0.587, 0.144])

def svd_embed_image(image_filename, k_vals):
    image = mpimg.imread(image_filename)
    image = rgb_to_grayscale(image)
    u,s,v = np.linalg.svd(image)
    nvals = k_vals.shape[0]
    n = image.shape[0]
    m = image.shape[1]
    recon_errs = np.empty(nvals)
    recons = np.empty((nvals, n, m))
    for i in range(nvals):
        k = k_vals[i]
        recons[i] = np.dot(u[:,:k], np.dot(np.diag(s[:k]),v[:k,:]))
        recon_errs[i] = np.linalg.norm(image - recons[i])
    return [recons, recon_errs]

def fn_min_max():
    f = lambda x,y: np.power(x, 3)/3.0 - 4*x + np.power(y, 3)/3.0 - 16*y
    n = 50
    l = 6
    xs, ys = np.meshgrid(np.linspace(-l, l, n), np.linspace(-l, l, n))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(xs, ys, f(xs, ys))
    ax.scatter(-2, -4, f(-2, -4), s=100, c='r', label='local extrema')
    ax.scatter(2, 4, f(2, 4), s=100, c='r')
    ax.scatter(-2, 4, f(-2, 4), s=100, c='y', label='saddle points')
    ax.scatter(2, -4, f(2, -4), s=100, c='y')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    plt.show(fig)

if __name__=="__main__":
    do_hw()
