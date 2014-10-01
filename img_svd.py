import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gs
from mpl_toolkits.mplot3d import Axes3D

def do_hw():
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
    ax_25 = fig.add_subplot(gspec[1,0])
    ax_25.imshow(recons[0], cmap="gray")
    ax_25.set_title("k=25")
    ax_50 = fig.add_subplot(gspec[1,1])
    ax_50.imshow(recons[1], cmap="gray")
    ax_50.set_title("k=50")
    ax_100 = fig.add_subplot(gspec[2,0])
    ax_100.imshow(recons[2], cmap="gray")
    ax_100.set_title("k=100")
    ax_200 = fig.add_subplot(gspec[2,1])
    ax_200.imshow(recons[3], cmap="gray")
    ax_200.set_title("k=200")
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
    m = 6
    xs, ys = np.meshgrid(np.linspace(-m, m, n), np.linspace(-m, m, n))
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

