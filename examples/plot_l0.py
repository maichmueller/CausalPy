from causalpy.neural_networks import HardConcreteDist, BinaryConcreteDist
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

if __name__ == "__main__":
    plt.style.use(["science", "bright"])
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(5.5, 7), sharex=True, sharey=True
    )
    # fig.add_subplot(111, frameon=False)
    # # hide tick and tick label of the big axes
    # plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    # plt.grid(False)
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    colors = plt.get_cmap("tab20").colors

    x = np.linspace(1e-5, 1, 10000, endpoint=False)
    i = 0
    for (alpha, beta) in [
        (0.0, 0.1),
        (0.0, 0.3),
        (0.0, 0.5),
        (0.0, 0.7),
    ]:
        bc = BinaryConcreteDist(log_alpha=alpha, beta=beta)
        hc = HardConcreteDist(bc, zeta=1.1, gamma=-0.1)

        y_hc = np.array([hc.pdf(z) for z in x])
        y_bc = np.array([bc.pdf(z) for z in x])

        ax1.plot(
            x,
            y_bc,
            color=colors[i],
            lw=1.5,
            label=r"BC ($\log \alpha, \lambda) = ("
            + f"{int(alpha)}"
            + ", "
            + f"{beta})$",
        )
        ax2.plot(
            x,
            y_hc,
            color=colors[i],
            lw=1.5,
            label=r"HC ($p_0, p_1  = "
            + f"{np.round(hc.pdf(0).numpy(), decimals=2):.2f}"
            + ", "
            + f"{np.round(hc.pdf(1).numpy(), decimals=2):.2f})$",
        )

        i += 2

    i = 0
    for (alpha, beta) in [
        (1.0, 0.1),
        (1.0, 0.3),
        (1.0, 0.5),
        (1.0, 0.7),
    ]:
        bc = BinaryConcreteDist(log_alpha=alpha, beta=beta)
        hc = HardConcreteDist(bc, zeta=1.1, gamma=-0.1)

        y_hc = np.array([hc.pdf(z) for z in x])
        y_bc = np.array([bc.pdf(z) for z in x])

        ax3.plot(
            x,
            y_bc,
            color=colors[i],
            lw=1.5,
            label=r"BC ($\log  \alpha, \lambda) = ("
            + f"{int(alpha)}"
            + ", "
            + f"{beta})$",
        )
        ax4.plot(
            x,
            y_hc,
            color=colors[i],
            lw=1.5,
            label=r"HC ($p_0, p_1  = "
            + f"{np.round(hc.pdf(0).numpy(), decimals=2):.2f}"
            + ", "
            + f"{np.round(hc.pdf(1).numpy(), decimals=2):.2f})$",
        )

        i += 2

    plt.ylim(0, 3)
    # plt.suptitle("Binary Concrete and associated Hard Concrete distributions", y=0.95)
    ax1.set_ylabel("Density")
    ax3.set_ylabel("Density")
    ax3.set_xlabel("$s$")
    ax4.set_xlabel("$z$")
    ax1.legend(loc="upper center", fancybox=True, shadow=True)
    ax2.legend(loc="upper center", fancybox=True, shadow=True)
    ax3.legend(loc="upper center", fancybox=True, shadow=True)
    ax4.legend(loc="upper center", fancybox=True, shadow=True)
    plt.legend(loc="upper center", fancybox=True, shadow=True)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.savefig("./plots/bc_hc_plot.pdf")
    plt.show()
