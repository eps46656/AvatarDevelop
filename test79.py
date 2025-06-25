import numpy as np
import matplotlib.pyplot as plt
import numpy as np


def main1():

    show_lb = -0.6
    show_rb = +0.6

    # Define the SDF range and target interval
    s = np.linspace(show_lb, show_rb, 1000)
    d_lb = -0.1
    d_rb = +0.1

    # Define the piecewise potential energy function
    def potential_energy_loss(s, d_lb, d_rb):
        energy = np.zeros_like(s)
        left_mask = s < d_lb
        right_mask = s > d_rb
        energy[left_mask] = (d_lb - s[left_mask]) ** 2
        energy[right_mask] = (s[right_mask] - d_rb) ** 2
        return energy

    energy = potential_energy_loss(s, d_lb, d_rb)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(
        s, energy, label='Potential Energy Loss ($\\mathcal{L}_{\\mathrm{ML},\\mathrm{pe}}$)', color='royalblue')
    plt.axvline(d_lb, linestyle='--', color='gray',
                label='$d_{lb}$ / $d_{rb}$')
    plt.axvline(d_rb, linestyle='--', color='gray')
    plt.fill_between(s, 0, energy.max(), where=(s >= d_lb) & (
        s <= d_rb), color='lightgreen', alpha=0.7, label='Zero-loss interval')

    plt.title('Potential Energy Loss for Mesh Layering Loss')
    plt.xlabel('Signed Distance Value ($s$)')
    plt.ylabel(
        'Potential Energy Loss ($\\mathcal{L}_{\\mathrm{ML},\\mathrm{pe}}$)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main1()
