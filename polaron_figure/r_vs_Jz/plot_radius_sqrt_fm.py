"""Plot sqrt(number of FM nearest-neighbor bonds) for several geometries."""

import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# Put each geometry's HDF5 files in its own folder under this single base folder.

HERE = Path(__file__).resolve().parent

# The .h5 files are supposed be put in a folder labeled by the geometry: OBCxOBC is given by NxxNy and cylinder is given by NxcNy 
BASE_FOLDER = HERE / "data"  
OUTPUT_FOLDER = HERE / "results_n_fm"

# The list of geometries I am using, ADD THE GEOMETRIES WITH THE BOUNDARIES HERE IF DATA FOR MORE IS ADDED.
GEOMETRIES = {
    "12x8": (12, 8, False),
    "9x9": (9, 9, False),
}
FILE_GLOB = "data_correlations_nh1_*_t1.0_Jz*.h5"
FIT_MAX_JZ = -0.45  # Change to -0.47 for the notebook's narrower fit window.
FM_THRESHOLD = 0.0


def bond_pairs(nx, ny, periodic_y):
    """I am just re-writing the bondPairs.jl bonds, since I don't know how to take data from .jl file and put it in .py file"""

    if periodic_y and ny % 2:
        raise ValueError("The staggered cylinder requires an even Ny.")

    bonds = []

    # Here, the numbered sites start with 1, not 0. 

    for col in range(1, nx + 1):
        for row in range(1, ny + 1):
            i = row + (col - 1) * ny  

            if row < ny:
                bonds.append((i, i + 1))
            elif periodic_y:
                bonds.append((i, 1 + (col - 1) * ny))

            if col < nx:
                bonds.append((i, i + ny))
                if row % 2 == 1:
                    if row > 1:
                        bonds.append((i, i + ny - 1))
                    elif periodic_y:
                        bonds.append((i, col * ny + ny))
                    if row < ny:
                        bonds.append((i, i + ny + 1))

    bonds = np.sort(np.asarray(bonds, dtype=int) - 1, axis=1)
    if len(set(map(tuple, bonds))) != len(bonds):
        raise ValueError("Duplicate bonds: check the lattice geometry.")
    return bonds

# This function gets the number of FM bonds, and I am make a leap of faith that polaron is circular so, r_{FM} \propto \sqrt{N_{FM}}
def read_radius(path, nx, ny, bonds):
    with h5py.File(path, "r") as data:
        correlation = data["holeSStot_Correlation"][()]
        hole_site = int(data["refSite"][()]) - 1

    if correlation.shape != (nx * ny, nx * ny):
        raise ValueError(f"{path.name}: correlation size disagrees with {nx}x{ny}.")

    # Every pair is stored with i < j, as in an upper-triangle correlation.
    spin_bonds = bonds[(bonds[:, 0] != hole_site) & (bonds[:, 1] != hole_site)]
    values = correlation[spin_bonds[:, 0], spin_bonds[:, 1]]
    n_fm = int(np.count_nonzero(values > FM_THRESHOLD))
    return n_fm, np.sqrt(n_fm), hole_site + 1


def power_law(delta_jz, a, b):
    return a * delta_jz**b


def fit_radius(jz, radius):
    use = (jz > -0.5) & (jz <= FIT_MAX_JZ) & (radius > 0)
    if use.sum() < 3:
        return None, np.zeros_like(use)

    x = jz[use] + 0.5
    y = radius[use]
    # A log fit is only the initial guess. The final fit minimizes errors in R.
    b_guess, log_a_guess = np.polyfit(np.log(x), np.log(y), 1)
    parameters, covariance = curve_fit(
        power_law, x, y, p0=(np.exp(log_a_guess), b_guess), maxfev=10000
    )
    return (parameters, np.sqrt(np.diag(covariance))), use


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    results = {}

    for geometry, (nx, ny, periodic_y) in GEOMETRIES.items():
        folder = BASE_FOLDER / geometry
        files = sorted(folder.glob(FILE_GLOB))
        if not files:
            print(f"No correlation files in {folder}; skipping {geometry}.")
            continue

        bonds = bond_pairs(nx, ny, periodic_y)
        rows = []
        for path in files:
            jz = float(path.stem.rsplit("_Jz", 1)[1])
            n_fm, radius, ref_site = read_radius(path, nx, ny, bonds)
            rows.append((jz, radius, n_fm, ref_site, path.name))
        rows.sort()

        jz_values = np.array([row[0] for row in rows])
        if len(set(jz_values)) != len(jz_values):
            raise ValueError(f"{geometry}: choose one state per Jz before fitting.")
        radii = np.array([row[1] for row in rows])
        fit, fit_mask = fit_radius(jz_values, radii)
        if fit is None:
            print(f"{geometry}: fewer than three points in the fit window.")
        results[geometry] = (rows, fit, fit_mask)

    if not results:
        raise ValueError(f"No HDF5 files found under {BASE_FOLDER}.")

    with (OUTPUT_FOLDER / "sqrt_fm_radii.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["geometry", "Jz", "delta_Jz", "N_FM", "sqrt_N_FM",
                         "refSite_Julia", "used_in_fit", "filename"])
        for geometry, (rows, _, fit_mask) in results.items():
            for row, used in zip(rows, fit_mask):
                jz, radius, n_fm, ref_site, filename = row
                writer.writerow([geometry, jz, jz + 0.5, n_fm, radius,
                                 ref_site, bool(used), filename])

    with (OUTPUT_FOLDER / "sqrt_fm_fits.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["geometry", "a", "b", "a_formal_error", "b_formal_error",
                         "n_fit_points"])
        for geometry, (_, fit, fit_mask) in results.items():
            if fit is not None:
                (a, b), (a_error, b_error) = fit
                writer.writerow([geometry, a, b, a_error, b_error, fit_mask.sum()])

    fig, ax = plt.subplots(figsize=(7.3, 5.3), layout="constrained")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for color, (geometry, (rows, fit, fit_mask)) in zip(colors, results.items()):
        jz = np.array([row[0] for row in rows])
        radius = np.array([row[1] for row in rows])
        positive = (jz > -0.5) & (radius > 0)
        ax.scatter(jz[positive] + 0.5, radius[positive],
                   color=color, s=44, label=f"{geometry} data")
        ax.scatter(jz[positive & ~fit_mask] + 0.5, radius[positive & ~fit_mask],
                   facecolors="white", edgecolors=color, s=44)
        if fit is not None:
            (a, b), _ = fit
            x_line = np.geomspace((jz[positive] + 0.5).min(),
                                  (jz[positive] + 0.5).max(), 300)
            ax.plot(x_line, power_law(x_line, a, b), "--", color=color,
                    lw=1.8, label=rf"{geometry} fit: $b={b:.3f}$")
            print(f"{geometry}: a={a:.6g}, b={b:.6g}, fit points={fit_mask.sum()}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\delta J_z=J_z+0.5$", fontsize = 16)
    ax.set_ylabel(r"$r_{\mathrm{FM}} \propto \sqrt{N_{\mathrm{FM}}}$", fontsize = 16)
    ax.grid(True, which="both", color="0.88", lw=0.6)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(OUTPUT_FOLDER / "sqrt_fm_scaling.pdf")
    fig.savefig(OUTPUT_FOLDER / "sqrt_fm_scaling.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
