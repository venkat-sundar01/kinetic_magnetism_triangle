"""Plot bond-map radii for several lattice geometries.

The first estimate is the zero crossing of the hole-centered bond profile.
Inspect the saved bond maps, then put any visually chosen radii in
RADIUS_OVERRIDES below. All distances are in nearest-neighbor spacings.
"""

import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle
from scipy.optimize import curve_fit


HERE = Path(__file__).resolve().parent
BASE_FOLDER = HERE / "data"  # Or set this to your existing r_Jz folder.
OUTPUT_FOLDER = HERE / "results_direct"

# The names match the geometry folders in the E-versus-Jz notebook.
# The last entry says whether the short direction is periodic.
GEOMETRIES = {
    "12x8": (12, 8, False),
    "15c6": (15, 6, True),
    "9x9": (9, 9, False),
}
FILE_GLOB = "data_correlations_nh1_*_t1.0_Jz*.h5"
FIT_MAX_JZ = -0.45  # Change to -0.47 for the notebook's narrower fit window.
BIN_WIDTH = 0.5  # Width of radial shells, in lattice spacings.

# After inspecting results/bond_maps, enter your measured radius for each Jz.
# An entry here replaces the automatic shell estimate in the fit and plot.
# Example: "9x9": {-0.485: 3.7, -0.48: 3.5}
RADIUS_OVERRIDES = {
    "12x8": {},
    "15c6": {},
    "9x9": {},
}


def bond_pairs(nx, ny, periodic_y):
    """bondPairs.jl bonds, plus the wrapped bonds for an XC cylinder."""
    if periodic_y and ny % 2:
        raise ValueError("The staggered cylinder requires an even Ny.")

    bonds = []
    for col in range(1, nx + 1):
        for row in range(1, ny + 1):
            i = row + (col - 1) * ny  # Julia's one-based site index.

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


def site_coordinates(nx, ny):
    """Triangular lattice with nearest-neighbor separation one."""
    return np.array([
        (col - 1 + (0.5 if row % 2 else 0.0),
         (row - 1) * np.sqrt(3.0) / 2.0)
        for col in range(1, nx + 1)
        for row in range(1, ny + 1)
    ])


def measure_bonds(path, nx, ny, bonds, coordinates, periodic_y):
    with h5py.File(path, "r") as data:
        correlation = data["holeSStot_Correlation"][()]
        hole_site = int(data["refSite"][()]) - 1

    if correlation.shape != (nx * ny, nx * ny):
        raise ValueError(f"{path.name}: correlation size disagrees with {nx}x{ny}.")
    if not 0 <= hole_site < nx * ny or not np.isfinite(correlation).all():
        raise ValueError(f"{path.name}: invalid hole site or correlations.")

    # A bond touching the projected hole does not contain two spins.
    spin_bonds = bonds[(bonds[:, 0] != hole_site) & (bonds[:, 1] != hole_site)]
    values = correlation[spin_bonds[:, 0], spin_bonds[:, 1]]

    first = coordinates[spin_bonds[:, 0]]
    second = coordinates[spin_bonds[:, 1]]
    displacement = second - first
    circumference = ny * np.sqrt(3.0) / 2.0
    if periodic_y:
        displacement[:, 1] = (
            (displacement[:, 1] + circumference / 2) % circumference
            - circumference / 2
        )
    midpoints = first + displacement / 2
    from_hole = midpoints - coordinates[hole_site]
    if periodic_y:
        from_hole[:, 1] = (from_hole[:, 1] + circumference / 2) % circumference - circumference / 2
    distances = np.linalg.norm(from_hole, axis=1)

    # For a hole shifted toward an open end, use the side facing the bulk.
    hole_col = hole_site // ny + 1
    space_left = hole_col - 1
    space_right = nx - hole_col
    if abs(space_left - space_right) <= 1:
        side = "all"
        use = np.ones(len(values), dtype=bool)
    elif space_left > space_right:
        side = "left"
        use = midpoints[:, 0] < coordinates[hole_site, 0]
    else:
        side = "right"
        use = midpoints[:, 0] > coordinates[hole_site, 0]

    return hole_site, side, spin_bonds, values, distances[use], values[use]


def shell_radius(distances, values):
    """First positive-to-nonpositive crossing of radial shell means."""
    edges = np.arange(0, distances.max() + 2 * BIN_WIDTH, BIN_WIDTH)
    centers = []
    means = []
    for left, right in zip(edges[:-1], edges[1:]):
        in_shell = (distances >= left) & (distances < right)
        if in_shell.any():
            centers.append((left + right) / 2)
            means.append(values[in_shell].mean())

    for k in range(len(means) - 1):
        if means[k] > 0 >= means[k + 1]:
            fraction = means[k] / (means[k] - means[k + 1])
            return centers[k] + fraction * (centers[k + 1] - centers[k])
    return None


def power_law(delta_jz, a, b):
    return a * delta_jz**b


def fit_radius(jz, radius):
    use = (jz > -0.5) & (jz <= FIT_MAX_JZ) & (radius > 0)
    if use.sum() < 3:
        return None, np.zeros_like(use)

    x = jz[use] + 0.5
    y = radius[use]
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
        coordinates = site_coordinates(nx, ny)
        rows = []
        for path in files:
            jz = float(path.stem.rsplit("_Jz", 1)[1])
            hole_site, side, spin_bonds, values, distances, selected_values = measure_bonds(
                path, nx, ny, bonds, coordinates, periodic_y
            )
            automatic = shell_radius(distances, selected_values)
            override = RADIUS_OVERRIDES[geometry].get(jz)
            if automatic is None and override is None:
                raise ValueError(f"{path.name}: no shell zero crossing; enter a radius override.")
            if override is not None and override <= 0:
                raise ValueError(f"{geometry}, Jz={jz}: radius must be positive.")
            radius = automatic if override is None else override
            source = "shell estimate" if override is None else "visually chosen"
            rows.append((jz, radius, automatic, source, hole_site + 1,
                         side, path.name))

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

    with (OUTPUT_FOLDER / "direct_radii.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["geometry", "Jz", "delta_Jz", "radius", "shell_estimate",
                         "radius_source", "refSite_Julia", "side_used",
                         "used_in_fit", "filename"])
        for geometry, (rows, _, fit_mask) in results.items():
            for row, used in zip(rows, fit_mask):
                jz, radius, automatic, source, ref_site, side, filename = row
                writer.writerow([geometry, jz, jz + 0.5, radius, automatic,
                                 source, ref_site, side, bool(used), filename])

    with (OUTPUT_FOLDER / "direct_fits.csv").open("w", newline="") as file:
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
    ax.set_xlabel(r"$\delta J_z=J_z+0.5$")
    ax.set_ylabel("Radius (nearest-neighbor spacings)")
    ax.grid(True, which="both", color="0.88", lw=0.6)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(OUTPUT_FOLDER / "direct_scaling.pdf")
    fig.savefig(OUTPUT_FOLDER / "direct_scaling.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
