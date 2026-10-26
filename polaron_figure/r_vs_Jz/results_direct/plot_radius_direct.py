"""Measure polaron radii from nearest-neighbor bonds and fit their scaling.

The script first suggests a radius from radial averages of the bond correlations.
Use plot_radius_bond_maps_csv.py to inspect the bond maps, then enter any
radii chosen by eye in RADIUS_OVERRIDES. Distances are in lattice spacings.
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# ----------------------------- User settings -----------------------------

HERE = Path(__file__).resolve().parent  # Replace with Path("...") in a notebook.
BASE_FOLDER = HERE / "data"  # Contains the 12x8, 15c6, and 9x9 folders.
OUTPUT_FOLDER = HERE / "results"

# (Nx, Ny, periodic in the y direction)
GEOMETRIES = {
    "12x8": (12, 8, False),
    "15c6": (15, 6, True),
    "9x9": (9, 9, False),
}
FILE_GLOB = "data_correlations_nh1_*_t1.0_Jz*.h5"
FIT_MAX_JZ = -0.45
BIN_WIDTH = 0.5

# Example: "9x9": {-0.485: 3.7, -0.48: 3.5}
# Leave a geometry empty to use the automatically suggested radii.
# Keep these overrides the same in plot_radius_bond_maps_csv.py.
RADIUS_OVERRIDES = {
    "12x8": {},
    "15c6": {},
    "9x9": {},
}


# ----------------------------- Lattice -----------------------------

def bond_pairs(nx, ny, periodic_y):
    """Return bondPairs.jl bonds, including the seam of an XC cylinder."""
    if periodic_y and ny % 2:
        raise ValueError("The staggered cylinder requires an even Ny.")

    bonds = []
    for col in range(1, nx + 1):
        for row in range(1, ny + 1):
            site = row + (col - 1) * ny  # Julia's one-based site number.

            if row < ny:
                bonds.append((site, site + 1))
            elif periodic_y:
                bonds.append((site, 1 + (col - 1) * ny))

            if col == nx:
                continue

            bonds.append((site, site + ny))
            if row % 2 == 1:
                if row > 1:
                    bonds.append((site, site + ny - 1))
                elif periodic_y:
                    bonds.append((site, col * ny + ny))
                if row < ny:
                    bonds.append((site, site + ny + 1))

    bonds = np.sort(np.asarray(bonds, dtype=int) - 1, axis=1)
    if len(set(map(tuple, bonds))) != len(bonds):
        raise ValueError("Duplicate nearest-neighbor bonds.")
    return bonds


def site_coordinates(nx, ny):
    """Triangular-lattice positions with nearest-neighbor distance one."""
    return np.array([
        (col - 1 + (0.5 if row % 2 else 0.0),
         (row - 1) * np.sqrt(3.0) / 2.0)
        for col in range(1, nx + 1)
        for row in range(1, ny + 1)
    ])


def minimum_image_y(dy, circumference):
    """Shortest y displacement around the periodic direction."""
    return (dy + circumference / 2) % circumference - circumference / 2


# ----------------------------- Radius measurement -----------------------------

def read_spin_bonds(path, nx, ny, bonds):
    """Read the correlation on each spin bond, excluding the hole's bonds."""
    with h5py.File(path, "r") as data:
        correlation = data["holeSStot_Correlation"][()]
        hole_site = int(data["refSite"][()]) - 1

    if correlation.shape != (nx * ny, nx * ny):
        raise ValueError(f"{path.name}: correlation size disagrees with {nx}x{ny}.")
    if not 0 <= hole_site < nx * ny or not np.isfinite(correlation).all():
        raise ValueError(f"{path.name}: invalid hole site or correlations.")

    spin_bonds = bonds[(bonds[:, 0] != hole_site) & (bonds[:, 1] != hole_site)]
    values = correlation[spin_bonds[:, 0], spin_bonds[:, 1]]
    return hole_site, spin_bonds, values


def radial_profile(nx, ny, periodic_y, coordinates, hole_site, bonds, values):
    """Bond distances and correlations on the side used for the radius."""
    first = coordinates[bonds[:, 0]]
    second = coordinates[bonds[:, 1]]
    step = second - first

    circumference = ny * np.sqrt(3.0) / 2.0
    if periodic_y:
        step[:, 1] = minimum_image_y(step[:, 1], circumference)
    midpoints = first + step / 2

    from_hole = midpoints - coordinates[hole_site]
    if periodic_y:
        from_hole[:, 1] = minimum_image_y(from_hole[:, 1], circumference)
    distances = np.linalg.norm(from_hole, axis=1)

    # A shifted hole uses only the half-plane facing the open-boundary bulk.
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

    return side, distances[use], values[use]


def shell_radius(distances, values):
    """Interpolate the first positive-to-nonpositive radial shell crossing."""
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


# ----------------------------- Fit and output -----------------------------

def power_law(delta_jz, a, b):
    return a * delta_jz**b


def fit_radius(jz, radius):
    used = (jz > -0.5) & (jz <= FIT_MAX_JZ) & (radius > 0)
    if used.sum() < 3:
        return None, np.zeros_like(used)

    x = jz[used] + 0.5
    y = radius[used]
    b_guess, log_a_guess = np.polyfit(np.log(x), np.log(y), 1)
    parameters, covariance = curve_fit(
        power_law, x, y, p0=(np.exp(log_a_guess), b_guess), maxfev=10000
    )
    return (parameters, np.sqrt(np.diag(covariance))), used


def analyze_geometry(geometry, nx, ny, periodic_y):
    """Measure all files for one geometry and fit its radius curve."""
    paths = sorted((BASE_FOLDER / geometry).glob(FILE_GLOB))
    if not paths:
        print(f"No correlation files in {BASE_FOLDER / geometry}; skipping {geometry}.")
        return None

    bonds = bond_pairs(nx, ny, periodic_y)
    coordinates = site_coordinates(nx, ny)
    points = []
    seen_jz = {}

    for path in paths:
        jz = float(path.stem.rsplit("_Jz", 1)[1])
        if jz in seen_jz:
            raise ValueError(
                f"{geometry}: multiple states at Jz={jz:g}: "
                f"{seen_jz[jz].name} and {path.name}. Keep one state per Jz."
            )
        seen_jz[jz] = path

        hole_site, spin_bonds, values = read_spin_bonds(path, nx, ny, bonds)
        side, distances, radial_values = radial_profile(
            nx, ny, periodic_y, coordinates, hole_site, spin_bonds, values
        )
        estimate = shell_radius(distances, radial_values)
        override = RADIUS_OVERRIDES[geometry].get(jz)
        if estimate is None and override is None:
            raise ValueError(f"{path.name}: no shell zero crossing; enter a radius override.")
        if override is not None and override <= 0:
            raise ValueError(f"{geometry}, Jz={jz}: radius must be positive.")

        radius = estimate if override is None else override
        source = "shell estimate" if override is None else "visually chosen"
        points.append({
            "jz": jz,
            "radius": radius,
            "estimate": estimate,
            "source": source,
            "ref_site": hole_site + 1,
            "side": side,
            "filename": path.name,
        })

    points.sort(key=lambda point: point["jz"])
    jz = np.array([point["jz"] for point in points])
    radius = np.array([point["radius"] for point in points])
    fit, used = fit_radius(jz, radius)
    if fit is None:
        print(f"{geometry}: fewer than three points in the fit window.")
    return {"points": points, "fit": fit, "used": used}


def plot_scaling(results):
    """Plot every geometry and its fit on shared logarithmic axes."""
    fig, ax = plt.subplots(figsize=(7.3, 5.3), layout="constrained")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for color, (geometry, result) in zip(colors, results.items()):
        points = result["points"]
        jz = np.array([point["jz"] for point in points])
        radius = np.array([point["radius"] for point in points])
        used = result["used"]
        positive = (jz > -0.5) & (radius > 0)

        ax.scatter(jz[positive] + 0.5, radius[positive],
                   color=color, s=44, label=f"{geometry} data")
        ax.scatter(jz[positive & ~used] + 0.5, radius[positive & ~used],
                   facecolors="white", edgecolors=color, s=44)

        if result["fit"] is not None:
            (a, b), _ = result["fit"]
            x_line = np.geomspace((jz[positive] + 0.5).min(),
                                  (jz[positive] + 0.5).max(), 300)
            ax.plot(x_line, power_law(x_line, a, b), "--", color=color,
                    lw=1.8, label=rf"{geometry} fit: $b={b:.3f}$")
            print(f"{geometry}: a={a:.6g}, b={b:.6g}, fit points={used.sum()}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\delta J_z=J_z+0.5$")
    ax.set_ylabel("Radius (nearest-neighbor spacings)")
    ax.grid(True, which="both", color="0.88", lw=0.6)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(OUTPUT_FOLDER / "direct_scaling.pdf")
    fig.savefig(OUTPUT_FOLDER / "direct_scaling.png", dpi=300)
    plt.close(fig)


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    results = {}
    for geometry, (nx, ny, periodic_y) in GEOMETRIES.items():
        result = analyze_geometry(geometry, nx, ny, periodic_y)
        if result is not None:
            results[geometry] = result

    if not results:
        raise ValueError(f"No HDF5 files found under {BASE_FOLDER}.")

    plot_scaling(results)


if __name__ == "__main__":
    main()
