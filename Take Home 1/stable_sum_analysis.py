from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, stats

plt.style.use("seaborn-v0_8-darkgrid")


@dataclass(frozen=True)
class StableParams:
    alpha: float
    beta: float = 0.0
    scale: float = 1.0
    loc: float = 0.0

    def pdf(self, x: np.ndarray | float) -> np.ndarray | float:
        return stats.levy_stable.pdf(
            x,
            self.alpha,
            self.beta,
            loc=self.loc,
            scale=self.scale,
        )

    def rvs(self, size: int, random_state: np.random.Generator) -> np.ndarray:
        return stats.levy_stable.rvs(
            self.alpha,
            self.beta,
            loc=self.loc,
            scale=self.scale,
            size=size,
            random_state=random_state,
        )

    def characteristic_function(self, t: np.ndarray | float) -> np.ndarray | float:
        # Symmetric, zero-location, unit-scale stable cf in S0 parameterization.
        return np.exp(-(np.abs(t) ** self.alpha))


def sum_characteristic_function(t: np.ndarray | float, p1: StableParams, p2: StableParams) -> np.ndarray | float:
    return p1.characteristic_function(t) * p2.characteristic_function(t)


def density_via_convolution(
    x_grid: Iterable[float],
    p1: StableParams,
    p2: StableParams,
    integration_limit: float = np.inf,
) -> np.ndarray:
    results = np.empty_like(np.asarray(x_grid, dtype=float))
    for idx, x in enumerate(results):
        integrand = lambda u: p1.pdf(u) * p2.pdf(x - u)
        val, _ = integrate.quad(integrand, -integration_limit, integration_limit, limit=400)
        results[idx] = val
    return results


def density_via_simulation(
    x_grid: Iterable[float],
    p1: StableParams,
    p2: StableParams,
    n_samples: int = 250_000,
    bandwidth: float | None = None,
    seed: int = 7,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x1 = p1.rvs(size=n_samples, random_state=rng)
    x2 = p2.rvs(size=n_samples, random_state=rng)
    s = x1 + x2
    kde = stats.gaussian_kde(s, bw_method=bandwidth)
    return kde(np.asarray(x_grid, dtype=float))


def density_via_cf_inversion(
    x_grid: Iterable[float],
    p1: StableParams,
    p2: StableParams,
) -> np.ndarray:
    results = np.empty_like(np.asarray(x_grid, dtype=float))
    cf = lambda t: np.real(sum_characteristic_function(t, p1, p2))

    for idx, x in enumerate(results):
        integrand = lambda t: np.cos(t * x) * cf(t)
        val, _ = integrate.quad(integrand, 0, np.inf, limit=500)
        results[idx] = val / np.pi
    return results


def density_via_fft(
    x_grid: Iterable[float],
    p1: StableParams,
    p2: StableParams,
    oversample_factor: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    x_grid = np.asarray(x_grid, dtype=float)
    n_target = x_grid.size
    n_fft = int(2 ** np.ceil(np.log2(n_target * oversample_factor)))
    x_min, x_max = x_grid.min(), x_grid.max()
    dx = (x_max - x_min) / (n_fft - 1)
    x_fft = np.linspace(x_min, x_min + dx * (n_fft - 1), n_fft)
    freq = np.fft.fftfreq(n_fft, d=dx) * 2 * np.pi
    phi_vals = sum_characteristic_function(freq, p1, p2)
    pdf_fft = np.fft.ifft(phi_vals)
    pdf_fft = np.real(pdf_fft) * (freq[1] - freq[0]) / (2 * np.pi)
    pdf_fft = np.fft.fftshift(pdf_fft)
    x_fft_shifted = np.fft.fftshift(np.linspace(0, dx * (n_fft - 1), n_fft) + x_min)
    return x_fft_shifted, pdf_fft


def interpolate_density(x_source: np.ndarray, density: np.ndarray, x_target: np.ndarray) -> np.ndarray:
    return np.interp(x_target, x_source, density, left=0.0, right=0.0)


def build_plot(
    x_values: np.ndarray,
    density_map: Dict[str, np.ndarray],
    title: str,
    filename: pathlib.Path,
    x_limits: Tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, values in density_map.items():
        ax.plot(x_values, values, label=label, linewidth=2)
    ax.set_xlim(*x_limits)
    ax.set_xlabel("x")
    ax.set_ylabel("f_S(x)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def main() -> None:
    params1 = StableParams(alpha=1.3)
    params2 = StableParams(alpha=1.7)

    x_full = np.linspace(-10, 10, 501)

    print("Computing density via direct convolution ...")
    density_conv = density_via_convolution(x_full, params1, params2)

    print("Computing density via simulation + KDE ...")
    density_sim = density_via_simulation(x_full, params1, params2)

    print("Computing density via characteristic-function inversion ...")
    density_cf = density_via_cf_inversion(x_full, params1, params2)

    print("Computing density via FFT-based inversion ...")
    x_fft, density_fft_raw = density_via_fft(x_full, params1, params2)
    density_fft = interpolate_density(x_fft, density_fft_raw, x_full)

    density_map = {
        "Convolution (integral)": density_conv,
        "Simulation + KDE": density_sim,
        "CF inversion": density_cf,
        "FFT inversion": density_fft,
    }

    output_dir = pathlib.Path.cwd()

    print("Building full-range plot ...")
    build_plot(
        x_values=x_full,
        density_map=density_map,
        title="Stable Sum Density Comparison",
        filename=output_dir / "stable_sum_density.png",
        x_limits=(-10, 10),
    )

    tail_mask = (x_full >= 4) & (x_full <= 10)
    x_tail = x_full[tail_mask]
    density_tail_map = {label: values[tail_mask] for label, values in density_map.items()}

    print("Building tail-range plot ...")
    build_plot(
        x_values=x_tail,
        density_map=density_tail_map,
        title="Stable Sum Density Comparison (Tail 4-10)",
        filename=output_dir / "stable_sum_density_tail.png",
        x_limits=(4, 10),
    )

    print("Outputs saved to:")
    print(output_dir / "stable_sum_density.png")
    print(output_dir / "stable_sum_density_tail.png")


if __name__ == "__main__":
    main()
