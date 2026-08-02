"""Shared fit shapes: Gaussian core, mirrored EMG, double-sided CB."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import exponnorm


def gauss(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def emg_left(x, amp, mu, sigma, tau):
    """Mirrored exponentially-modified Gaussian (tail on the left)."""
    sigma, tau = max(sigma, 1e-4), max(tau, 1e-4)
    return amp * exponnorm.pdf(-x, K=tau / sigma, loc=-mu, scale=sigma)


def dscb(x, amp, mu, sigma, a_lo, n_lo, a_hi, n_hi):
    """Double-sided Crystal Ball: Gaussian core with power-law tails on
    both sides (transition at a_lo/a_hi sigmas, tail exponents n_lo/n_hi).
    """
    sigma = max(sigma, 1e-6)
    a_lo, a_hi = abs(a_lo), abs(a_hi)
    n_lo, n_hi = max(n_lo, 1.01), max(n_hi, 1.01)
    t = (np.asarray(x) - mu) / sigma
    out = np.exp(-0.5 * t**2)
    lo = t < -a_lo
    out[lo] = (
        np.exp(-0.5 * a_lo**2)
        * (a_lo / n_lo * (n_lo / a_lo - a_lo - t[lo])) ** -n_lo
    )
    hi = t > a_hi
    out[hi] = (
        np.exp(-0.5 * a_hi**2)
        * (a_hi / n_hi * (n_hi / a_hi - a_hi + t[hi])) ** -n_hi
    )
    return amp * out


def dscb_fit(values, bins=80, rng=None):
    """Binned double-sided Crystal Ball fit.

    Returns (popt, chi2_ndf) with popt = (amp, mu, sigma, a_lo, n_lo,
    a_hi, n_hi); raises RuntimeError if the fit does not converge.
    """
    values = np.asarray(values)
    med = float(np.median(values))
    core = 0.5 * (np.percentile(values, 84) - np.percentile(values, 16))
    if rng is None:
        rng = (med - 8 * core, med + 8 * core)
    counts, edges = np.histogram(values, bins=bins, range=rng)
    centers = 0.5 * (edges[:-1] + edges[1:])
    popt, _ = curve_fit(
        dscb,
        centers,
        counts,
        p0=[counts.max(), med, core, 1.5, 3.0, 1.5, 3.0],
        bounds=(
            [0, rng[0], 1e-4, 0.1, 1.01, 0.1, 1.01],
            [np.inf, rng[1], np.inf, 10, 100, 10, 100],
        ),
        sigma=np.sqrt(np.maximum(counts, 1)),
        maxfev=50000,
    )
    sel = counts > 0
    chi2 = np.sum(
        (counts[sel] - dscb(centers[sel], *popt)) ** 2
        / np.maximum(counts[sel], 1)
    ) / max(int(sel.sum()) - 7, 1)
    return popt, chi2


def emg_left_fit(values, bins=50, rng=None):
    """Binned mirrored-EMG fit.

    Returns (amp, mu, sigma, tau, chi2_ndf); raises RuntimeError if the
    fit does not converge. rng defaults to a median-centred window.
    """
    values = np.asarray(values)
    med = float(np.median(values))
    if rng is None:
        rng = (med - 0.45, med + 0.3)
    counts, edges = np.histogram(values, bins=bins, range=rng)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binw = edges[1] - edges[0]
    popt, _ = curve_fit(
        emg_left,
        centers,
        counts,
        p0=[len(values) * binw, med + 0.05, 0.05, 0.06],
        sigma=np.sqrt(np.maximum(counts, 1)),
        maxfev=50000,
    )
    sel = counts > 0
    chi2 = np.sum(
        (counts[sel] - emg_left(centers[sel], *popt)) ** 2
        / np.maximum(counts[sel], 1)
    ) / max(int(sel.sum()) - 4, 1)
    amp, mu, sigma, tau = popt
    return float(amp), float(mu), abs(float(sigma)), abs(float(tau)), chi2


def gauss_fit(values):
    """Iterative +-2 sigma core Gaussian fit.

    Returns (mu, sigma, mu_err, sigma_err). Falls back to robust moments
    (median, half the 16-84% spread) if the fit fails or the sample is
    too small.
    """
    values = np.asarray(values)
    mu = float(np.median(values))
    sigma = float(
        0.5 * (np.percentile(values, 84) - np.percentile(values, 16))
    )
    mu_err = sigma / np.sqrt(max(len(values), 1))
    sigma_err = mu_err / np.sqrt(2)
    for _ in range(2):
        core = values[(values > mu - 2 * sigma) & (values < mu + 2 * sigma)]
        if len(core) < 20:
            break
        counts, edges = np.histogram(core, bins=40)
        centers = 0.5 * (edges[:-1] + edges[1:])
        try:
            popt, pcov = curve_fit(
                gauss,
                centers,
                counts,
                p0=[counts.max(), mu, sigma],
                sigma=np.sqrt(np.maximum(counts, 1)),
            )
            mu, sigma = float(popt[1]), abs(float(popt[2]))
            mu_err, sigma_err = np.sqrt(np.diag(pcov))[1:3]
        except RuntimeError:
            break
    return mu, sigma, float(mu_err), float(sigma_err)
