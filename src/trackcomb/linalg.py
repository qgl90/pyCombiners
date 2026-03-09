"""Vectorized linear algebra primitives for small matrices."""

from __future__ import annotations

import numpy as np


def mahalanobis_2x2(dx, dy, cxx, cxy, cyy):
    """Compute 2x2 Mahalanobis chi2 using analytic inverse."""
    det = cxx * cyy - cxy * cxy
    singular = np.abs(det) < 1e-18
    safe_det = np.where(singular, 1.0, det)
    inv_det = 1.0 / safe_det
    chi2 = (cyy * dx * dx - 2.0 * cxy * dx * dy + cxx * dy * dy) * inv_det
    return np.where(singular, np.nan, chi2)


def solve_2x2(a00, a01, a10, a11, b0, b1):
    """Solve 2x2 linear systems using Cramer's rule."""
    det = a00 * a11 - a01 * a10
    singular = np.abs(det) < 1e-18
    safe_det = np.where(singular, 1.0, det)
    inv_det = 1.0 / safe_det
    x0 = (a11 * b0 - a01 * b1) * inv_det
    x1 = (a00 * b1 - a10 * b0) * inv_det
    x0 = np.where(singular, np.nan, x0)
    x1 = np.where(singular, np.nan, x1)
    return x0, x1, singular


def mahalanobis_3x3(dx, dy, dz, c00, c01, c02, c11, c12, c22):
    """Compute 3x3 symmetric Mahalanobis chi2 using analytic inverse."""
    # Cofactors of symmetric C
    A00 = c11 * c22 - c12 * c12
    A01 = c02 * c12 - c01 * c22
    A02 = c01 * c12 - c02 * c11
    A11 = c00 * c22 - c02 * c02
    A12 = c01 * c02 - c00 * c12
    A22 = c00 * c11 - c01 * c01
    det = c00 * A00 + c01 * A01 + c02 * A02
    singular = np.abs(det) < 1e-30
    safe_det = np.where(singular, 1.0, det)
    inv_det = 1.0 / safe_det
    chi2 = (
        dx * (A00 * dx + A01 * dy + A02 * dz)
        + dy * (A01 * dx + A11 * dy + A12 * dz)
        + dz * (A02 * dx + A12 * dy + A22 * dz)
    ) * inv_det
    return np.where(singular, np.nan, chi2)


def solve_3x3_sym(c00, c01, c02, c11, c12, c22, b0, b1, b2):
    """Solve 3x3 symmetric linear systems using adjugate."""
    A00 = c11 * c22 - c12 * c12
    A01 = c02 * c12 - c01 * c22
    A02 = c01 * c12 - c02 * c11
    A11 = c00 * c22 - c02 * c02
    A12 = c01 * c02 - c00 * c12
    A22 = c00 * c11 - c01 * c01
    det = c00 * A00 + c01 * A01 + c02 * A02
    singular = np.abs(det) < 1e-30
    safe_det = np.where(singular, 1.0, det)
    inv_det = 1.0 / safe_det
    x0 = (A00 * b0 + A01 * b1 + A02 * b2) * inv_det
    x1 = (A01 * b0 + A11 * b1 + A12 * b2) * inv_det
    x2 = (A02 * b0 + A12 * b1 + A22 * b2) * inv_det
    x0 = np.where(singular, np.nan, x0)
    x1 = np.where(singular, np.nan, x1)
    x2 = np.where(singular, np.nan, x2)
    return x0, x1, x2


def inv_3x3_sym(c00, c01, c02, c11, c12, c22):
    """Invert 3x3 symmetric matrices using adjugate."""
    A00 = c11 * c22 - c12 * c12
    A01 = c02 * c12 - c01 * c22
    A02 = c01 * c12 - c02 * c11
    A11 = c00 * c22 - c02 * c02
    A12 = c01 * c02 - c00 * c12
    A22 = c00 * c11 - c01 * c01
    det = c00 * A00 + c01 * A01 + c02 * A02
    singular = np.abs(det) < 1e-30
    safe_det = np.where(singular, 1.0, det)
    inv_det = 1.0 / safe_det
    i00 = np.where(singular, np.nan, A00 * inv_det)
    i01 = np.where(singular, np.nan, A01 * inv_det)
    i02 = np.where(singular, np.nan, A02 * inv_det)
    i11 = np.where(singular, np.nan, A11 * inv_det)
    i12 = np.where(singular, np.nan, A12 * inv_det)
    i22 = np.where(singular, np.nan, A22 * inv_det)
    return i00, i01, i02, i11, i12, i22


def solve_cholesky(A, b):
    """Solve symmetric positive-definite systems via Cholesky decomposition."""
    N, d = b.shape
    x = np.full_like(b, np.nan)
    try:
        L = np.linalg.cholesky(A)
        # Forward substitution: L y = b
        y = np.linalg.solve(L, b[:, :, np.newaxis])[:, :, 0]
        # Back substitution: L^T x = y
        x = np.linalg.solve(np.swapaxes(L, -2, -1), y[:, :, np.newaxis])[
            :, :, 0
        ]
    except np.linalg.LinAlgError:
        # Fallback: solve one-by-one
        for i in range(N):
            try:
                Li = np.linalg.cholesky(A[i])
                yi = np.linalg.solve(Li, b[i])
                x[i] = np.linalg.solve(Li.T, yi)
            except np.linalg.LinAlgError:
                pass  # stays nan
    return x


def inv_cholesky(A):
    """Invert symmetric positive-definite matrices via Cholesky decomposition."""
    N, d, _ = A.shape
    result = np.full_like(A, np.nan)
    try:
        L = np.linalg.cholesky(A)
        eye = np.broadcast_to(np.eye(d), (N, d, d)).copy()
        result = np.linalg.solve(
            np.swapaxes(L, -2, -1),
            np.linalg.solve(L, eye),
        )
    except np.linalg.LinAlgError:
        for i in range(N):
            try:
                Li = np.linalg.cholesky(A[i])
                eye_d = np.eye(d)
                result[i] = np.linalg.solve(Li.T, np.linalg.solve(Li, eye_d))
            except np.linalg.LinAlgError:
                pass  # stays nan
    return result


def mahalanobis_cholesky(delta, A):
    """Compute Mahalanobis chi2 via Cholesky decomposition."""
    x = solve_cholesky(A, delta)
    return np.sum(delta * x, axis=-1)
