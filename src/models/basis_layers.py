"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math

import numpy as np
import sympy as sym
import torch
from scipy import special as sp
from scipy.optimize import brentq
from scipy.special import binom
from torch_geometric.nn.models.schnet import GaussianSmearing


def Jn(r, n):
    """
    numerical spherical bessel functions of order n
    """
    return sp.spherical_jn(n, r)


def Jn_zeros(n, k):
    """
    Compute the first k zeros of the spherical bessel functions up to order n (excluded)
    """
    zerosj = np.zeros((n, k), dtype="float32")
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype="float32")
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj


def spherical_bessel_formulas(n):
    """
    Computes the sympy formulas for the spherical bessel functions up to order n (excluded)
    """
    x = sym.symbols("x")
    # j_i = (-x)^i * (1/x * d/dx)^î * sin(x)/x
    j = [sym.sin(x) / x]  # j_0
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        j += [sym.simplify(b * (-x) ** i)]
        a = sym.simplify(b)
    return j


def bessel_basis(n, k):
    """
    Compute the sympy formulas for the normalized and rescaled spherical bessel functions up to
    order n (excluded) and maximum frequency k (excluded).

    Returns:
        bess_basis: list
            Bessel basis formulas taking in a single argument x.
            Has length n where each element has length k. -> In total n*k many.
    """
    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1) ** 2]
        normalizer_tmp = (
            1 / np.array(normalizer_tmp) ** 0.5
        )  # sqrt(2/(j_l+1)**2) , sqrt(1/c**3) not taken into account yet
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sym.symbols("x")
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [
                sym.simplify(
                    normalizer[order][i]
                    * f[order].subs(x, zeros[order, i] * x)
                )
            ]
        bess_basis += [bess_basis_tmp]
    return bess_basis


def sph_harm_prefactor(l_degree, m_order):
    """Computes the constant pre-factor for the spherical harmonic of degree l and order m.

    Parameters
    ----------
        l_degree: int
            Degree of the spherical harmonic. l >= 0
        m_order: int
            Order of the spherical harmonic. -l <= m <= l

    Returns
    -------
        factor: float

    """
    # sqrt((2*l+1)/4*pi * (l-m)!/(l+m)! )
    return (
        (2 * l_degree + 1)
        / (4 * np.pi)
        * np.math.factorial(l_degree - abs(m_order))
        / np.math.factorial(l_degree + abs(m_order))
    ) ** 0.5


def associated_legendre_polynomials(
    L_maxdegree, zero_m_only=True, pos_m_only=True
):
    """Computes string formulas of the associated legendre polynomials up to degree L (excluded).

    Parameters
    ----------
        L_maxdegree: int
            Degree up to which to calculate the associated legendre polynomials (degree L is excluded).
        zero_m_only: bool
            If True only calculate the polynomials for the polynomials where m=0.
        pos_m_only: bool
            If True only calculate the polynomials for the polynomials where m>=0. Overwritten by zero_m_only.

    Returns
    -------
        polynomials: list
            Contains the sympy functions of the polynomials (in total L many if zero_m_only is True else L^2 many).
    """
    # calculations from http://web.cmb.usc.edu/people/alber/Software/tomominer/docs/cpp/group__legendre__polynomials.html
    z = sym.symbols("z")
    P_l_m = [
        [0] * (2 * l_degree + 1) for l_degree in range(L_maxdegree)
    ]  # for order l: -l <= m <= l

    P_l_m[0][0] = 1
    if L_maxdegree > 0:
        if zero_m_only:
            # m = 0
            P_l_m[1][0] = z
            for l_degree in range(2, L_maxdegree):
                P_l_m[l_degree][0] = sym.simplify(
                    (
                        (2 * l_degree - 1) * z * P_l_m[l_degree - 1][0]
                        - (l_degree - 1) * P_l_m[l_degree - 2][0]
                    )
                    / l_degree
                )
            return P_l_m
        else:
            # for m >= 0
            for l_degree in range(1, L_maxdegree):
                P_l_m[l_degree][l_degree] = sym.simplify(
                    (1 - 2 * l_degree)
                    * (1 - z ** 2) ** 0.5
                    * P_l_m[l_degree - 1][l_degree - 1]
                )  # P_00, P_11, P_22, P_33

            for m_order in range(0, L_maxdegree - 1):
                P_l_m[m_order + 1][m_order] = sym.simplify(
                    (2 * m_order + 1) * z * P_l_m[m_order][m_order]
                )  # P_10, P_21, P_32, P_43

            for l_degree in range(2, L_maxdegree):
                for m_order in range(l_degree - 1):  # P_20, P_30, P_31
                    P_l_m[l_degree][m_order] = sym.simplify(
                        (
                            (2 * l_degree - 1)
                            * z
                            * P_l_m[l_degree - 1][m_order]
                            - (l_degree + m_order - 1)
                            * P_l_m[l_degree - 2][m_order]
                        )
                        / (l_degree - m_order)
                    )

            if not pos_m_only:
                # for m < 0: P_l(-m) = (-1)^m * (l-m)!/(l+m)! * P_lm
                for l_degree in range(1, L_maxdegree):
                    for m_order in range(
                        1, l_degree + 1
                    ):  # P_1(-1), P_2(-1) P_2(-2)
                        P_l_m[l_degree][-m_order] = sym.simplify(
                            (-1) ** m_order
                            * np.math.factorial(l_degree - m_order)
                            / np.math.factorial(l_degree + m_order)
                            * P_l_m[l_degree][m_order]
                        )

            return P_l_m


def real_sph_harm(L_maxdegree, use_theta, use_phi=True, zero_m_only=True):
    """
    Computes formula strings of the the real part of the spherical harmonics up to degree L (excluded).
    Variables are either spherical coordinates phi and theta (or cartesian coordinates x,y,z) on the UNIT SPHERE.

    Parameters
    ----------
        L_maxdegree: int
            Degree up to which to calculate the spherical harmonics (degree L is excluded).
        use_theta: bool
            - True: Expects the input of the formula strings to contain theta.
            - False: Expects the input of the formula strings to contain z.
        use_phi: bool
            - True: Expects the input of the formula strings to contain phi.
            - False: Expects the input of the formula strings to contain x and y.
            Does nothing if zero_m_only is True
        zero_m_only: bool
            If True only calculate the harmonics where m=0.

    Returns
    -------
        Y_lm_real: list
            Computes formula strings of the the real part of the spherical harmonics up
            to degree L (where degree L is not excluded).
            In total L^2 many sph harm exist up to degree L (excluded). However, if zero_m_only only is True then
            the total count is reduced to be only L many.
    """
    z = sym.symbols("z")
    P_l_m = associated_legendre_polynomials(L_maxdegree, zero_m_only)
    if zero_m_only:
        # for all m != 0: Y_lm = 0
        Y_l_m = [[0] for l_degree in range(L_maxdegree)]
    else:
        Y_l_m = [
            [0] * (2 * l_degree + 1) for l_degree in range(L_maxdegree)
        ]  # for order l: -l <= m <= l

    # convert expressions to spherical coordiantes
    if use_theta:
        # replace z by cos(theta)
        theta = sym.symbols("theta")
        for l_degree in range(L_maxdegree):
            for m_order in range(len(P_l_m[l_degree])):
                if not isinstance(P_l_m[l_degree][m_order], int):
                    P_l_m[l_degree][m_order] = P_l_m[l_degree][m_order].subs(
                        z, sym.cos(theta)
                    )

    ## calculate Y_lm
    # Y_lm = N * P_lm(cos(theta)) * exp(i*m*phi)
    #             { sqrt(2) * (-1)^m * N * P_l|m| * sin(|m|*phi)   if m < 0
    # Y_lm_real = { Y_lm                                           if m = 0
    #             { sqrt(2) * (-1)^m * N * P_lm * cos(m*phi)       if m > 0

    for l_degree in range(L_maxdegree):
        Y_l_m[l_degree][0] = sym.simplify(
            sph_harm_prefactor(l_degree, 0) * P_l_m[l_degree][0]
        )  # Y_l0

    if not zero_m_only:
        phi = sym.symbols("phi")
        for l_degree in range(1, L_maxdegree):
            # m > 0
            for m_order in range(1, l_degree + 1):
                Y_l_m[l_degree][m_order] = sym.simplify(
                    2 ** 0.5
                    * (-1) ** m_order
                    * sph_harm_prefactor(l_degree, m_order)
                    * P_l_m[l_degree][m_order]
                    * sym.cos(m_order * phi)
                )
            # m < 0
            for m_order in range(1, l_degree + 1):
                Y_l_m[l_degree][-m_order] = sym.simplify(
                    2 ** 0.5
                    * (-1) ** m_order
                    * sph_harm_prefactor(l_degree, -m_order)
                    * P_l_m[l_degree][m_order]
                    * sym.sin(m_order * phi)
                )

        # convert expressions to cartesian coordinates
        if not use_phi:
            # replace phi by atan2(y,x)
            x = sym.symbols("x")
            y = sym.symbols("y")
            for l_degree in range(L_maxdegree):
                for m_order in range(len(Y_l_m[l_degree])):
                    Y_l_m[l_degree][m_order] = sym.simplify(
                        Y_l_m[l_degree][m_order].subs(phi, sym.atan2(y, x))
                    )
    return Y_l_m

class PolynomialEnvelope(torch.nn.Module):
    """
    Polynomial envelope function that ensures a smooth cutoff.

    Parameters
    ----------
        exponent: int
            Exponent of the envelope function.
    """

    def __init__(self, exponent):
        super().__init__()
        assert exponent > 0
        self.p = exponent
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, d_scaled):
        env_val = (
            1
            + self.a * d_scaled ** self.p
            + self.b * d_scaled ** (self.p + 1)
            + self.c * d_scaled ** (self.p + 2)
        )
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))


class ExponentialEnvelope(torch.nn.Module):
    """
    Exponential envelope function that ensures a smooth cutoff,
    as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
    SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
    and Nonlocal Effects
    """

    def __init__(self):
        super().__init__()

    def forward(self, d_scaled):
        env_val = torch.exp(
            -(d_scaled ** 2) / ((1 - d_scaled) * (1 + d_scaled))
        )
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))


class SphericalBesselBasis(torch.nn.Module):
    """
    1D spherical Bessel basis

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    """

    def __init__(
        self,
        num_radial: int,
        cutoff: float,
    ):
        super().__init__()
        self.norm_const = math.sqrt(2 / (cutoff ** 3))
        # cutoff ** 3 to counteract dividing by d_scaled = d / cutoff

        # Initialize frequencies at canonical positions
        self.frequencies = torch.nn.Parameter(
            data=torch.tensor(
                np.pi * np.arange(1, num_radial + 1, dtype=np.float32)
            ),
            requires_grad=True,
        )

    def forward(self, d_scaled):
        return (
            self.norm_const
            / d_scaled[:, None]
            * torch.sin(self.frequencies * d_scaled[:, None])
        )  # (num_edges, num_radial)


class BernsteinBasis(torch.nn.Module):
    """
    Bernstein polynomial basis,
    as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
    SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
    and Nonlocal Effects

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    pregamma_initial: float
        Initial value of exponential coefficient gamma.
        Default: gamma = 0.5 * a_0**-1 = 0.94486,
        inverse softplus -> pregamma = log e**gamma - 1 = 0.45264
    """

    def __init__(
        self,
        num_radial: int,
        pregamma_initial: float = 0.45264,
    ):
        super().__init__()
        prefactor = binom(num_radial - 1, np.arange(num_radial))
        self.register_buffer(
            "prefactor",
            torch.tensor(prefactor, dtype=torch.float),
            persistent=False,
        )

        self.pregamma = torch.nn.Parameter(
            data=torch.tensor(pregamma_initial, dtype=torch.float),
            requires_grad=True,
        )
        self.softplus = torch.nn.Softplus()

        exp1 = torch.arange(num_radial)
        self.register_buffer("exp1", exp1[None, :], persistent=False)
        exp2 = num_radial - 1 - exp1
        self.register_buffer("exp2", exp2[None, :], persistent=False)

    def forward(self, d_scaled):
        gamma = self.softplus(self.pregamma)  # constrain to positive
        exp_d = torch.exp(-gamma * d_scaled)[:, None]
        return (
            self.prefactor * (exp_d ** self.exp1) * ((1 - exp_d) ** self.exp2)
        )


class RadialBasis(torch.nn.Module):
    """

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    rbf: dict = {"name": "gaussian"}
        Basis function and its hyperparameters.
    envelope: dict = {"name": "polynomial", "exponent": 5}
        Envelope function and its hyperparameters.
    """

    def __init__(
        self,
        num_radial: int,
        cutoff: float,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
    ):
        super().__init__()
        self.inv_cutoff = 1 / cutoff

        env_name = envelope["name"].lower()
        env_hparams = envelope.copy()
        del env_hparams["name"]

        if env_name == "polynomial":
            self.envelope = PolynomialEnvelope(**env_hparams)
        elif env_name == "exponential":
            self.envelope = ExponentialEnvelope(**env_hparams)
        else:
            raise ValueError(f"Unknown envelope function '{env_name}'.")

        rbf_name = rbf["name"].lower()
        rbf_hparams = rbf.copy()
        del rbf_hparams["name"]

        # RBFs get distances scaled to be in [0, 1]
        if rbf_name == "gaussian":
            self.rbf = GaussianSmearing(
                start=0, stop=1, num_gaussians=num_radial, **rbf_hparams
            )
        elif rbf_name == "spherical_bessel":
            self.rbf = SphericalBesselBasis(
                num_radial=num_radial, cutoff=cutoff, **rbf_hparams
            )
        elif rbf_name == "bernstein":
            self.rbf = BernsteinBasis(num_radial=num_radial, **rbf_hparams)
        else:
            raise ValueError(f"Unknown radial basis function '{rbf_name}'.")

    def forward(self, d):
        d_scaled = d * self.inv_cutoff

        env = self.envelope(d_scaled)
        return env[:, None] * self.rbf(d_scaled)  # (nEdges, num_radial)


class CircularBasisLayer(torch.nn.Module):
    """
    2D Fourier Bessel Basis

    Parameters
    ----------
    num_spherical: int
        Controls maximum frequency.
    radial_basis: RadialBasis
        Radial basis functions
    cbf: dict
        Name and hyperparameters of the cosine basis function
    efficient: bool
        Whether to use the "efficient" summation order
    """

    def __init__(
        self,
        num_spherical: int,
        radial_basis: RadialBasis,
        cbf: str,
        efficient: bool = False,
    ):
        super().__init__()

        self.radial_basis = radial_basis
        self.efficient = efficient

        cbf_name = cbf["name"].lower()
        cbf_hparams = cbf.copy()
        del cbf_hparams["name"]

        if cbf_name == "gaussian":
            self.cosφ_basis = GaussianSmearing(
                start=-1, stop=1, num_gaussians=num_spherical, **cbf_hparams
            )
        elif cbf_name == "spherical_harmonics":
            Y_lm = real_sph_harm(
                num_spherical, use_theta=False, zero_m_only=True
            )
            sph_funcs = []  # (num_spherical,)

            # convert to tensorflow functions
            z = sym.symbols("z")
            modules = {"sin": torch.sin, "cos": torch.cos, "sqrt": torch.sqrt}
            m_order = 0  # only single angle
            for l_degree in range(len(Y_lm)):  # num_spherical
                if (
                    l_degree == 0
                ):  # Y_00 is only a constant -> function returns value and not tensor
                    first_sph = sym.lambdify(
                        [z], Y_lm[l_degree][m_order], modules
                    )
                    sph_funcs.append(
                        lambda z: torch.zeros_like(z) + first_sph(z)
                    )
                else:
                    sph_funcs.append(
                        sym.lambdify([z], Y_lm[l_degree][m_order], modules)
                    )
            self.cosφ_basis = lambda cosφ: torch.stack(
                [f(cosφ) for f in sph_funcs], dim=1
            )
        else:
            raise ValueError(f"Unknown cosine basis function '{cbf_name}'.")

    def forward(self, D_ca, cosφ_cab, id3_ca):
        rbf = self.radial_basis(D_ca)  # (num_edges, num_radial)
        cbf = self.cosφ_basis(cosφ_cab)  # (num_triplets, num_spherical)

        if not self.efficient:
            rbf = rbf[id3_ca]  # (num_triplets, num_radial)
            out = (rbf[:, None, :] * cbf[:, :, None]).view(
                -1, rbf.shape[-1] * cbf.shape[-1]
            )
            return (out,)
            # (num_triplets, num_radial * num_spherical)
        else:
            return (rbf[None, :, :], cbf)
            # (1, num_edges, num_radial), (num_edges, num_spherical)
