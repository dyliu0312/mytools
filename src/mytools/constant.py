"""
define some constants and conversions

"""

from astropy.cosmology import Planck15 as plk
from astropy.constants import c, k_B, m_p, m_e, h # type: ignore
import numpy as np
import astropy.units as u

# physical constants
PI = np.pi
Planck_constant = h
Boltzmann_constant = k_B

# cosmology constants
light_speed = c.to(u.km/u.s)
Hubble_constant_dimensionless = plk.h

# HI constants
A_HI = 2.86888e-15 / u.s # HI spontaneous emission rate
HI_MASS = m_p + m_e
HI_restfreq = 1420405751.78*u.Hz
HI_restwave = HI_restfreq.to(u.cm, equivalencies=u.spectral())

# conversion constants
SIGMA2FWHM = (8 * np.log(2))**0.5  # turn guass sigma into full width at half maximum power point
FWHM2SIGMA = 1/SIGMA2FWHM