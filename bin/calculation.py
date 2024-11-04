"""
define some functions to calculate quantities.

"""

from constant import (
    PI,
    A_HI,
    HI_MASS,
    FWHM2SIGMA,
    HI_restfreq,
    HI_restwave,
    light_speed,
    Planck_constant,
    Boltzmann_constant,
    Hubble_constant_dimensionless,
    plk, # type: ignore
    np,
    u
)

def dv2df(dv, z= None, freq=HI_restfreq):
    """
    Turn velocity width into frequency width.
    
    Args:
        dv: velocity width.
        z: redshift, default is None. \
            If provided, used the **redshifted** frequency as reference.
        freq: the reference freqeuncy, default is the HI rest frequency.

    Returns:
        the corresponding frequency width.
    """
    if z is not None:
        freq = freq/(1+z)

    df = dv/light_speed * freq

    return df

def df2dv(df, z= None, freq=HI_restfreq):
    """
    Turn frequency width into velocity width.
    
    Args:
        df: the frequency width.
        z: redshift, default is None. \
            If provided, used the **redshifted** frequency as reference.
        freq: the reference freqeuncy, default is the HI rest frequency.
        
    Returns:
        the corresponding velocity width.
    """
    if z is not None:
        freq = freq/(1+z)

    dv = df/freq * light_speed

    return dv
    
def dc_to_z(
    dc,
    box_length,
    box_z_center = 0.1
    ):
    """
    Turn relative comoving distance in the box into redshfit using **differential** method.
    
    Args:
        dc:           relative comoving distance in the box.
        box_length:   length of the simulation box in comoving distance.
        box_z_center: the redshfit of the center of simulation box.

    Returns:
        the redshift z
        
    """
    hubble_z = plk.H(box_z_center)   # in unit km/Mpc/s
    
    z = box_z_center + (dc - box_length/2)/Hubble_constant_dimensionless * (hubble_z /light_speed)
    
    return z

def freq2z(
    freq,
    restfreq=HI_restfreq
    ):
    """
    turn observed frequency into redshfit.
    """
    return restfreq/freq - 1

def z2freq(
    z,
    sight_veocity = None,
    restfreq=HI_restfreq,
    neglect_second_term=True
    ):
    """
    turn redshift into observed frequency.
    
    Args:
        z: redshfit.
        sight_velocity: perculiar velocity in the line of sight.
        restfreq: rest frequency, default is the HI rest frequency.
        neglect_second_term: approximate the reshift by 
            neglecting the second order term $z_{pec}z_{z_cos}$. 

    Returns:
        observed frequency.
    """
    
    if sight_veocity is None:
        return restfreq/(1+z)
    else:
        z_peculiar = (sight_veocity/light_speed).value
        z_obs = z + z_peculiar
        if not neglect_second_term:
            z_obs += z*z_peculiar
        return restfreq/(1+z_obs)
    
def resolution(
    diameter=300*u.m,
    wavelength=HI_restwave,
    factor=1.22
    ):
    """
    get the resolution of the telescope.
    
    Args:
        diameter: diameter of fully illuminated aperture, defalut is 300 m for FAST case.
        wavelength: observing wavelength, default is HI rest wavelength.
        factor: coefficient of resolution formula, default is 1.22.
    
    Returns:
        resolution in unit arcmin.
    """
    return (factor*wavelength/diameter).to(u.arcmin, equivalencies= u.dimensionless_angles())

def mainbeam(
    z=0.,
    diameter=300*u.m,
    **args
    ):
    """
    return the size of main beam at redshift z.
    
    The diameter of the FAST reflector is 500 m,
    but the fully illuminated aperture at any time is D = 300 m.
    Since the telescope is designed to track objects.
    """
    HI_wave_obs = HI_restwave * (1+z)
    return resolution(diameter=diameter, wavelength=HI_wave_obs, **args)

def beam_solid_angle(
    a=mainbeam(),
    b=None
    ):
    """
    calculate the solid angle convered by beam,
    assuming an elliptical gaussian normalised beam sensitivity response model.

    Args:
        a: beam angular major axis, measured at the half power point,
            defalut is the main beam of FAST at z=0.
        b: beam angular minor axis, measured at the half power point,
            default is same with a.
      
    Returns:
        solid angle of the beam.
    """
    
    sigma_x = a * FWHM2SIGMA
    
    if b is not None:
        sigma_y = b * FWHM2SIGMA
        return 2 * PI * sigma_x * sigma_y
    else:
        return 2 * PI * sigma_x**2
    
def drift_integration_time(
    beamsize=mainbeam(),
    dec=26*u.deg,
    angular_velocity=0.25 *u.arcmin/u.s
    ):
    """
    calculate the integration time of the pixel within the FWHM beam width, during the drift scan.
    
    Args:
        beamsize: the FWHM beam width, default is the main beam of FAST at z=0.
        dec: the Declination angle of the pointing direction, default is 26 degree, 
            assuming drift scan pointing at the Zenith of the FAST site.
        angular_velocity: the angular_velocity of teh Earth rotation.
    
    Returns:
        the intergration time.
    """
    return beamsize/(angular_velocity*np.cos(dec))

def get_integration_time_FAST_HIIM(
    z=0.1,
    factor=2,
    **args
    ):
    """
    get the integration time of the pixel within the FWHM beam width, during the FAST drift scan for HIIM.
    
    Args:
        z: redshift, defalut is 0.1.
        factor: default is 2, considering the overlap of 19-beam strips 
            during the FAST dift HIIM survey, most pixels will be scaned twice.
        **args: args passed to function `drift_integration_time`.
    """
    beamsize = mainbeam(z)
    return factor*drift_integration_time(beamsize, **args)

def tb_sky(
    freq=HI_restwave
    ):
    """
    calculate the brightness temperature of the sky.
    
    **High latitude, and away from galactic plane**.
    
    Args:
        freq: the frequency, defalut is the HI rest frequency.
    """
    t = 2.73 + 25.2 * (0.408/freq.to_value(u.GHz))**2.75  # type: ignore
    
    return t *u.K

def circular_area(diameter=300*u.m):
    """
    return the area of the circular aperture.
    """
    return PI * (diameter/2)**2

def SEFD(
    t_sys=20*u.K,
    effective_area=circular_area()
    ):
    """
    calculate the system equivalent flux density.
    
    Args:
        T_sys: the system temperature, defalut is 20 K.
        effective_area: the effective collecting area, default is PI*300**2/4 for FAST.
    
    Returns:
        the SEFD in unit Jy.
    """
    return (2*Boltzmann_constant*t_sys/effective_area).to(u.Jy)

def flux_density_sensitivity(
    t_sys=20*u.K, 
    n_p=2,
    delta_f=0.1*u.MHz,
    delta_t=48*u.s,
    system_efficiency=0.7, 
    effective_area=circular_area()
    ):
    """
    calculate the flux density sensitivity.
    
    Args:
        t_sys: the system temperature, default is 20 K for FAST.
        n_p:  number of polarizations, default is 2.
        delta_f: the frequency resolution, default is 0.1 MHz in our simulation.
        delta_t: the integration time for each pixle, default is 48 s in our simulaiton.
        system_efficiency: the system efficiency, default is 0.7 for FAST.
        effective_area: the effective collecting area, default is the PI*300**2/4 for FAST.
        
    Returns:
        the flux density sensitivity in unit mJy.
    """    
    
    A = SEFD(t_sys, effective_area=effective_area).to(u.mJy)/system_efficiency
    B = (n_p * delta_f * delta_t ).to_value('')
    
    return A/B**0.5

def snu2tb(
    flux_density,
    omega_mb, 
    freq, 
    z=None):
    """
    convert the flux density into brightness temperature using Rayleigh-Jeans approximation.
    
    Args:
        flux_density: the flux density.
        omega_mb: the solid angel of main beam.
        freq: the frequency of flux.
        z: redshift, default is None. 
            **If z was given, additionally multiple a factor (1+z)^3 considering in the relativistical Universe.**
    
    Returns:
        brightness temperature in unit mK.
    """
    
    tb = (flux_density/omega_mb).to(u.mK, equivalencies= u.brightness_temperature(freq))
    
    if z is not None:
        return tb * (1+z)**3
    else: 
        return tb

def brightness_temperature_sensitivity(
    z=0.1,
    t_sys=20*u.K, 
    n_pol=2,
    delta_f=0.1*u.MHz,
    delta_t=48*u.s,
    system_efficiency=0.7, 
    effictive_area=circular_area()
    ):
    """
    calculate the brightness temperature sensitivity.
    
    Args:
        z: redshift.
        t_sys: the system temperature.
        n_pol:  number of polarizations, default is 2.
        delta_f: the frequency resolution, default is 0.1 MHz in our simulation.
        delta_t: the integration time for each pixle, default is 48 s in our simulaiton.
        system_efficiency: the system efficiency, default is 0.7 for FAST.
        effective_area: the effective collecting area, default is the PI*300**2/4 for FAST.
        
    Returns:
        the brightness temperature sensitivty in unit mK.
    """
    
    freq = z2freq(z)
    omega_mb = beam_solid_angle(mainbeam(z))
    sigma_flux_density = flux_density_sensitivity(t_sys, n_pol, delta_f, delta_t, system_efficiency, effictive_area)
    return snu2tb(sigma_flux_density, omega_mb, freq, z=z)

def mass_to_brightness_temperature(
    z=0.1,
    m=1e10 * u.solMass,
    pixel_width=20*u.kpc,
    delta_f=0.1*u.MHz
    ):
    """
    convert the HI mass into brightness temperature.
    
    Notes: 
        1. the solid angle is calculated as the area of the pixel divided by the square of the distance to the source.
        2. do **NOT** consider the an relativistic Universe.
        3. the unit do **NOT** convert to Kelvin, considering the AtrributeError may occur when using the dask.array to do such calculation.
        4. while using dask.array to compute Tb, the return result is an **array type**, not a quantity type. 
        Use the `unit_factor` function to manually convert it to the desired K or mK unit.
        
    Args:
        z: redshift, default is 0.1 in our simulation.
        m: the HI mass, default is 1e10 solar mass **only for test**.
        pixel_width: the pixel width for calculating the solid angel of pixel,
            defalut is 20 kpc in our simulaiton.
        delta_f: the frequency resolution, this is used to convert flux into flux density, 
            simply assuming it's uniform distributed within the frequency width.
            Default is 0.1 MHz in our simulation.
    
    Returns:
        the brightness temperature of a voxel with HI mass `m`, in a unit (**NOT Kelvin**) that need to be decomposed. 
    
    """

    a = 3*light_speed**2 * Planck_constant * A_HI /( 32 * PI * Boltzmann_constant * HI_MASS * HI_restfreq)
    dl = plk.luminosity_distance(z)
    omega_p = (pixel_width/dl) **2
    b = m/(omega_p*delta_f * dl**2)
    c = a*b
        
    return (1+z)**3 *c

def unit_factor(current_unit, target_unit):
    """
    Calculate the conversion factor between two units.
    
    Examples:
        unit_factor('m', 'cm') = 100
        
        unit_factor(mass_to_brightness_temperature().unit, u.K) = 2e-9
        
    """
    return u.Unit(current_unit).to(target_unit)

def get_beam_npix(z=0.1, beamsize= None, pixel_width=20*u.kpc):
    """
    calculate the number of pixels covered by the beam.
    """
    if beamsize is None:
        beamsize = mainbeam(z)
        
    dc = plk.kpc_comoving_per_arcmin(z) * beamsize *FWHM2SIGMA
    return (dc/pixel_width).decompose()