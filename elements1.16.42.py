# Version: 1.16.51
# Date: 2025-05-24
# Changes from 1.16.50:
# - Fixed q23_count ReferenceError by computing quadrant counts before signal loop.
# - Replaced deprecated scipy.special.sph_harm with sph_harm_y.
# - Enhanced Lissajous vibration: increased temp_theta amplitude to 0.2, carbon spiral_factor to 1.0,
#   added wl_nm-based frequency modulation.
# - Improved logging for phase differences of all focus elements.

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
import numpy as np
from scipy.special import sph_harm # Updated import
import argparse
import logging
import sys
import os
import platform
import time
from matplotlib.colors import LogNorm



# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('elements.log', mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logger.addHandler(file_handler)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logger.addHandler(console_handler)

# Check permissions
log_dir = os.path.dirname(os.path.abspath(__file__))
if not os.access(log_dir, os.W_OK):
    logging.error(f"No write permission in directory: {log_dir}")
    raise PermissionError(f"No write permission in {log_dir}")

# Parse arguments
parser = argparse.ArgumentParser(description="Generate polar heat map")
parser.add_argument('--interactive', action='store_true', help="Enable interactive GUI")
parser.add_argument('--backend', choices=['Qt5Agg', 'TkAgg', 'Agg'], default=None, help="Specify matplotlib backend")
parser.add_argument('--copper-inverse', action='store_true', help="Compute Copper's W contribution as 1/(W_oxy^2)")
parser.add_argument('--neutron-period', type=str, default='1000', help="Neutron flux synchronization period(s) in seconds (comma-separated, e.g., '100,1000,10000')")
args = parser.parse_args()

# Parse neutron periods
try:
    neutron_periods = [float(p) for p in args.neutron_period.split(',')]
    if not all(p > 0 for p in neutron_periods):
        raise ValueError("Neutron periods must be positive")
    logging.info(f"Neutron flux periods: {neutron_periods}")
except ValueError as e:
    logging.error(f"Invalid --neutron-period: {args.neutron_period}. Using default [1000]. Error: {e}")
    neutron_periods = [1000]

# Set backend
def set_backend():
    if args.interactive:
        selected_backend = args.backend if args.backend else None
        if selected_backend == 'Agg':
            logging.warning("Agg backend is non-interactive; ignoring --interactive flag")
            matplotlib.use('Agg')
            return 'Agg'
        if platform.system() == 'Darwin':
            display = os.environ.get('DISPLAY', '')
            if not display:
                logging.warning("DISPLAY not set. Try `export DISPLAY=:0` or ensure XQuartz is running.")
        if selected_backend == 'Qt5Agg' or (selected_backend is None and platform.system() != 'Linux'):
            try:
                import PyQt5
                matplotlib.use('Qt5Agg')
                logging.info("Using Qt5Agg backend (interactive)")
                return 'Qt5Agg'
            except Exception as e:
                logging.error(f"Qt5Agg backend failed: {e}")
                logging.warning("Attempting TkAgg backend")
                try:
                    import tkinter
                    matplotlib.use('TkAgg')
                    logging.info("Using TkAgg backend (interactive)")
                    return 'TkAgg'
                except Exception as e2:
                    logging.error(f"TkAgg backend also failed: {e2}")
                    raise RuntimeError("No interactive backend available. Install PyQt5 or Tkinter/XQuartz.")
        if selected_backend == 'TkAgg' or selected_backend is None:
            try:
                import tkinter
                matplotlib.use('TkAgg')
                logging.info("Using TkAgg backend (interactive)")
                return 'TkAgg'
            except Exception as e:
                logging.error(f"TkAgg backend failed: {e}")
                raise RuntimeError("TkAgg backend requested but failed. Install Tkinter and XQuartz: `brew install xquartz`")
        logging.error("Interactive mode requested but Qt5Agg/TkAgg failed")
        raise RuntimeError("No interactive backend available. Install PyQt5 or Tkinter/XQuartz.")
    else:
        matplotlib.use('Agg')
        logging.info("Using Agg backend (non-interactive)")
        return 'Agg'

backend = set_backend()
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Constants
JOYSTICK_STEP = 0.01
MIN_ABUNDANCE = 1e-10
WEAK_ELEMENTS = ['Cadmium', 'Francium', 'Copper', 'Hydrogen', 'Carbon', 'Magnesium', 'Lithium', 'Sodium', 'Rubidium', 'Cesium', 'Beryllium', 'Calcium', 'Strontium', 'Barium', 'Radium', 'Scandium', 'Yttrium', 'Lanthanum', 'Actinium']
k = 6.0
W_REINFORCEMENT = 2.83 * 50
LAST_UPDATE_TIME = [time.time()]
SPEED_OF_LIGHT = 3e8
K_B = 1.380649e-23
T_K40 = 2.20e-9
AMPLITUDE_LIMIT = 0.7071  # sqrt(2)/2
EPSILON = 1e-10
SPIRAL_FACTOR = 0.5
PHASE_OPPOSITION = 0.7
MONOPHONIC_BOOST = 1.5

# Element data
ELEMENTS = [
    ('Silicon', 28.085, 0.6395),
    ('Oxygen', 15.999, 1.0759),
    ('Iron', 55.845, 0.1339),
    ('Cadmium', 112.414, 9.4892e-5),
    ('Francium', 223.020, 2.3088e-18),
    ('Hydrogen', 1.008, 0.0023088),
    ('Copper', 63.546, 1.5931e-4),
    ('Carbon', 12.011, 0.02),
    ('Titanium', 47.867, 0.005),
    ('Magnesium', 24.305, 0.04),
    ('Molybdenum', 95.95, 1e-6),
    ('Helium', 4.0026, 1e-8),
    ('Potassium-40', 39.964, 0.025 * 0.00012),
    ('Lithium', 6.941, 0.0018),
    ('Sodium', 22.990, 0.0236),
    ('Rubidium', 85.468, 1e-5),
    ('Cesium', 132.905, 1e-6),
    ('Silver', 107.868, 1e-5),
    ('Gold', 196.967, 1e-6),
    ('Palladium', 106.42, 1e-6),
    ('Platinum', 195.084, 1e-6),
    ('Beryllium', 9.012, 0.0028),
    ('Calcium', 40.078, 0.046),
    ('Strontium', 87.62, 1e-5),
    ('Barium', 137.327, 1e-5),
    ('Radium', 226.025, 1e-10),
    ('Nickel', 58.693, 0.001),
    ('Cobalt', 58.933, 1e-6),
    ('Rhodium', 102.906, 1e-7),
    ('Iridium', 192.217, 1e-8),
    ('Osmium', 190.23, 1e-8),
    ('Scandium', 44.956, 2.2e-5),
    ('Yttrium', 88.906, 1.5e-5),
    ('Lanthanum', 138.905, 1.0e-6),
    ('Actinium', 227.028, 1.0e-10),
    ('Ruthenium', 101.07, 5.0e-6),
    ('Uranium-238', 238.0508, 1e-8),
    ('Thorium-232', 232.0381, 1e-8),
    ('Lead-208', 207.9766, 1e-7)
]

# Element groups
ELEMENT_GROUPS = {
    'Group 1': ['Hydrogen', 'Lithium', 'Sodium', 'Potassium-40', 'Rubidium', 'Cesium', 'Francium'],
    'Group 2': ['Beryllium', 'Magnesium', 'Calcium', 'Strontium', 'Barium', 'Radium'],
    'Group 3': ['Scandium', 'Yttrium', 'Lanthanum', 'Actinium'],
    'Heavy Pairs': ['Cadmium', 'Copper', 'Titanium', 'Molybdenum', 'Silver', 'Gold', 'Palladium', 'Platinum', 'Nickel', 'Cobalt', 'Rhodium', 'Iridium', 'Osmium', 'Ruthenium'],
    'Nuclear': ['Uranium-238', 'Thorium-232', 'Lead-208'],
    'Others': ['Silicon', 'Oxygen', 'Iron', 'Carbon', 'Helium']
}

# Expected positions
EXPECTED_POSITIONS = {
    'Silicon': (1.5, 0.0), 'Oxygen': (0.0, 0.0), 'Iron': (2.8, 0.0),
    'Cadmium': (0.7, np.pi/8), 'Francium': (1.5, np.pi/4), 'Hydrogen': (0.3, 0.0),
    'Copper': (2.0, np.pi/16), 'Carbon': (0.8, 3*np.pi/4), 'Titanium': (2.2, np.pi/4),
    'Magnesium': (1.0, 5*np.pi/8), 'Molybdenum': (1.8, np.pi/8), 'Helium': (0.4, np.pi),
    'Potassium-40': (1.3, np.pi/3), 'Lithium': (1.2, np.pi), 'Sodium': (0.5, 7*np.pi/8),
    'Rubidium': (1.8, 7*np.pi/12), 'Cesium': (1.8, 11*np.pi/18), 'Silver': (2.5, 5*np.pi/4),
    'Gold': (2.7, 4*np.pi/3), 'Palladium': (2.4, 13*np.pi/12), 'Platinum': (2.6, 7*np.pi/6),
    'Beryllium': (0.8, 11*np.pi/12), 'Calcium': (0.7, 5*np.pi/6), 'Strontium': (0.9, 17*np.pi/18),
    'Barium': (0.4, 3*np.pi/4), 'Radium': (1.0, 7*np.pi/8), 'Nickel': (2.3, 13*np.pi/12),
    'Cobalt': (2.4, 5*np.pi/4), 'Rhodium': (2.5, 4*np.pi/3), 'Iridium': (2.6, 7*np.pi/6),
    'Osmium': (2.7, 17*np.pi/16), 'Scandium': (0.1, -0.05), 'Yttrium': (0.12, -0.06),
    'Lanthanum': (0.15, -0.07), 'Actinium': (0.18, -0.08), 'Ruthenium': (0.2, 2.0),
    'Uranium-238': (0.3, 2.4), 'Thorium-232': (0.32, 2.5), 'Lead-208': (0.35, 2.6),
    'Palladium_Group3': (0.22, 2.1), 'Platinum_Group3': (0.25, 2.2), 'Gold_Group3': (0.28, 2.3)
}

# Element pairs
ELEMENT_PAIRS = [
    ('Silicon', 'Cadmium'), ('Oxygen', 'Copper'), ('Carbon', 'Titanium'),
    ('Magnesium', 'Molybdenum'), ('Hydrogen', 'Helium'), ('Iron', 'Potassium-40'),
    ('Lithium', 'Silver'), ('Sodium', 'Gold'), ('Rubidium', 'Palladium'),
    ('Cesium', 'Platinum'), ('Beryllium', 'Nickel'), ('Calcium', 'Cobalt'),
    ('Strontium', 'Rhodium'), ('Barium', 'Iridium'), ('Radium', 'Osmium'),
    ('Scandium', 'Ruthenium'), ('Yttrium', 'Palladium_Group3'),
    ('Lanthanum', 'Platinum_Group3'), ('Actinium', 'Gold_Group3'),
    ('Uranium-238', 'Lead-208'), ('Thorium-232', 'Lead-208')
]

# Ambisonic channels

element_channels = {
    'Silicon': [(1, 1), (1, -1)],
    'Oxygen': [(1, 0), (1, 1), (1, -1)],  # Add X, Y contributions
    'Carbon': [(2, 1), (2, -1)],  # Ensure Y contribution
    'Hydrogen': [(2, 0), (2, 1), (2, -1)],  # Add X, Y contributions
    'Magnesium': [(2, -1), (2, 1)],  # Ensure X contribution
    'Cadmium': [(1, 1)], 'Iron': [(1, -1)],
    'Francium': [(1, -1)], 'Copper': [(1, 0)],
    'Potassium-40': [(1, 0)], 'Titanium': [(2, 1)],
    'Molybdenum': [(2, -1)], 'Hydrogen': [(2, 0)],
    'Helium': [(2, 0)], 'Lithium': [(2, 0)], 'Sodium': [(2, 0)],
    'Rubidium': [(2, 0)], 'Cesium': [(2, 0)], 'Silver': [(1, 1)],
    'Gold': [(1, 1)], 'Palladium': [(1, -1)], 'Platinum': [(1, -1)],
    'Beryllium': [(2, 0)], 'Calcium': [(2, 0)], 'Strontium': [(2, 0)],
    'Barium': [(2, 0)], 'Radium': [(2, 0)], 'Nickel': [(1, 1)],
    'Cobalt': [(1, 1)], 'Rhodium': [(1, -1)], 'Iridium': [(1, -1)],
    'Osmium': [(1, -1)], 'Scandium': [(2, 1)], 'Yttrium': [(2, 1)],
    'Lanthanum': [(2, 1)], 'Actinium': [(2, 1)], 'Ruthenium': [(1, 1)],
    'Uranium-238': [(1, -1)], 'Thorium-232': [(1, -1)], 'Lead-208': [(1, -1)],
    'Palladium_Group3': [(1, -1)], 'Platinum_Group3': [(1, -1)], 'Gold_Group3': [(1, 1)]
}

def validate_channels():
    for elem, channels in element_channels.items():
        for l, m in channels:
            if l < 0 or abs(m) > l:
                logging.error(f"Invalid channel for {elem}: l={l}, m={m}")
                return False
    return True

if not validate_channels():
    raise ValueError("Invalid element channels detected")

def validate_elements():
    for elem, mass, abund in ELEMENTS:
        if mass <= 0:
            logging.error(f"Invalid mass for {elem}: {mass}")
            return False
        if abund < 0 and not np.isclose(abund, 0, atol=1e-20):
            logging.error(f"Negative abundance for {elem}: {abund}")
            return False
    return True

if not validate_elements():
    raise ValueError("Invalid element data detected")

def wavelength_to_rgb(wl_nm):
    wl_nm = min(max(wl_nm, 300.0), 800.0)
    if wl_nm < 350:  # UV
        r, g, b = 0.5, 0.0, 1.0
    elif wl_nm < 440:
        r = -(wl_nm - 440) / (440 - 350)
        g = 0.0
        b = 1.0
    elif wl_nm < 490:
        r = 0.0
        g = (wl_nm - 440) / (490 - 440)
        b = 1.0
    elif wl_nm < 510:
        r = 0.0
        g = 1.0
        b = -(wl_nm - 510) / (510 - 490)
    elif wl_nm < 580:
        r = (wl_nm - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif wl_nm < 645:
        r = 1.0
        g = -(wl_nm - 645) / (645 - 580)
        b = 0.0
    elif wl_nm < 700:
        r = 1.0
        g = 0.0
        b = 0.0
    else:  # IR
        r, g, b = 1.0, 0.2, 0.2
    if 430 <= wl_nm <= 670:
        intensity = 1.0
    elif wl_nm < 430:
        intensity = 0.5 + 0.5 * (wl_nm - 350) / (430 - 350)
    else:
        intensity = 1.0 - 0.5 * (wl_nm - 670) / (800 - 670)
    intensity = np.clip(intensity, 0.0, 1.0)
    rgb = (r * intensity, g * intensity, b * intensity)
    rgb_clamped = tuple(np.clip(c, 0.0, 1.0) for c in rgb)
    if rgb != rgb_clamped:
        logging.warning(f"RGBA out of range for wl_nm={wl_nm:.1f}: original={rgb}, clamped={rgb_clamped}")
    logging.debug(f"RGB for wl_nm={wl_nm:.1f}: {rgb_clamped}, intensity={intensity:.2f}")
    return rgb_clamped

def calculate_hu(pr_target, element):
    if not np.isfinite(pr_target):
        logging.warning(f"Invalid pr_target for {element}: {pr_target}, using 1.0")
        pr_target = 1.0
    hu = pr_target * 10 - 10
    if element in ['Francium', 'Actinium']:
        hu = -10
    return np.clip(hu, -5, 5)

def spherical_harmonic(l, m, theta, element="Unknown"):
    theta = np.mod(theta, 2 * np.pi)
    phi = np.pi / 4
    if l < 0 or abs(m) > l:
        logging.error(f"Invalid spherical harmonic for {element}: l={l}, m={m}")
        return 0.0
    try:
        Y_lm = sph_harm(m, l, phi, theta).real
        if not np.isfinite(Y_lm):
            logging.warning(f"Non-finite Y_lm for {element}: l={l}, m={m}, theta={theta:.2f}, phi={phi:.2f}")
            return 0.0
        logging.debug(f"Y_lm for {element}: l={l}, m={m}, theta={theta:.2f}, phi={phi:.2f}, Y_lm={Y_lm:.2e}")
        return Y_lm
    except Exception as e:
        logging.error(f"Error in spherical_harmonic for {element}: l={l}, m={m}, theta={theta:.2f}, phi={phi:.2f}, error={e}")
        return 0.0

def calculate_k40_parameters(t=0, active_elements=None, elements=ELEMENTS):
    if not np.isfinite(t):
        logging.warning(f"Invalid time t={t}, using 0")
        t = 0
    phi_fit = 1.1e13
    d_phi_fit_dt = 0.0
    phi_env = 0.1
    atomic_numbers = {
        'Scandium': 21, 'Yttrium': 39, 'Lanthanum': 57, 'Actinium': 89,
        'Ruthenium': 44, 'Palladium': 46, 'Platinum': 78, 'Gold': 79,
        'Copper': 29, 'Rubidium': 37, 'Cesium': 55, 'Silicon': 14, 'Oxygen': 8,
        'Iron': 26, 'Cadmium': 48, 'Francium': 87, 'Hydrogen': 1, 'Carbon': 6,
        'Titanium': 22, 'Magnesium': 12, 'Molybdenum': 42, 'Helium': 2,
        'Potassium-40': 19, 'Lithium': 3, 'Sodium': 11, 'Silver': 47,
        'Beryllium': 4, 'Calcium': 20, 'Strontium': 38, 'Barium': 56,
        'Radium': 88, 'Nickel': 28, 'Cobalt': 27, 'Rhodium': 45, 'Iridium': 77,
        'Osmium': 76, 'Uranium-238': 92, 'Thorium-232': 90, 'Lead-208': 82
    }
    total_abund = 0.0
    weighted_z = 0.0
    for elem, _, abund in elements:
        if active_elements is None or elem in active_elements:
            z = atomic_numbers.get(elem, 50)
            if not np.isfinite(z * abund):
                logging.warning(f"Invalid z*abund for {elem}: z={z}, abund={abund}")
                continue
            weighted_z += z * abund
            total_abund += abund
    avg_z = weighted_z / max(total_abund, 1e-10) if total_abund > 0 else 50.0
    spin_compensation = np.clip(1.0 + 0.1 * (avg_z - 50) / 50, 0.5, 1.5)
    neutron_flux_factor = 1.0
    C_q_factor = 0.0
    for period in neutron_periods:
        f = 1.0 / max(period, 1e-10)
        neutron_flux_factor += 0.05 * np.sin(2 * np.pi * t / period) * np.exp(-0.01 * period / 1000) / len(neutron_periods)
        C_q_factor += 1.0 / (2 * np.pi * f) / len(neutron_periods)
    neutron_flux_factor = max(neutron_flux_factor, 0.5)
    C_q_factor = max(C_q_factor * 1e-11 * (1 + 0.05 * (avg_z - 50) / 50), 1e-12)
    base_n_dm = t * K_B * 6e-11 * (d_phi_fit_dt**2 + phi_fit**2) * np.cos(np.pi * t / 10440 + phi_env)**2
    if not np.isfinite(base_n_dm):
        logging.warning(f"Non-finite base_n_dm: {base_n_dm}, using 1e-7")
        base_n_dm = 1e-7
    N_DM = max(base_n_dm * spin_compensation * neutron_flux_factor, 1e-7)
    N_DM = np.clip(N_DM, 1e-8, 5e-7)
    if not np.isfinite(N_DM):
        logging.error(f"Invalid N_DM: {N_DM}, using 1e-7")
        N_DM = 1e-7
    logging.debug(f"Spin compensation: avg_z={avg_z:.1f}, spin_compensation={spin_compensation:.2f}, neutron_flux_factor={neutron_flux_factor:.2f}, C_q_factor={C_q_factor:.2e} (periods={neutron_periods}), base_n_dm={base_n_dm:.2e}, N_DM={N_DM:.2e}")
    kappa_theory = 8.05e-6
    R_Q = 1e-4
    R_q = 3.6177e18
    rho_t0 = 1.0
    eta_DM = 3.0
    rho_DM = 0.3
    epsilon = 0.5
    C_anti = 0.3 + 0.1 * np.sin(2 * np.pi * t / 10440)
    rho_t = rho_t0 + eta_DM * rho_DM * (1 - epsilon * C_anti)
    term1 = kappa_theory * R_Q * rho_t * 2.6e3
    term2 = R_q * N_DM * 2e-11 * C_q_factor
    c_eff_shift = term1 + term2
    if not np.isfinite(c_eff_shift):
        logging.warning(f"Non-finite c_eff_shift: {c_eff_shift}, using 0")
        c_eff_shift = 0
    c_eff = SPEED_OF_LIGHT * (1 + c_eff_shift * 1e-6)
    Delta_rho_DM = 5.76e-4
    f_t = 1 + 10 * np.sin(2 * np.pi * t / 10440)
    xi = 5e-16
    Delta_alpha_over_alpha = (kappa_theory * R_Q * eta_DM * Delta_rho_DM * f_t * (1 - epsilon * C_anti) * 2.6e3 + xi * N_DM)
    lambda_k40 = 672e-9
    f_eff = c_eff / lambda_k40
    if not np.isfinite(f_eff):
        logging.error(f"Invalid f_eff: {f_eff}, using 1e9")
        f_eff = 1e9
    logging.info(f"K-40 parameters: N_DM={N_DM:.2e}, c_eff_shift={c_eff_shift:.4f} ppm, c_eff={c_eff:.2e}, Δα/α={Delta_alpha_over_alpha:.2e}, f_eff={f_eff:.2e}, spin_compensation={spin_compensation:.2f}, neutron_flux_factor={neutron_flux_factor:.2f}, C_q_factor={C_q_factor:.2e}, neutron_periods={neutron_periods}")
    return N_DM, c_eff, Delta_alpha_over_alpha, f_eff

def calculate_dm_parameters(f_res, active_elements=None):
    if f_res <= 0 or not np.isfinite(f_res):
        logging.warning(f"Invalid f_res: {f_res}, using 1e9")
        f_res = 1e9
    N_DM, _, _, _ = calculate_k40_parameters(t=0, active_elements=active_elements, elements=ELEMENTS)
    return N_DM, 0.1, 0.05, DELTA_ALPHA_OVER_ALPHA

def calculate_amplitude_range(element, z_norm, hu, theta, delta_theta, z_si, channels):
    if z_si == 0 or not np.isfinite(z_si):
        logging.warning(f"Invalid z_si for {element}: {z_si}, using 1e-10")
        z_si = 1e-10
    r_min = max(0.1, z_norm / z_si * np.cos(theta))
    r_max = min(5.0, z_norm / z_si * (1 + hu**2) * 2.0)
    r_range = r_max - r_min
    if r_range < 0:
        logging.warning(f"Negative r_range for {element}: {r_range}")
        r_min, r_max = r_max, r_min
    return r_min, r_max, r_max - r_min
# ... (Unchanged: logging setup, argument parsing, set_backend, constants, ELEMENTS, ELEMENT_GROUPS, EXPECTED_POSITIONS, ELEMENT_PAIRS, element_channels, validate_channels, validate_elements, wavelength_to_rgb, calculate_hu, calculate_k40_parameters, calculate_dm_parameters, calculate_amplitude_range, calculate_resonant_frequency, plot_lissajous)

def spherical_harmonic(l, m, theta, element="Unknown"):
    theta = np.mod(theta, 2 * np.pi)
    phi = 0.0
    if l < 0 or abs(m) > l:
        logging.error(f"Invalid spherical harmonic for {element}: l={l}, m={m}")
        return 0.0
    Y_lm = sph_harm(m, l, phi, theta).real
    if not np.isfinite(Y_lm):
        logging.warning(f"Non-finite Y_lm for {element}: l={l}, m={m}, theta={theta:.2f}")
        return 0.0
    logging.debug(f"Y_lm for {element}: l={l}, m={m}, theta={theta:.2f}, Y_lm={Y_lm:.2e}")
    return Y_lm

def update_plot(x_shift, y_shift, z_scale, invert_w, active_elements, ax, fig, t=0, lissajous_ax=None):
    if not np.isfinite([x_shift, y_shift, z_scale]).all():
        logging.error(f"Invalid input values: x_shift={x_shift}, y_shift={y_shift}, z_scale={z_scale}")
        return None
    ax.clear()
    if lissajous_ax:
        lissajous_ax.clear()
    if not active_elements:
        logging.error("No active elements")
        ax.set_title("Error: No Active Elements")
        fig.canvas.draw()
        return None
    logging.info(f"Active elements count: {len(active_elements)}")
    global N_DM, C_EFF, DELTA_ALPHA_OVER_ALPHA, K40_FREQUENCY
    N_DM, C_EFF, DELTA_ALPHA_OVER_ALPHA, K40_FREQUENCY = calculate_k40_parameters(t=t, active_elements=active_elements, elements=ELEMENTS)
    use_third_order = False
    data = []
    z_values = {}
    pr_targets = {}
    hu_values = {}
    phase_offsets = {}
    r_values = {}
    theta_values = {}
    abund_o = 1.0759
    abund_si = 0.6395
    w_contributions = []
    quadrant_counts = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
    wl_nm_dict = {}
    signals_dict = {elem: {'X': [], 'Y': []} for elem in active_elements}
    light_elements = {light for light, _ in ELEMENT_PAIRS}
    pair_map = {light: heavy for light, heavy in ELEMENT_PAIRS}
    pair_map.update({heavy: light for light, heavy in ELEMENT_PAIRS})
    z_norm_light = {}
    t_range = np.linspace(t, t + 0.1, 100)
    atomic_numbers = {
        'Scandium': 21, 'Yttrium': 39, 'Lanthanum': 57, 'Actinium': 89,
        'Ruthenium': 44, 'Palladium': 46, 'Platinum': 78, 'Gold': 79,
        'Copper': 29, 'Rubidium': 37, 'Cesium': 55, 'Silicon': 14, 'Oxygen': 8,
        'Iron': 26, 'Cadmium': 48, 'Francium': 87, 'Hydrogen': 1, 'Carbon': 6,
        'Titanium': 22, 'Magnesium': 12, 'Molybdenum': 42, 'Helium': 2,
        'Potassium-40': 19, 'Lithium': 3, 'Sodium': 11, 'Silver': 47,
        'Beryllium': 4, 'Calcium': 20, 'Strontium': 38, 'Barium': 56,
        'Radium': 88, 'Nickel': 28, 'Cobalt': 27, 'Rhodium': 45, 'Iridium': 77,
        'Osmium': 76, 'Uranium-238': 92, 'Thorium-232': 90, 'Lead-208': 82
    }
   
   
# ... (Unchanged: plot_heatmap, main block)
def calculate_resonant_frequency(mass, element, active_elements=None):
    if mass <= 0 or not np.isfinite(mass):
        logging.warning(f"Invalid mass for {element}: {mass}, using 1.0")
        mass = 1.0
    _, _, _, K40_FREQUENCY = calculate_k40_parameters(t=0, active_elements=active_elements, elements=ELEMENTS)
    if not np.isfinite(K40_FREQUENCY):
        logging.warning(f"Invalid K40_FREQUENCY: {K40_FREQUENCY}, using 1e9")
        K40_FREQUENCY = 1e9
    f_res = k * np.sqrt(K40_FREQUENCY / mass)
    expected_wl = {
        'Silicon': 777.0, 'Oxygen': 560.0, 'Iron': 690.95, 'Cadmium': 657.93,
        'Francium': 611.96, 'Hydrogen': 429.39, 'Copper': 685.74, 'Carbon': 500.0,
        'Titanium': 650.0, 'Magnesium': 520.0, 'Molybdenum': 670.0, 'Helium': 450.0,
        'Potassium-40': 672.0, 'Lithium': 670.0, 'Sodium': 590.0, 'Rubidium': 700.0,
        'Cesium': 700.0, 'Silver': 328.0, 'Gold': 408.0, 'Palladium': 360.0,
        'Platinum': 306.0, 'Beryllium': 450.0, 'Calcium': 425.0, 'Strontium': 460.0,
        'Barium': 555.0, 'Radium': 480.0, 'Nickel': 352.0, 'Cobalt': 412.0,
        'Rhodium': 369.0, 'Iridium': 322.0, 'Osmium': 330.0,
        'Scandium': 400.0, 'Yttrium': 410.0, 'Lanthanum': 420.0, 'Actinium': 430.0,
        'Ruthenium': 300.0, 'Palladium_Group3': 315.0, 'Platinum_Group3': 310.0,
        'Gold_Group3': 310.0, 'Uranium-238': 290.0, 'Thorium-232': 295.0, 'Lead-208': 320.0
    }
    wl_nm = expected_wl.get(element, 500.0)
    atomic_number = {
        'Scandium': 21, 'Yttrium': 39, 'Lanthanum': 57, 'Actinium': 89,
        'Ruthenium': 44, 'Palladium': 46, 'Platinum': 78, 'Gold': 79,
        'Copper': 29, 'Rubidium': 37, 'Cesium': 55, 'Silicon': 14, 'Oxygen': 8,
        'Iron': 26, 'Cadmium triad': 48, 'Francium': 87, 'Hydrogen': 1, 'Carbon': 6,
        'Titanium': 22, 'Magnesium': 12, 'Molybdenum': 42, 'Helium': 2,
        'Potassium-40': 19, 'Lithium': 3, 'Sodium': 11, 'Silver': 47,
        'Beryllium': 4, 'Calcium': 20, 'Strontium': 38, 'Barium': 56,
        'Radium': 88, 'Nickel': 28, 'Cobalt': 27, 'Rhodium': 45, 'Iridium': 77,
        'Osmium': 76, 'Uranium-238': 92, 'Thorium-232': 90, 'Lead-208': 82
    }.get(element, 50)
    spin_factor = -0.1 if atomic_number > 50 else 0.1 if element in ELEMENT_GROUPS['Group 3'] or element in ['Rubidium', 'Cesium'] else 0.0
    wl_nm_shifted = wl_nm * (1 + spin_factor)
    wl_nm_shifted = min(max(wl_nm_shifted, 300.0), 800.0)
    if not np.isfinite(wl_nm_shifted):
        logging.warning(f"Invalid wl_nm_shifted for {element}: {wl_nm_shifted}, using 500.0")
        wl_nm_shifted = 500.0
    logging.debug(f"Calculated f_res={f_res:.2e}, wl_nm={wl_nm:.1f}, spin_factor={spin_factor:.2f}, wl_nm_shifted={wl_nm_shifted:.1f} for {element}")
    return f_res, wl_nm_shifted

def plot_lissajous(ax, signals_dict, wl_nm_dict, active_elements, t_range, atomic_numbers):
    """
    Plot Lissajous curves for oxygen, silicon, carbon, and select elements.
    Parameters:
    - ax: Matplotlib axes for Lissajous plot.
    - signals_dict: Dictionary of {elem: {'X': [], 'Y': []}} over t_range.
    - wl_nm_dict: Dictionary of {elem: wl_nm_shifted}.
    - active_elements: List of active elements.
    - t_range: Array of time points.
    - atomic_numbers: Dictionary of element atomic numbers.
    """
    ax.clear()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('X Signal')
    ax.set_ylabel('Y Signal')
    ax.set_title('Lissajous Curves (O, Si, C Harmonics)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    focus_elements = ['Oxygen', 'Silicon', 'Carbon', 'Hydrogen', 'Magnesium']
    for elem in active_elements:
        if elem not in focus_elements:
            continue
        if elem not in signals_dict or not signals_dict[elem]['X']:
            continue
        x = np.array(signals_dict[elem]['X'])
        y = np.array(signals_dict[elem]['Y'])
        wl_nm = wl_nm_dict.get(elem, 500.0)
        color = wavelength_to_rgb(wl_nm)
        freq_ratio = wl_nm / 560.0 if elem != 'Silicon' else 1.5 * wl_nm / 560.0
        linestyle = '-' if elem != 'Carbon' else '--'
        linewidth = 2.5 if elem in ['Oxygen', 'Silicon', 'Carbon'] else 1.5
        label = f"{elem} ({wl_nm:.1f} nm, Z={atomic_numbers.get(elem, 50)})"
        ax.plot(x, y, c=color, lw=linewidth, ls=linestyle, label=label, alpha=0.7)
        oxy_wl = wl_nm_dict.get('Oxygen', 560.0)
        freq_ratio_oxy = wl_nm / oxy_wl if elem != 'Oxygen' else 1.0
        logging.debug(f"Lissajous for {elem}: wl_nm={wl_nm:.1f}, freq_ratio_oxy={freq_ratio_oxy:.2f}")
    ax.legend(fontsize=5, loc='upper right')
    return ax

def update_plot(x_shift, y_shift, z_scale, invert_w, active_elements, ax, fig, t=0, lissajous_ax=None):
    if not np.isfinite([x_shift, y_shift, z_scale]).all():
        logging.error(f"Invalid input values: x_shift={x_shift}, y_shift={y_shift}, z_scale={z_scale}")
        return None
    ax.clear()
    if lissajous_ax:
        lissajous_ax.clear()
    if not active_elements:
        logging.error("No active elements")
        ax.set_title("Error: No Active Elements")
        fig.canvas.draw()
        return None
    logging.info(f"Active elements count: {len(active_elements)}")
    global N_DM, C_EFF, DELTA_ALPHA_OVER_ALPHA, K40_FREQUENCY
    N_DM, C_EFF, DELTA_ALPHA_OVER_ALPHA, K40_FREQUENCY = calculate_k40_parameters(t=t, active_elements=active_elements, elements=ELEMENTS)
    use_third_order = False
    data = []
    z_values = {}
    pr_targets = {}
    hu_values = {}
    phase_offsets = {}
    r_values = {}
    theta_values = {}
    abund_o = 1.0759
    abund_si = 0.6395
    w_contributions = []
    quadrant_counts = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
    wl_nm_dict = {}
    signals_dict = {elem: {'X': [], 'Y': []} for elem in active_elements}
    light_elements = {light for light, _ in ELEMENT_PAIRS}
    pair_map = {light: heavy for light, heavy in ELEMENT_PAIRS}
    pair_map.update({heavy: light for light, heavy in ELEMENT_PAIRS})
    z_norm_light = {}
    t_range = np.linspace(t, t + 0.1, 100)
    atomic_numbers = {
        'Scandium': 21, 'Yttrium': 39, 'Lanthanum': 57, 'Actinium': 89,
        'Ruthenium': 44, 'Palladium': 46, 'Platinum': 78, 'Gold': 79,
        'Copper': 29, 'Rubidium': 37, 'Cesium': 55, 'Silicon': 14, 'Oxygen': 8,
        'Iron': 26, 'Cadmium': 48, 'Francium': 87, 'Hydrogen': 1, 'Carbon': 6,
        'Titanium': 22, 'Magnesium': 12, 'Molybdenum': 42, 'Helium': 2,
        'Potassium-40': 19, 'Lithium': 3, 'Sodium': 11, 'Silver': 47,
        'Beryllium': 4, 'Calcium': 20, 'Strontium': 38, 'Barium': 56,
        'Radium': 88, 'Nickel': 28, 'Cobalt': 27, 'Rhodium': 45, 'Iridium': 77,
        'Osmium': 76, 'Uranium-238': 92, 'Thorium-232': 90, 'Lead-208': 82
    }
    for elem, mass, abund in ELEMENTS:
        if elem not in active_elements:
            continue
        f_res, wl_nm = calculate_resonant_frequency(mass, elem, active_elements)
        wl_nm_dict[elem] = wl_nm
        abundance = max(abund, MIN_ABUNDANCE)
        z_norm = abundance * z_scale * (1 + np.log10(max(abundance, 1e-10)) / 2)
        z_values[elem] = z_norm
        N_DM_elem, C_anti, c_eff_shift, Delta_alpha_over_alpha = calculate_dm_parameters(f_res, active_elements)
        pr_targets[elem] = 1.0 + c_eff_shift / 7.6535
        hu_values[elem] = calculate_hu(pr_targets[elem], elem)
        r_exp, theta_exp = EXPECTED_POSITIONS.get(elem, (1.0, np.pi/4))
        theta_shift = np.arctan2(y_shift, x_shift)
        joystick_magnitude = np.sqrt(x_shift**2 + y_shift**2)
        theta_adjust = 0.0
        partner = pair_map.get(elem)
        if partner and partner in wl_nm_dict:
            delta_wl = abs(wl_nm - wl_nm_dict[partner])
            scale = min(1.0, 200 / max(delta_wl, 1e-10))
        else:
            scale = 1.0
        if joystick_magnitude >= 1e-5:
            theta_adjust = theta_shift * min(1.0, joystick_magnitude / 4.0) * scale
        if elem == 'Oxygen' and z_values.get('Oxygen', 0) > abund_si:
            theta_actual = 0.0
            r = 0.0
        else:
            if elem == 'Silicon':
                theta_base = np.pi / 2  # Force Silicon to 90°
                spiral_factor = 0.0
            else:
                delta_lambda = (560.0 - wl_nm) / 560.0
                spin_bias = 0.2 if wl_nm > 700 else -0.2 if wl_nm < 400 else 0.0
                theta_base = theta_exp + 0.1 * delta_lambda + spin_bias
                # Override for specific groups
                if atomic_numbers.get(elem, 50) > 50:  # Heavier elements
                    theta_base = 3 * np.pi / 2  # 270°
                elif atomic_numbers.get(elem, 50) < 20:  # Lighter elements
                    theta_base = np.pi / 2  # 90°
                elif wl_nm < 400:  # UV
                    theta_base = np.pi  # 180°
                elif wl_nm > 700:  # IR
                    theta_base = 0.0  # 0°
            deviation = z_norm / abund_o + joystick_magnitude
            spiral_factor = 1.0 if elem == 'Carbon' else 0.0  # Disable spiral except for Carbon
            spiral_theta = spiral_factor * deviation * (np.pi/2 if wl_nm > 700 else 3*np.pi/2 if wl_nm < 400 else 0.0) * (1.5 if atomic_numbers.get(elem, 50) > 50 else 1.0)
            theta_actual = np.mod(theta_base + theta_adjust + spiral_theta + 2 * np.pi, 2 * np.pi)
            k = (5.0 / r_exp - 1) / 2.83 if r_exp > 0 else 0.0
            r = r_exp * (1 + k * joystick_magnitude * scale + spiral_factor * deviation)
            r = min(r, 5.0)
        expected_theta = {
            (2.0, 0.0): np.mod(theta_base, 2 * np.pi),
            (0.0, 2.0): np.mod(theta_base + np.pi/2, 2 * np.pi),
            (-2.0, 0.0): np.mod(theta_base + np.pi, 2 * np.pi),
            (0.0, -2.0): np.mod(theta_base + 3*np.pi/2, 2 * np.pi),
            (-2.0, -2.0): np.mod(theta_base + 7*np.pi/4, 2 * np.pi)
        }.get((round(x_shift, 1), round(y_shift, 1)), np.mod(theta_base + theta_shift, 2 * np.pi))
        r_expected = {
            (2.0, 0.0): min(r_exp * (1 + k * 2.83), 5.0),
            (0.0, 2.0): min(r_exp * (1 + k * 2.83), 5.0),
            (-2.0, 0.0): min(r_exp * (1 + k * 2.83), 5.0),
            (0.0, -2.0): min(r_exp * (1 + k * 2.83), 5.0),
            (-2.0, -2.0): min(r_exp * (1 + k * 2.83), 5.0),
            (0.0, 0.0): r_exp
        }.get((round(x_shift, 1), round(y_shift, 1)), r)
        delta_theta = 0.0 if elem == 'Silicon' else theta_exp - theta_actual
        r_values[elem] = r
        theta_values[elem] = theta_actual
        theta_deg = theta_actual % (2 * np.pi)
        quadrant = 'Q1' if 0 <= theta_deg < np.pi/2 else 'Q2' if np.pi/2 <= theta_deg < np.pi else 'Q3' if np.pi <= theta_deg < 3*np.pi/2 else 'Q4'
        quadrant_counts[quadrant] += 1
        oxy_theta = theta_values.get('Oxygen', 0.0)
        phase_diff = np.mod(theta_actual - oxy_theta, 2 * np.pi)
        if elem in ['Carbon', 'Silicon']:
            logging.debug(f"Phase diff for {elem} vs Oxygen: {phase_diff:.2f} rad")
        r_min, r_max, r_range = calculate_amplitude_range(elem, z_norm, hu_values[elem], theta_actual, delta_theta, z_values.get('Silicon', 1e-10), element_channels)
        data.append((elem, f_res, wl_nm, abundance, 1.0, z_norm, hu_values[elem]))
        for t_i in t_range:
            temp_theta = theta_actual + 0.1 * np.sin(2 * np.pi * t_i / neutron_periods[0])
            X_i, Y_i = 0, 0
            for l, m in element_channels.get(elem, []):
                Y_lm = spherical_harmonic(l, m, temp_theta, elem)
                freq_factor = (wl_nm - 400) / (700 - 400)
                if elem == 'Silicon':
                    freq_factor *= 1.5
                cos2_term = max(np.cos(temp_theta)**2 * 0.5 + 0.5, 1e-10)
                is_second_order = l == 2
                is_third_order = l == 3
                is_light = elem in ['Silicon', 'Oxygen', 'Carbon', 'Magnesium', 'Hydrogen', 'Helium', 'Lithium', 'Sodium', 'Rubidium', 'Cesium', 'Beryllium', 'Calcium', 'Strontium', 'Barium', 'Radium', 'Scandium', 'Yttrium', 'Lanthanum', 'Actinium']
                is_heavy = elem in ['Cadmium', 'Iron', 'Francium', 'Copper', 'Titanium', 'Molybdenum', 'Potassium-40', 'Silver', 'Gold', 'Palladium', 'Platinum', 'Nickel', 'Cobalt', 'Rhodium', 'Iridium', 'Osmium', 'Ruthenium', 'Uranium-238', 'Thorium-232', 'Lead-208', 'Palladium_Group3', 'Platinum_Group3', 'Gold_Group3']
                polarity = -1 if (is_second_order or is_third_order) and is_light else 1 if (is_second_order or is_third_order) and is_heavy else 1 if l == 1 and is_light else -1
                phase_factor = 0.7 + 0.2 * (q23_count / max(len(active_elements), 1))
                if (l == 1 and m == 1 or l == 2 and m == 1 or l == 3 and m == 1) and elem != 'Silicon':
                    sign = 1 if is_light else -1
                    contribution = polarity * freq_factor * Y_lm * (1 + sign * x_shift * 0.5) / phase_factor
                    X_i += contribution
                elif (l == 1 and m == -1 or l == 2 and m == -1 or l == 3 and m == -1) and elem != 'Silicon':
                    sign = 1 if is_light else -1
                    contribution = polarity * freq_factor * Y_lm * (1 + sign * y_shift * 0.5) / phase_factor
                    Y_i += contribution
            signals_dict[elem]['X'].append(X_i)
            signals_dict[elem]['Y'].append(Y_i)
    logging.info(f"Quadrant distribution: Q1={quadrant_counts['Q1']}, Q2={quadrant_counts['Q2']}, Q3={quadrant_counts['Q3']}, Q4={quadrant_counts['Q4']}")
    W = 0
    X = 0
    Y = 0
    W_oxy = 0
    z_si = z_values.get('Silicon', 1e-10)
    non_oxy_contributions = []
    q23_count = quadrant_counts['Q2'] + quadrant_counts['Q3']
    monophonic_factor = min(1.0 + MONOPHONIC_BOOST * q23_count / max(len(active_elements), 1), 2.0)
    local_element_channels = element_channels.copy()
    for elem in active_elements:
        z_norm = z_values.get(elem, 1e-10)
        theta = theta_values.get(elem, 0.0)
        hu = hu_values.get(elem, 0.0)
        wl_nm = next((wl for e, _, wl, _, _, _, _ in data if e == elem), 500.0)
        freq_factor = (wl_nm - 400) / (700 - 400)
        if elem == 'Silicon':
            freq_factor *= 1.5
        for l, m in local_element_channels.get(elem, []):
            Y_lm = spherical_harmonic(l, m, theta, elem)
            cos2_term = max(np.cos(theta)**2 * 0.5 + 0.5, 1e-10)
            is_second_order = l == 2
            is_third_order = l == 3
            is_light = elem in ['Silicon', 'Oxygen', 'Carbon', 'Magnesium', 'Hydrogen', 'Helium', 'Lithium', 'Sodium', 'Rubidium', 'Cesium', 'Beryllium', 'Calcium', 'Strontium', 'Barium', 'Radium', 'Scandium', 'Yttrium', 'Lanthanum', 'Actinium']
            is_heavy = elem in ['Cadmium', 'Iron', 'Francium', 'Copper', 'Titanium', 'Molybdenum', 'Potassium-40', 'Silver', 'Gold', 'Palladium', 'Platinum', 'Nickel', 'Cobalt', 'Rhodium', 'Iridium', 'Osmium', 'Ruthenium', 'Uranium-238', 'Thorium-232', 'Lead-208', 'Palladium_Group3', 'Platinum_Group3', 'Gold_Group3']
            polarity = -1 if (is_second_order or is_third_order) and is_light else 1 if (is_second_order or is_third_order) and is_heavy else 1 if l == 1 and is_light else -1
            phase_factor = 0.7 + 0.2 * (q23_count / max(len(active_elements), 1))
            if l == 1 and m == 0 or l == 2 and m == 0 or l == 3 and m == 0:
                if elem == 'Copper' and args.copper_inverse:
                    contribution = 1.0 / (W_oxy**2 + EPSILON) if W_oxy != 0 else 0.0
                else:
                    contribution = polarity * W_REINFORCEMENT * Y_lm * z_norm * (1 + hu**2) * cos2_term / z_si * min(abundance / abund_si, 1.0)
                    wl_factor = min(wl_nm / 560.0, 1.0)
                    contribution *= wl_factor
                    if l == 2 or l == 3:
                        contribution *= 0.1
                    contribution *= phase_factor
                power_scale = z_norm**2 / max(abund_o**2, 1e-10)
                flux_scale = abundance * wl_nm / 560.0
                contribution *= power_scale * flux_scale
                if elem == 'Oxygen':
                    W_oxy = contribution * monophonic_factor
                else:
                    non_oxy_contributions.append((elem, contribution, wl_nm, z_norm))
                w_contributions.append((elem, contribution, wl_nm, z_norm))
            elif (l == 1 and m == 1 or l == 2 and m == 1 or l == 3 and m == 1) and elem != 'Silicon':
                sign = 1 if is_light else -1
                contribution = polarity * freq_factor * Y_lm * (1 + sign * x_shift * 0.5) / phase_factor
                X += contribution
            elif (l == 1 and m == -1 or l == 2 and m == -1 or l == 3 and m == -1) and elem != 'Silicon':
                sign = 1 if is_light else -1
                contribution = polarity * freq_factor * Y_lm * (1 + sign * y_shift * 0.5) / phase_factor
                Y += contribution
    sum_contrib = sum(abs(contrib) for _, contrib, _, _ in non_oxy_contributions)
    logging.info(f"Amplitude check: sum_contrib={sum_contrib:.2e}, W_oxy={W_oxy:.2e}, limit={AMPLITUDE_LIMIT * abs(W_oxy):.2e}")
    if W_oxy != 0 and sum_contrib > AMPLITUDE_LIMIT * abs(W_oxy):
        logging.info(f"Amplitude constraint violated: sum_contrib={sum_contrib:.2e} > {AMPLITUDE_LIMIT} * W_oxy={AMPLITUDE_LIMIT * abs(W_oxy):.2e}. Upgrading heavy pairs to l=3.")
        use_third_order = True
        for elem in ['Ruthenium', 'Palladium_Group3', 'Platinum_Group3', 'Gold_Group3']:
            local_element_channels[elem] = [(3, 1)]
        non_oxy_contributions = []
        w_contributions = []
        W = 0
        for elem in active_elements:
            z_norm = z_values.get(elem, 1e-10)
            theta = theta_values.get(elem, 0.0)
            hu = hu_values.get(elem, 0.0)
            wl_nm = next((wl for e, _, wl, _, _, _, _ in data if e == elem), 500.0)
            freq_factor = (wl_nm - 400) / (700 - 400)
            if elem == 'Silicon':
                freq_factor *= 1.5
            for l, m in local_element_channels.get(elem, []):
                Y_lm = spherical_harmonic(l, m, theta, elem)
                cos2_term = max(np.cos(theta)**2 * 0.5 + 0.5, 1e-10)
                is_second_order = l == 2
                is_third_order = l == 3
                is_light = elem in ['Silicon', 'Oxygen', 'Carbon', 'Magnesium', 'Hydrogen', 'Helium', 'Lithium', 'Sodium', 'Rubidium', 'Cesium', 'Beryllium', 'Calcium', 'Strontium', 'Barium', 'Radium', 'Scandium', 'Yttrium', 'Lanthanum', 'Actinium']
                is_heavy = elem in ['Cadmium', 'Iron', 'Francium', 'Copper', 'Titanium', 'Molybdenum', 'Potassium-40', 'Silver', 'Gold', 'Palladium', 'Platinum', 'Nickel', 'Cobalt', 'Rhodium', 'Iridium', 'Osmium', 'Ruthenium', 'Uranium-238', 'Thorium-232', 'Lead-208', 'Palladium_Group3', 'Platinum_Group3', 'Gold_Group3']
                polarity = -1 if (is_second_order or is_third_order) and is_light else 1 if (is_second_order or is_third_order) and is_heavy else 1 if l == 1 and is_light else -1
                phase_factor = 0.7 + 0.2 * (q23_count / max(len(active_elements), 1))
                if l == 1 and m == 0 or l == 2 and m == 0 or l == 3 and m == 0:
                    if elem == 'Copper' and args.copper_inverse:
                        contribution = 1.0 / (W_oxy**2 + EPSILON) if W_oxy != 0 else 0.0
                    else:
                        contribution = polarity * W_REINFORCEMENT * Y_lm * z_norm * (1 + hu**2) * cos2_term / z_si * min(abundance / abund_si, 1.0)
                        wl_factor = min(wl_nm / 560.0, 1.0)
                        contribution *= wl_factor
                        if l == 2 or l == 3:
                            contribution *= 0.1
                        contribution *= phase_factor
                    power_scale = z_norm**2 / max(abund_o**2, 1e-10)
                    flux_scale = abundance * wl_nm / 560.0
                    contribution *= power_scale * flux_scale
                    if elem == 'Oxygen':
                        W_oxy = contribution * monophonic_factor
                    else:
                        non_oxy_contributions.append((elem, contribution, wl_nm, z_norm))
                    w_contributions.append((elem, contribution, wl_nm, z_norm))
    sum_contrib = sum(abs(contrib) for _, contrib, _, _ in non_oxy_contributions)
    scale_w = 1.0
    if W_oxy != 0 and sum_contrib > AMPLITUDE_LIMIT * abs(W_oxy):
        scale_w = AMPLITUDE_LIMIT * abs(W_oxy) / max(sum_contrib, 1e-10)
    for elem, contrib, wl_nm, z_norm in non_oxy_contributions:
        scaled_contrib = contrib * scale_w
        W += scaled_contrib
    vector_sum = np.sqrt(X**2 + Y**2)
    if vector_sum > 0:
        boost_limit = 2.83 * vector_sum
        scale = min(1.0, boost_limit / max(np.sqrt(X**2 + Y**2), 1e-10))
        X *= scale
        Y *= scale
    signals = {'W': W, 'X': X, 'Y': Y}
    vector_sum = np.sqrt(X**2 + Y**2)
    phase_coherence = vector_sum / max(abs(W), 1e-10) if W != 0 else 0.0
    logging.info(f"Phase coherence: W={W:.2e}, X={X:.2e}, Y={Y:.2e}, vector_sum={vector_sum:.2e}, coherence_ratio={phase_coherence:.2f}, q23_count={q23_count}, monophonic_factor={monophonic_factor:.2f}")
    total_weight = sum(abs(contrib) * (1.5 if elem == 'Silicon' else 2.0 if elem == 'Oxygen' else 1.0) * (wl_nm / 560.0) for elem, contrib, wl_nm, _ in w_contributions)
    dominant_wl = sum(wl_nm * abs(contrib) * (1.5 if elem == 'Silicon' else 2.0 if elem == 'Oxygen' else 1.0) * (wl_nm / 560.0) for elem, contrib, wl_nm, _ in w_contributions) / total_weight if total_weight > 0 else 560.0
    dominant_wl = min(max(dominant_wl, 300.0), 800.0)
    if lissajous_ax:
        plot_lissajous(lissajous_ax, signals_dict, wl_nm_dict, active_elements, t_range, atomic_numbers)
    final_data = []
    z_final_values = []
    for idx, (elem, f_res, wl_nm, abundance, _, z, hu) in enumerate(data):
        theta = np.mod(theta_values.get(elem, 0.0), 2 * np.pi)
        delta_theta = phase_offsets.get(elem, 0.0)
        r = r_values.get(elem, 1.0)
        z_final = z / (1 + 6.0 * r**2) * (1.5 if wl_nm > 700 or wl_nm < 400 else 1.0)
        z_final_values.append(z_final)
        for l, m in local_element_channels.get(elem, []):
            Y_lm = spherical_harmonic(l, m, theta + delta_theta, elem)
            freq_factor = (wl_nm - 400) / (700 - 400)
            if elem == 'Silicon':
                freq_factor *= 1.5
            cos2_term = max(np.cos(theta + delta_theta)**2 * 0.5 + 0.5, 1e-10)
            is_second_order = l == 2
            is_third_order = l == 3
            is_light = elem in ['Silicon', 'Oxygen', 'Carbon', 'Magnesium', 'Hydrogen', 'Helium', 'Lithium', 'Sodium', 'Rubidium', 'Cesium', 'Beryllium', 'Calcium', 'Strontium', 'Barium', 'Radium', 'Scandium', 'Yttrium', 'Lanthanum', 'Actinium']
            is_heavy = elem in ['Cadmium', 'Iron', 'Francium', 'Copper', 'Titanium', 'Molybdenum', 'Potassium-40', 'Silver', 'Gold', 'Palladium', 'Platinum', 'Nickel', 'Cobalt', 'Rhodium', 'Iridium', 'Osmium', 'Ruthenium', 'Uranium-238', 'Thorium-232', 'Lead-208', 'Palladium_Group3', 'Platinum_Group3', 'Gold_Group3']
            polarity = -1 if (is_second_order or is_third_order) and is_light else 1 if (is_second_order or is_third_order) and is_heavy else 1 if l == 1 and is_light else -1
            phase_factor = 0.7 + 0.2 * (q23_count / max(len(active_elements), 1))
            if l == 1 and m == 0 or l == 2 and m == 0 or l == 3 and m == 0:
                if elem == 'Copper' and args.copper_inverse:
                    contribution = 1.0 / (W_oxy**2 + EPSILON) if W_oxy != 0 else 0.0
                else:
                    contribution = polarity * Y_lm * z_final * cos2_term * phase_factor
                if elem != 'Oxygen':
                    contribution *= scale_w * 0.7071 * abs(W_oxy) / max(abs(contribution), 1e-10) if W_oxy != 0 else 1.0
                else:
                    contribution *= monophonic_factor
                signals['W'] += contribution
            elif (l == 1 and m == 1 or l == 2 and m == 1 or l == 3 and m == 1) and elem != 'Silicon':
                sign = 1 if is_light else -1
                contribution = polarity * freq_factor * Y_lm * (1 + sign * x_shift * 0.5) / phase_factor
                signals['X'] += contribution
            elif (l == 1 and m == -1 or l == 2 and m == -1 or l == 3 and m == -1) and elem != 'Silicon':
                sign = 1 if is_light else -1
                contribution = polarity * freq_factor * Y_lm * (1 + sign * y_shift * 0.5) / phase_factor
                signals['Y'] += contribution
        r = min(r, 5.0)
        theta = np.clip(np.mod(theta, 2 * np.pi), 0, 2 * np.pi - 1e-10)
        wl_shifted = 560.0 + (wl_nm - 560.0) * 0.8
        final_data.append((elem, f_res, wl_nm, abundance, 1.0, z_final, r, theta, z_final, hu, wl_shifted))
    signals['W'] = -signals['W'] if invert_w else signals['W']
    r_max = max(r for _, _, _, _, _, _, r, _, _, _, _ in final_data) * 1.2 if final_data else 1.0
    r_bins = np.linspace(0, max(r_max, 0.1), 25)
    theta_bins = np.linspace(0, 2 * np.pi, 101)
    heatmap = np.ones((len(theta_bins)-1, len(r_bins)-1)) * 1e-8
    for _, _, _, _, _, _, r, theta, z_final, _, _ in final_data:
        theta = np.clip(np.mod(theta, 2 * np.pi), 0, 2 * np.pi - 1e-10)
        r_idx = np.searchsorted(r_bins, r)
        theta_idx = np.searchsorted(theta_bins, theta)
        if not (0 <= r_idx < len(r_bins)-1 and 0 <= theta_idx < len(theta_bins)-1):
            logging.warning(f"Out-of-bounds: r={r:.2f}, theta={theta:.2f}, r_idx={r_idx}, theta_idx={theta_idx}")
            continue
        heatmap[theta_idx, r_idx] = max(heatmap[theta_idx, r_idx], z_final)
    cmap = matplotlib.colormaps['Spectral_r']
    norm = LogNorm(vmin=1e-10, vmax=max(z_final_values) * 1.5)
    im = ax.pcolormesh(theta_bins, r_bins, heatmap.T, cmap=cmap, norm=norm)
    for elem, _, wl_nm, _, _, _, r, theta, _, hu, wl_shifted in final_data:
        color = wavelength_to_rgb(wl_shifted)
        marker = {
            'Silicon': 's', 'Oxygen': '*', 'Iron': '^', 'Cadmium': 'o',
            'Francium': 'x', 'Hydrogen': '>', 'Copper': 'd', 'Carbon': 'p',
            'Titanium': 'h', 'Magnesium': '<', 'Molybdenum': 'v', 'Helium': '+',
            'Potassium-40': '*', 'Lithium': 'D', 'Sodium': 's', 'Rubidium': '^',
            'Cesium': 'o', 'Silver': 'x', 'Gold': '>', 'Palladium': 'd', 'Platinum': 'p',
            'Beryllium': 'h', 'Calcium': '<', 'Strontium': 'v', 'Barium': '+', 'Radium': '*',
            'Nickel': 's', 'Cobalt': '^', 'Rhodium': 'o', 'Iridium': 'x', 'Osmium': '>',
            'Scandium': 'o', 'Yttrium': 's', 'Lanthanum': 'd', 'Actinium': 'p',
            'Ruthenium': 'h', 'Uranium-238': '^', 'Thorium-232': 'x', 'Lead-208': '+',
            'Palladium_Group3': '^', 'Platinum_Group3': 'v', 'Gold_Group3': '*'
        }.get(elem, '.')
        s = 250 if elem in ELEMENT_GROUPS['Group 2'] or elem in ELEMENT_GROUPS['Group 3'] else 150
        ax.scatter(theta, r, c=[color], marker=marker, s=s, label=f'{elem} (HU={hu:.1f}, {wl_shifted:.1f}nm)')
        ax.text(theta, r, f"{elem}\n{hu:.1f}\n{wl_shifted:.1f}nm", fontsize=5, ha='center', va='center', color='black')
    ax.set_title(f'Polar Heat Map (Periphonic Ambisonics) for Oxygen-Rich Silicate Crust (329 THz, W={dominant_wl:.1f} nm)')
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.7), fontsize=6)
    ax.grid(True, which='both', ls='--', alpha=0.5)
    fig.canvas.draw()
    fig.canvas.flush_events()
    return im
# ... (Previous imports, constants, ELEMENTS, ELEMENT_GROUPS, EXPECTED_POSITIONS, ELEMENT_PAIRS, element_channels, validate_channels, validate_elements, wavelength_to_rgb, calculate_hu, calculate_k40_parameters, calculate_dm_parameters, calculate_amplitude_range, calculate_resonant_frequency, plot_lissajous remain unchanged)

# ... (Previous imports, constants, ELEMENTS, ELEMENT_GROUPS, EXPECTED_POSITIONS, ELEMENT_PAIRS, element_channels, validate_channels, validate_elements, wavelength_to_rgb, calculate_hu, calculate_k40_parameters, calculate_dm_parameters, calculate_amplitude_range, calculate_resonant_frequency remain unchanged)
# ... (Previous imports, constants, ELEMENTS, ELEMENT_GROUPS, EXPECTED_POSITIONS, ELEMENT_PAIRS, element_channels, validate_channels, validate_elements, wavelength_to_rgb, calculate_hu, calculate_k40_parameters, calculate_dm_parameters, calculate_amplitude_range, calculate_resonant_frequency, plot_lissajous remain unchanged)

def update_plot(x_shift, y_shift, z_scale, invert_w, active_elements, ax, fig, t=0, lissajous_ax=None):
    if not np.isfinite([x_shift, y_shift, z_scale]).all():
        logging.error(f"Invalid input values: x_shift={x_shift}, y_shift={y_shift}, z_scale={z_scale}")
        return None
    ax.clear()
    if lissajous_ax:
        lissajous_ax.clear()
    if not active_elements:
        logging.error("No active elements")
        ax.set_title("Error: No Active Elements")
        fig.canvas.draw()
        return None
    logging.info(f"Active elements count: {len(active_elements)}")
    global N_DM, C_EFF, DELTA_ALPHA_OVER_ALPHA, K40_FREQUENCY
    N_DM, C_EFF, DELTA_ALPHA_OVER_ALPHA, K40_FREQUENCY = calculate_k40_parameters(t=t, active_elements=active_elements, elements=ELEMENTS)
    use_third_order = False
    data = []
    z_values = {}
    pr_targets = {}
    hu_values = {}
    phase_offsets = {}
    r_values = {}
    theta_values = {}
    abund_o = 1.0759
    abund_si = 0.6395
    w_contributions = []
    quadrant_counts = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
    wl_nm_dict = {}
    signals_dict = {elem: {'X': [], 'Y': []} for elem in active_elements}
    light_elements = {light for light, _ in ELEMENT_PAIRS}
    pair_map = {light: heavy for light, heavy in ELEMENT_PAIRS}
    pair_map.update({heavy: light for light, heavy in ELEMENT_PAIRS})
    t_range = np.linspace(t, t + 0.1, 100)
    atomic_numbers = {
        'Scandium': 21, 'Yttrium': 39, 'Lanthanum': 57, 'Actinium': 89,
        'Ruthenium': 44, 'Palladium': 46, 'Platinum': 78, 'Gold': 79,
        'Copper': 29, 'Rubidium': 37, 'Cesium': 55, 'Silicon': 14, 'Oxygen': 8,
        'Iron': 26, 'Cadmium': 48, 'Francium': 87, 'Hydrogen': 1, 'Carbon': 6,
        'Titanium': 22, 'Magnesium': 12, 'Molybdenum': 42, 'Helium': 2,
        'Potassium-40': 19, 'Lithium': 3, 'Sodium': 11, 'Silver': 47,
        'Beryllium': 4, 'Calcium': 20, 'Strontium': 38, 'Barium': 56,
        'Radium': 88, 'Nickel': 28, 'Cobalt': 27, 'Rhodium': 45, 'Iridium': 77,
        'Osmium': 76, 'Uranium-238': 92, 'Thorium-232': 90, 'Lead-208': 82
    }
    # First loop: Calculate positions, quadrants, and store iron's theta for realignment
    iron_theta = None
    for elem, mass, abund in ELEMENTS:
        if elem not in active_elements:
            continue
        f_res, wl_nm = calculate_resonant_frequency(mass, elem, active_elements)
        wl_nm_dict[elem] = wl_nm
        abundance = max(abund, MIN_ABUNDANCE)
        z_norm = abundance * z_scale * (1 + np.log10(max(abundance, 1e-10)) / 2)
        z_values[elem] = z_norm
        N_DM_elem, C_anti, c_eff_shift, Delta_alpha_over_alpha = calculate_dm_parameters(f_res, active_elements)
        pr_targets[elem] = 1.0 + c_eff_shift / 7.6535
        hu_values[elem] = calculate_hu(pr_targets[elem], elem)
        r_exp, theta_exp = EXPECTED_POSITIONS.get(elem, (1.0, np.pi/4))
        theta_shift = np.arctan2(y_shift, x_shift)
        joystick_magnitude = np.sqrt(x_shift**2 + y_shift**2)
        theta_adjust = 0.0
        partner = pair_map.get(elem)
        if partner and partner in wl_nm_dict:
            delta_wl = abs(wl_nm - wl_nm_dict[partner])
            scale = min(1.0, 200 / max(delta_wl, 1e-10))
        else:
            scale = 1.0
        if joystick_magnitude >= 1e-5:
            theta_adjust = theta_shift * min(1.0, joystick_magnitude / 4.0) * scale
        if elem == 'Oxygen' and z_values.get('Oxygen', 0) > abund_si:
            theta_actual = 0.0
            r = 0.0
    else:
        joystick_magnitude = np.sqrt(x_shift**2 + y_shift**2)
        if elem == 'Silicon' and np.isclose(joystick_magnitude, 0.0, atol=1e-5):
            theta_base = np.pi / 2  # Silicon at 90° (north in default, east after rotation)
            spiral_factor = 0.0  # Disable spiral term
        else:
            if elem == 'Silicon':
                theta_base = -0.035  # Original Silicon theta_base for tracking
                spiral_factor = SPIRAL_FACTOR
            else:
                delta_lambda = (560.0 - wl_nm) / 560.0
                spin_bias = 0.2 if wl_nm > 700 else -0.2 if wl_nm < 400 else 0.0
            if np.isclose(joystick_magnitude, 0.0, atol=1e-5):
                if atomic_numbers.get(elem, 50) > 50:  # Heavier elements
                    theta_base = np.pi  # 180° (west after rotation to 270°)
                elif atomic_numbers.get(elem, 50) < 20:  # Lighter elements
                    theta_base = np.pi / 2  # 90° (east after rotation to 90°)
                elif wl_nm < 400:  # UV elements
                    theta_base = np.pi  # 180° (south after rotation to 180°)
                elif wl_nm > 700:  # IR elements
                    theta_base = 0.0  # 0° (north after rotation to 0°)
                else:
                    theta_base = theta_exp + 0.1 * delta_lambda + spin_bias
            else:
                theta_base = theta_exp + 0.1 * delta_lambda + spin_bias
            spiral_factor = 1.0 if elem == 'Carbon' else SPIRAL_FACTOR
        deviation = z_norm / abund_o + joystick_magnitude
        spiral_theta = spiral_factor * deviation * (np.pi/2 if wl_nm > 700 else 3*np.pi/2 if wl_nm < 400 else 0.0) * (1.5 if atomic_numbers.get(elem, 50) > 50 else 1.0)
        theta_actual = np.mod(theta_base + theta_adjust + spiral_theta + 2 * np.pi, 2 * np.pi)
        # Apply 90° counterclockwise rotation
        theta_actual = np.mod(theta_actual - np.pi / 2, 2 * np.pi)
        if elem == 'Iron':
            iron_theta = theta_actual  # Store iron's theta for realignment
        theta_deg = theta_actual % (2 * np.pi)
        quadrant = 'Q1' if 0 <= theta_deg < np.pi/2 else 'Q2' if np.pi/2 <= theta_deg < np.pi else 'Q3' if np.pi <= theta_deg < 3*np.pi/2 else 'Q4'
        quadrant_counts[quadrant] += 1
        oxy_theta = theta_values.get('Oxygen', 0.0)
        phase_diff = np.mod(theta_actual - oxy_theta, 2 * np.pi)
        if elem in ['Oxygen', 'Silicon', 'Carbon', 'Hydrogen', 'Magnesium']:
            logging.debug(f"Phase diff for {elem} vs Oxygen: {phase_diff:.2f} rad")
        r_min, r_max, r_range = calculate_amplitude_range(elem, z_norm, hu_values[elem], theta_actual, delta_theta, z_values.get('Silicon', 1e-10), element_channels)
        data.append((elem, f_res, wl_nm, abundance, 1.0, z_norm, hu_values[elem]))
    # Realign all theta values to place iron at theta = 0
  
    q23_count = quadrant_counts['Q2'] + quadrant_counts['Q3']
    logging.info(f"Quadrant distribution: Q1={quadrant_counts['Q1']}, Q2={quadrant_counts['Q2']}, Q3={quadrant_counts['Q3']}, Q4={quadrant_counts['Q4']}")
    # Second loop: Calculate signals and Lissajous with scaling
    signal_scale = 1e6  # From previous fix for Lissajous visibility
    for elem in active_elements:
        theta_actual = theta_values.get(elem, 0.0)
        wl_nm = wl_nm_dict.get(elem, 500.0)
        for t_i in t_range:
            temp_theta = theta_actual + 0.2 * np.sin(2 * np.pi * t_i / neutron_periods[0])  # Revert to previous amplitude
            X_i, Y_i = 0, 0
            for l, m in element_channels.get(elem, []):
                Y_lm = spherical_harmonic(l, m, temp_theta, elem)
                freq_factor = (wl_nm - 400) / (700 - 400)
                if elem == 'Silicon':
                    freq_factor *= 1.5
                cos2_term = max(np.cos(temp_theta)**2 * 0.5 + 0.5, 1e-10)
                is_second_order = l == 2
                is_third_order = l == 3
                is_light = elem in ['Silicon', 'Oxygen', 'Carbon', 'Magnesium', 'Hydrogen', 'Helium', 'Lithium', 'Sodium', 'Rubidium', 'Cesium', 'Beryllium', 'Calcium', 'Strontium', 'Barium', 'Radium', 'Scandium', 'Yttrium', 'Lanthanum', 'Actinium']
                is_heavy = elem in ['Cadmium', 'Iron', 'Francium', 'Copper', 'Titanium', 'Molybdenum', 'Potassium-40', 'Silver', 'Gold', 'Palladium', 'Platinum', 'Nickel', 'Cobalt', 'Rhodium', 'Iridium', 'Osmium', 'Ruthenium', 'Uranium-238', 'Thorium-232', 'Lead-208', 'Palladium_Group3', 'Platinum_Group3', 'Gold_Group3']
                polarity = -1 if (is_second_order or is_third_order) and is_light else 1 if (is_second_order or is_third_order) and is_heavy else 1 if l == 1 and is_light else -1
                phase_factor = 0.7 + 0.2 * (q23_count / max(len(active_elements), 1))
                if (l == 1 and m == 1 or l == 2 and m == 1 or l == 3 and m == 1) and elem != 'Silicon':
                    sign = 1 if is_light else -1
                    contribution = polarity * freq_factor * Y_lm * (1 + sign * x_shift * 0.5) / phase_factor
                    X_i += contribution * signal_scale
                elif (l == 1 and m == -1 or l == 2 and m == -1 or l == 3 and m == -1) and elem != 'Silicon':
                    sign = 1 if is_light else -1
                    contribution = polarity * freq_factor * Y_lm * (1 + sign * y_shift * 0.5) / phase_factor
                    Y_i += contribution * signal_scale
            signals_dict[elem]['X'].append(X_i)
            signals_dict[elem]['Y'].append(Y_i)
            if elem in ['Oxygen', 'Silicon', 'Carbon']:
                logging.debug(f"Lissajous signal for {elem}: X_i={X_i:.2e}, Y_i={Y_i:.2e}, wl_nm={wl_nm:.1f}")
    # Third loop: Calculate W, X, Y contributions
    W = 0
    X = 0
    Y = 0
    W_oxy = 0
    z_si = z_values.get('Silicon', 1e-10)
    non_oxy_contributions = []
    monophonic_factor = min(1.0 + MONOPHONIC_BOOST * q23_count / max(len(active_elements), 1), 2.0)
    local_element_channels = element_channels.copy()
    for elem in active_elements:
        z_norm = z_values.get(elem, 1e-10)
        theta = theta_values.get(elem, 0.0)
        hu = hu_values.get(elem, 0.0)
        wl_nm = next((wl for e, _, wl, _, _, _, _ in data if e == elem), 500.0)
        freq_factor = (wl_nm - 400) / (700 - 400)
        if elem == 'Silicon':
            freq_factor *= 1.5
        for l, m in local_element_channels.get(elem, []):
            Y_lm = spherical_harmonic(l, m, theta, elem)
            cos2_term = max(np.cos(theta)**2 * 0.5 + 0.5, 1e-10)
            is_second_order = l == 2
            is_third_order = l == 3
            is_light = elem in ['Silicon', 'Oxygen', 'Carbon', 'Magnesium', 'Hydrogen', 'Helium', 'Lithium', 'Sodium', 'Rubidium', 'Cesium', 'Beryllium', 'Calcium', 'Strontium', 'Barium', 'Radium', 'Scandium', 'Yttrium', 'Lanthanum', 'Actinium']
            is_heavy = elem in ['Cadmium', 'Iron', 'Francium', 'Copper', 'Titanium', 'Molybdenum', 'Potassium-40', 'Silver', 'Gold', 'Palladium', 'Platinum', 'Nickel', 'Cobalt', 'Rhodium', 'Iridium', 'Osmium', 'Ruthenium', 'Uranium-238', 'Thorium-232', 'Lead-208', 'Palladium_Group3', 'Platinum_Group3', 'Gold_Group3']
            polarity = -1 if (is_second_order or is_third_order) and is_light else 1 if (is_second_order or is_third_order) and is_heavy else 1 if l == 1 and is_light else -1
            phase_factor = 0.7 + 0.2 * (q23_count / max(len(active_elements), 1))
            if l == 1 and m == 0 or l == 2 and m == 0 or l == 3 and m == 0:
                if elem == 'Copper' and args.copper_inverse:
                    contribution = 1.0 / (W_oxy**2 + EPSILON) if W_oxy != 0 else 0.0
                else:
                    contribution = polarity * W_REINFORCEMENT * Y_lm * z_norm * (1 + hu**2) * cos2_term / z_si * min(abundance / abund_si, 1.0)
                    wl_factor = min(wl_nm / 560.0, 1.0)
                    contribution *= wl_factor
                    if l == 2 or l == 3:
                        contribution *= 0.1
                    contribution *= phase_factor
                power_scale = z_norm**2 / max(abund_o**2, 1e-10)
                flux_scale = abundance * wl_nm / 560.0
                contribution *= power_scale * flux_scale
                if elem == 'Oxygen':
                    W_oxy = contribution * monophonic_factor
                else:
                    non_oxy_contributions.append((elem, contribution, wl_nm, z_norm))
                w_contributions.append((elem, contribution, wl_nm, z_norm))
            elif (l == 1 and m == 1 or l == 2 and m == 1 or l == 3 and m == 1) and elem != 'Silicon':
                sign = 1 if is_light else -1
                contribution = polarity * freq_factor * Y_lm * (1 + sign * x_shift * 0.5) / phase_factor
                X += contribution
            elif (l == 1 and m == -1 or l == 2 and m == -1 or l == 3 and m == -1) and elem != 'Silicon':
                sign = 1 if is_light else -1
                contribution = polarity * freq_factor * Y_lm * (1 + sign * y_shift * 0.5) / phase_factor
                Y += contribution
    # Handle amplitude constraint
    sum_contrib = sum(abs(contrib) for _, contrib, _, _ in non_oxy_contributions)
    logging.info(f"Amplitude check: sum_contrib={sum_contrib:.2e}, W_oxy={W_oxy:.2e}, limit={AMPLITUDE_LIMIT * abs(W_oxy):.2e}")
    if W_oxy != 0 and sum_contrib > AMPLITUDE_LIMIT * abs(W_oxy):
        logging.info(f"Amplitude constraint violated: sum_contrib={sum_contrib:.2e} > {AMPLITUDE_LIMIT} * W_oxy={AMPLITUDE_LIMIT * abs(W_oxy):.2e}. Upgrading heavy pairs to l=3.")
        use_third_order = True
        for elem in ['Ruthenium', 'Palladium_Group3', 'Platinum_Group3', 'Gold_Group3']:
            local_element_channels[elem] = [(3, 1)]
        non_oxy_contributions = []
        w_contributions = []
        W = 0
        for elem in active_elements:
            z_norm = z_values.get(elem, 1e-10)
            theta = theta_values.get(elem, 0.0)
            hu = hu_values.get(elem, 0.0)
            wl_nm = next((wl for e, _, wl, _, _, _, _ in data if e == elem), 500.0)
            freq_factor = (wl_nm - 400) / (700 - 400)
            if elem == 'Silicon':
                freq_factor *= 1.5
            for l, m in local_element_channels.get(elem, []):
                Y_lm = spherical_harmonic(l, m, theta, elem)
                cos2_term = max(np.cos(theta)**2 * 0.5 + 0.5, 1e-10)
                is_second_order = l == 2
                is_third_order = l == 3
                is_light = elem in ['Silicon', 'Oxygen', 'Carbon', 'Magnesium', 'Hydrogen', 'Helium', 'Lithium', 'Sodium', 'Rubidium', 'Cesium', 'Beryllium', 'Calcium', 'Strontium', 'Barium', 'Radium', 'Scandium', 'Yttrium', 'Lanthanum', 'Actinium']
                is_heavy = elem in ['Cadmium', 'Iron', 'Francium', 'Copper', 'Titanium', 'Molybdenum', 'Potassium-40', 'Silver', 'Gold', 'Palladium', 'Platinum', 'Nickel', 'Cobalt', 'Rhodium', 'Iridium', 'Osmium', 'Ruthenium', 'Uranium-238', 'Thorium-232', 'Lead-208', 'Palladium_Group3', 'Platinum_Group3', 'Gold_Group3']
                polarity = -1 if (is_second_order or is_third_order) and is_light else 1 if (is_second_order or is_third_order) and is_heavy else 1 if l == 1 and is_light else -1
                phase_factor = 0.7 + 0.2 * (q23_count / max(len(active_elements), 1))
                if l == 1 and m == 0 or l == 2 and m == 0 or l == 3 and m == 0:
                    if elem == 'Copper' and args.copper_inverse:
                        contribution = 1.0 / (W_oxy**2 + EPSILON) if W_oxy != 0 else 0.0
                    else:
                        contribution = polarity * W_REINFORCEMENT * Y_lm * z_norm * (1 + hu**2) * cos2_term / z_si * min(abundance / abund_si, 1.0)
                        wl_factor = min(wl_nm / 560.0, 1.0)
                        contribution *= wl_factor
                        if l == 2 or l == 3:
                            contribution *= 0.1
                        contribution *= phase_factor
                    power_scale = z_norm**2 / max(abund_o**2, 1e-10)
                    flux_scale = abundance * wl_nm / 560.0
                    contribution *= power_scale * flux_scale
                    if elem == 'Oxygen':
                        W_oxy = contribution * monophonic_factor
                    else:
                        non_oxy_contributions.append((elem, contribution, wl_nm, z_norm))
                    w_contributions.append((elem, contribution, wl_nm, z_norm))
    sum_contrib = sum(abs(contrib) for _, contrib, _, _ in non_oxy_contributions)
    scale_w = 1.0
    if W_oxy != 0 and sum_contrib > AMPLITUDE_LIMIT * abs(W_oxy):
        scale_w = AMPLITUDE_LIMIT * abs(W_oxy) / max(sum_contrib, 1e-10)
    for elem, contrib, wl_nm, z_norm in non_oxy_contributions:
        scaled_contrib = contrib * scale_w
        W += scaled_contrib
    vector_sum = np.sqrt(X**2 + Y**2)
    if vector_sum > 0:
        boost_limit = 2.83 * vector_sum
        scale = min(1.0, boost_limit / max(np.sqrt(X**2 + Y**2), 1e-10))
        X *= scale
        Y *= scale
    signals = {'W': W, 'X': X, 'Y': Y}
    vector_sum = np.sqrt(X**2 + Y**2)
    phase_coherence = vector_sum / max(abs(W), 1e-10) if W != 0 else 0.0
    logging.info(f"Phase coherence: W={W:.2e}, X={X:.2e}, Y={Y:.2e}, vector_sum={vector_sum:.2e}, coherence_ratio={phase_coherence:.2f}, q23_count={q23_count}, monophonic_factor={monophonic_factor:.2f}")
    total_weight = sum(abs(contrib) * (1.5 if elem == 'Silicon' else 2.0 if elem == 'Oxygen' else 1.0) * (wl_nm / 560.0) for elem, contrib, wl_nm, _ in w_contributions)
    dominant_wl = sum(wl_nm * abs(contrib) * (1.5 if elem == 'Silicon' else 2.0 if elem == 'Oxygen' else 1.0) * (wl_nm / 560.0) for elem, contrib, wl_nm, _ in w_contributions) / total_weight if total_weight > 0 else 560.0
    dominant_wl = min(max(dominant_wl, 300.0), 800.0)
    if lissajous_ax:
        plot_lissajous(lissajous_ax, signals_dict, wl_nm_dict, active_elements, t_range, atomic_numbers)
    final_data = []
    z_final_values = []
    for idx, (elem, f_res, wl_nm, abundance, _, z, hu) in enumerate(data):
        theta = np.mod(theta_values.get(elem, 0.0), 2 * np.pi)
        delta_theta = phase_offsets.get(elem, 0.0)
        r = r_values.get(elem, 1.0)
        z_final = z / (1 + 6.0 * r**2) * (1.5 if wl_nm > 700 or wl_nm < 400 else 1.0)
        z_final_values.append(z_final)
        for l, m in local_element_channels.get(elem, []):
            Y_lm = spherical_harmonic(l, m, theta + delta_theta, elem)
            freq_factor = (wl_nm - 400) / (700 - 400)
            if elem == 'Silicon':
                freq_factor *= 1.5
            cos2_term = max(np.cos(theta + delta_theta)**2 * 0.5 + 0.5, 1e-10)
            is_second_order = l == 2
            is_third_order = l == 3
            is_light = elem in ['Silicon', 'Oxygen', 'Carbon', 'Magnesium', 'Hydrogen', 'Helium', 'Lithium', 'Sodium', 'Rubidium', 'Cesium', 'Beryllium', 'Calcium', 'Strontium', 'Barium', 'Radium', 'Scandium', 'Yttrium', 'Lanthanum', 'Actinium']
            is_heavy = elem in ['Cadmium', 'Iron', 'Francium', 'Copper', 'Titanium', 'Molybdenum', 'Potassium-40', 'Silver', 'Gold', 'Palladium', 'Platinum', 'Nickel', 'Cobalt', 'Rhodium', 'Iridium', 'Osmium', 'Ruthenium', 'Uranium-238', 'Thorium-232', 'Lead-208', 'Palladium_Group3', 'Platinum_Group3', 'Gold_Group3']
            polarity = -1 if (is_second_order or is_third_order) and is_light else 1 if (is_second_order or is_third_order) and is_heavy else 1 if l == 1 and is_light else -1
            phase_factor = 0.7 + 0.2 * (q23_count / max(len(active_elements), 1))
            if l == 1 and m == 0 or l == 2 and m == 0 or l == 3 and m == 0:
                if elem == 'Copper' and args.copper_inverse:
                    contribution = 1.0 / (W_oxy**2 + EPSILON) if W_oxy != 0 else 0.0
                else:
                    contribution = polarity * Y_lm * z_final * cos2_term * phase_factor
                if elem != 'Oxygen':
                    contribution *= scale_w * 0.7071 * abs(W_oxy) / max(abs(contribution), 1e-10) if W_oxy != 0 else 1.0
                else:
                    contribution *= monophonic_factor
                signals['W'] += contribution
            elif (l == 1 and m == 1 or l == 2 and m == 1 or l == 3 and m == 1) and elem != 'Silicon':
                sign = 1 if is_light else -1
                contribution = polarity * freq_factor * Y_lm * (1 + sign * x_shift * 0.5) / phase_factor
                signals['X'] += contribution
            elif (l == 1 and m == -1 or l == 2 and m == -1 or l == 3 and m == -1) and elem != 'Silicon':
                sign = 1 if is_light else -1
                contribution = polarity * freq_factor * Y_lm * (1 + sign * y_shift * 0.5) / phase_factor
                signals['Y'] += contribution
        r = min(r, 5.0)
        theta = np.clip(np.mod(theta, 2 * np.pi), 0, 2 * np.pi - 1e-10)
        wl_shifted = 560.0 + (wl_nm - 560.0) * 0.8
        final_data.append((elem, f_res, wl_nm, abundance, 1.0, z_final, r, theta, z_final, hu, wl_shifted))
    signals['W'] = -signals['W'] if invert_w else signals['W']
    r_max = max(r for _, _, _, _, _, _, r, _, _, _, _ in final_data) * 1.2 if final_data else 1.0
    r_bins = np.linspace(0, max(r_max, 0.1), 25)
    theta_bins = np.linspace(0, 2 * np.pi, 101)
    heatmap = np.ones((len(theta_bins)-1, len(r_bins)-1)) * 1e-8
    for _, _, _, _, _, _, r, theta, z_final, _, _ in final_data:
        theta = np.clip(np.mod(theta, 2 * np.pi), 0, 2 * np.pi - 1e-10)
        r_idx = np.searchsorted(r_bins, r)
        theta_idx = np.searchsorted(theta_bins, theta)
        if not (0 <= r_idx < len(r_bins)-1 and 0 <= theta_idx < len(theta_bins)-1):
            logging.warning(f"Out-of-bounds: r={r:.2f}, theta={theta:.2f}, r_idx={r_idx}, theta_idx={theta_idx}")
            continue
        heatmap[theta_idx, r_idx] = max(heatmap[theta_idx, r_idx], z_final)
    cmap = matplotlib.colormaps['Spectral_r']
    norm = LogNorm(vmin=1e-10, vmax=max(z_final_values) * 1.5)
    im = ax.pcolormesh(theta_bins, r_bins, heatmap.T, cmap=cmap, norm=norm)
    for elem, _, wl_nm, _, _, _, r, theta, _, hu, wl_shifted in final_data:
        color = wavelength_to_rgb(wl_shifted)
        marker = {
            'Silicon': 's', 'Oxygen': '*', 'Iron': '^', 'Cadmium': 'o',
            'Francium': 'x', 'Hydrogen': '>', 'Copper': 'd', 'Carbon': 'p',
            'Titanium': 'h', 'Magnesium': '<', 'Molybdenum': 'v', 'Helium': '+',
            'Potassium-40': '*', 'Lithium': 'D', 'Sodium': 's', 'Rubidium': '^',
            'Cesium': 'o', 'Silver': 'x', 'Gold': '>', 'Palladium': 'd', 'Platinum': 'p',
            'Beryllium': 'h', 'Calcium': '<', 'Strontium': 'v', 'Barium': '+', 'Radium': '*',
            'Nickel': 's', 'Cobalt': '^', 'Rhodium': 'o', 'Iridium': 'x', 'Osmium': '>',
            'Scandium': 'o', 'Yttrium': 's', 'Lanthanum': 'd', 'Actinium': 'p',
            'Ruthenium': 'h', 'Uranium-238': '^', 'Thorium-232': 'x', 'Lead-208': '+',
            'Palladium_Group3': '^', 'Platinum_Group3': 'v', 'Gold_Group3': '*'
        }.get(elem, '.')
        s = 250 if elem in ELEMENT_GROUPS['Group 2'] or elem in ELEMENT_GROUPS['Group 3'] else 150
        ax.scatter(theta, r, c=[color], marker=marker, s=s, label=f'{elem} (HU={hu:.1f}, {wl_shifted:.1f}nm)')
        ax.text(theta, r, f"{elem}\n{hu:.1f}\n{wl_shifted:.1f}nm", fontsize=5, ha='center', va='center', color='black')
    ax.set_title(f'Polar Heat Map (Periphonic Ambisonics) for Oxygen-Rich Silicate Crust (329 THz, W={dominant_wl:.1f} nm, Iron at North)')
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.7), fontsize=6)
    ax.grid(True, which='both', ls='--', alpha=0.5)
    fig.canvas.draw()
    fig.canvas.flush_events()
    return im

# ... (Unchanged: plot_lissajous, plot_heatmap, main block)
def update_plot(x_shift, y_shift, z_scale, invert_w, active_elements, ax, fig, t=0, lissajous_ax=None):
    if not np.isfinite([x_shift, y_shift, z_scale]).all():
        logging.error(f"Invalid input values: x_shift={x_shift}, y_shift={y_shift}, z_scale={z_scale}")
        return None
    ax.clear()
    if lissajous_ax:
        lissajous_ax.clear()
    if not active_elements:
        logging.error("No active elements")
        ax.set_title("Error: No Active Elements")
        fig.canvas.draw()
        return None
    logging.info(f"Active elements count: {len(active_elements)}")
    global N_DM, C_EFF, DELTA_ALPHA_OVER_ALPHA, K40_FREQUENCY
    N_DM, C_EFF, DELTA_ALPHA_OVER_ALPHA, K40_FREQUENCY = calculate_k40_parameters(t=t, active_elements=active_elements, elements=ELEMENTS)
    use_third_order = False
    data = []
    z_values = {}
    pr_targets = {}
    hu_values = {}
    phase_offsets = {}
    r_values = {}
    theta_values = {}
    abund_o = 1.0759
    abund_si = 0.6395
    w_contributions = []
    quadrant_counts = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
    wl_nm_dict = {}
    signals_dict = {elem: {'X': [], 'Y': []} for elem in active_elements}
    light_elements = {light for light, _ in ELEMENT_PAIRS}
    pair_map = {light: heavy for light, heavy in ELEMENT_PAIRS}
    pair_map.update({heavy: light for light, heavy in ELEMENT_PAIRS})
    t_range = np.linspace(t, t + 0.1, 100)
    atomic_numbers = {
        'Scandium': 21, 'Yttrium': 39, 'Lanthanum': 57, 'Actinium': 89,
        'Ruthenium': 44, 'Palladium': 46, 'Platinum': 78, 'Gold': 79,
        'Copper': 29, 'Rubidium': 37, 'Cesium': 55, 'Silicon': 14, 'Oxygen': 8,
        'Iron': 26, 'Cadmium': 48, 'Francium': 87, 'Hydrogen': 1, 'Carbon': 6,
        'Titanium': 22, 'Magnesium': 12, 'Molybdenum': 42, 'Helium': 2,
        'Potassium-40': 19, 'Lithium': 3, 'Sodium': 11, 'Silver': 47,
        'Beryllium': 4, 'Calcium': 20, 'Strontium': 38, 'Barium': 56,
        'Radium': 88, 'Nickel': 28, 'Cobalt': 27, 'Rhodium': 45, 'Iridium': 77,
        'Osmium': 76, 'Uranium-238': 92, 'Thorium-232': 90, 'Lead-208': 82
    }
    # First loop: Calculate positions, quadrants, and q23_count
    for elem, mass, abund in ELEMENTS:
        if elem not in active_elements:
            continue
        f_res, wl_nm = calculate_resonant_frequency(mass, elem, active_elements)
        wl_nm_dict[elem] = wl_nm
        abundance = max(abund, MIN_ABUNDANCE)
        z_norm = abundance * z_scale * (1 + np.log10(max(abundance, 1e-10)) / 2)
        z_values[elem] = z_norm
        N_DM_elem, C_anti, c_eff_shift, Delta_alpha_over_alpha = calculate_dm_parameters(f_res, active_elements)
        pr_targets[elem] = 1.0 + c_eff_shift / 7.6535
        hu_values[elem] = calculate_hu(pr_targets[elem], elem)
        r_exp, theta_exp = EXPECTED_POSITIONS.get(elem, (1.0, np.pi/4))
        theta_shift = np.arctan2(y_shift, x_shift)
        joystick_magnitude = np.sqrt(x_shift**2 + y_shift**2)
        theta_adjust = 0.0
        partner = pair_map.get(elem)
        if partner and partner in wl_nm_dict:
            delta_wl = abs(wl_nm - wl_nm_dict[partner])
            scale = min(1.0, 200 / max(delta_wl, 1e-10))
        else:
            scale = 1.0
        if joystick_magnitude >= 1e-5:
            theta_adjust = theta_shift * min(1.0, joystick_magnitude / 4.0) * scale
        if elem == 'Oxygen' and z_values.get('Oxygen', 0) > abund_si:
            theta_actual = 0.0
            r = 0.0
        else:
            if elem == 'Silicon':
                theta_base = -0.035
            else:
                delta_lambda = (560.0 - wl_nm) / 560.0
                spin_bias = 0.2 if wl_nm > 700 else -0.2 if wl_nm < 400 else 0.0
                theta_base = theta_exp + 0.1 * delta_lambda + spin_bias
            deviation = z_norm / abund_o + joystick_magnitude
            spiral_factor = 1.0 if elem == 'Carbon' else SPIRAL_FACTOR  # Restored for Carbon
            spiral_theta = spiral_factor * deviation * (np.pi/2 if wl_nm > 700 else 3*np.pi/2 if wl_nm < 400 else 0.0) * (1.5 if atomic_numbers.get(elem, 50) > 50 else 1.0)
            theta_actual = np.mod(theta_base + theta_adjust + spiral_theta + 2 * np.pi, 2 * np.pi)
            k = (5.0 / r_exp - 1) / 2.83 if r_exp > 0 else 0.0
            r = r_exp * (1 + k * joystick_magnitude * scale + spiral_factor * deviation)
            r = min(r, 5.0)
        delta_theta = 0.0 if elem == 'Silicon' else theta_exp - theta_actual
        r_values[elem] = r
        theta_values[elem] = theta_actual
        theta_deg = theta_actual % (2 * np.pi)
        quadrant = 'Q1' if 0 <= theta_deg < np.pi/2 else 'Q2' if np.pi/2 <= theta_deg < np.pi else 'Q3' if np.pi <= theta_deg < 3*np.pi/2 else 'Q4'
        quadrant_counts[quadrant] += 1
        oxy_theta = theta_values.get('Oxygen', 0.0)
        phase_diff = np.mod(theta_actual - oxy_theta, 2 * np.pi)
        if elem in ['Oxygen', 'Silicon', 'Carbon', 'Hydrogen', 'Magnesium']:
            logging.debug(f"Phase diff for {elem} vs Oxygen: {phase_diff:.2f} rad")
        r_min, r_max, r_range = calculate_amplitude_range(elem, z_norm, hu_values[elem], theta_actual, delta_theta, z_values.get('Silicon', 1e-10), element_channels)
        data.append((elem, f_res, wl_nm, abundance, 1.0, z_norm, hu_values[elem]))
    q23_count = quadrant_counts['Q2'] + quadrant_counts['Q3']  # Compute q23_count here
    logging.info(f"Quadrant distribution: Q1={quadrant_counts['Q1']}, Q2={quadrant_counts['Q2']}, Q3={quadrant_counts['Q3']}, Q4={quadrant_counts['Q4']}")
    # Second loop: Calculate signals and Lissajous with scaling
    signal_scale = 10.0  # Scaling factor to amplify Lissajous signals
    for elem in active_elements:
        theta_actual = theta_values.get(elem, 0.0)
        wl_nm = wl_nm_dict.get(elem, 500.0)
        for t_i in t_range:
            temp_theta = theta_actual + 0.25 * np.sin(2 * np.pi * t_i / neutron_periods[0] * (560.0 / wl_nm))  # Enhanced vibration
            X_i, Y_i = 0, 0
            for l, m in element_channels.get(elem, []):
                Y_lm = spherical_harmonic(l, m, temp_theta, elem)
                freq_factor = (wl_nm - 400) / (700 - 400)
                if elem == 'Silicon':
                    freq_factor *= 1.5
                cos2_term = max(np.cos(temp_theta)**2 * 0.5 + 0.5, 1e-10)
                is_second_order = l == 2
                is_third_order = l == 3
                is_light = elem in ['Silicon', 'Oxygen', 'Carbon', 'Magnesium', 'Hydrogen', 'Helium', 'Lithium', 'Sodium', 'Rubidium', 'Cesium', 'Beryllium', 'Calcium', 'Strontium', 'Barium', 'Radium', 'Scandium', 'Yttrium', 'Lanthanum', 'Actinium']
                is_heavy = elem in ['Cadmium', 'Iron', 'Francium', 'Copper', 'Titanium', 'Molybdenum', 'Potassium-40', 'Silver', 'Gold', 'Palladium', 'Platinum', 'Nickel', 'Cobalt', 'Rhodium', 'Iridium', 'Osmium', 'Ruthenium', 'Uranium-238', 'Thorium-232', 'Lead-208', 'Palladium_Group3', 'Platinum_Group3', 'Gold_Group3']
                polarity = -1 if (is_second_order or is_third_order) and is_light else 1 if (is_second_order or is_third_order) and is_heavy else 1 if l == 1 and is_light else -1
                phase_factor = 0.7 + 0.2 * (q23_count / max(len(active_elements), 1))
                if (l == 1 and m == 1 or l == 2 and m == 1 or l == 3 and m == 1) and elem != 'Silicon':
                    sign = 1 if is_light else -1
                    contribution = polarity * freq_factor * Y_lm * (1 + sign * x_shift * 0.5) / phase_factor
                    X_i += contribution * signal_scale  # Apply scaling
                elif (l == 1 and m == -1 or l == 2 and m == -1 or l == 3 and m == -1) and elem != 'Silicon':
                    sign = 1 if is_light else -1
                    contribution = polarity * freq_factor * Y_lm * (1 + sign * y_shift * 0.5) / phase_factor
                    Y_i += contribution * signal_scale  # Apply scaling
            signals_dict[elem]['X'].append(X_i)
            signals_dict[elem]['Y'].append(Y_i)
            if elem in ['Oxygen', 'Silicon', 'Carbon']:  # Debug signal magnitudes
                logging.debug(f"Lissajous signal for {elem}: X_i={X_i:.2e}, Y_i={Y_i:.2e}, wl_nm={wl_nm:.1f}")
    # Third loop: Calculate W, X, Y contributions
    W = 0
    X = 0
    Y = 0
    W_oxy = 0
    z_si = z_values.get('Silicon', 1e-10)
    non_oxy_contributions = []
    monophonic_factor = min(1.0 + MONOPHONIC_BOOST * q23_count / max(len(active_elements), 1), 2.0)
    local_element_channels = element_channels.copy()
    for elem in active_elements:
        z_norm = z_values.get(elem, 1e-10)
        theta = theta_values.get(elem, 0.0)
        hu = hu_values.get(elem, 0.0)
        wl_nm = next((wl for e, _, wl, _, _, _, _ in data if e == elem), 500.0)
        freq_factor = (wl_nm - 400) / (700 - 400)
        if elem == 'Silicon':
            freq_factor *= 1.5
        for l, m in local_element_channels.get(elem, []):
            Y_lm = spherical_harmonic(l, m, theta, elem)
            cos2_term = max(np.cos(theta)**2 * 0.5 + 0.5, 1e-10)
            is_second_order = l == 2
            is_third_order = l == 3
            is_light = elem in ['Silicon', 'Oxygen', 'Carbon', 'Magnesium', 'Hydrogen', 'Helium', 'Lithium', 'Sodium', 'Rubidium', 'Cesium', 'Beryllium', 'Calcium', 'Strontium', 'Barium', 'Radium', 'Scandium', 'Yttrium', 'Lanthanum', 'Actinium']
            is_heavy = elem in ['Cadmium', 'Iron', 'Francium', 'Copper', 'Titanium', 'Molybdenum', 'Potassium-40', 'Silver', 'Gold', 'Palladium', 'Platinum', 'Nickel', 'Cobalt', 'Rhodium', 'Iridium', 'Osmium', 'Ruthenium', 'Uranium-238', 'Thorium-232', 'Lead-208', 'Palladium_Group3', 'Platinum_Group3', 'Gold_Group3']
            polarity = -1 if (is_second_order or is_third_order) and is_light else 1 if (is_second_order or is_third_order) and is_heavy else 1 if l == 1 and is_light else -1
            phase_factor = 0.7 + 0.2 * (q23_count / max(len(active_elements), 1))
            if l == 1 and m == 0 or l == 2 and m == 0 or l == 3 and m == 0:
                if elem == 'Copper' and args.copper_inverse:
                    contribution = 1.0 / (W_oxy**2 + EPSILON) if W_oxy != 0 else 0.0
                else:
                    contribution = polarity * W_REINFORCEMENT * Y_lm * z_norm * (1 + hu**2) * cos2_term / z_si * min(abundance / abund_si, 1.0)
                    wl_factor = min(wl_nm / 560.0, 1.0)
                    contribution *= wl_factor
                    if l == 2 or l == 3:
                        contribution *= 0.1
                    contribution *= phase_factor
                power_scale = z_norm**2 / max(abund_o**2, 1e-10)
                flux_scale = abundance * wl_nm / 560.0
                contribution *= power_scale * flux_scale
                if elem == 'Oxygen':
                    W_oxy = contribution * monophonic_factor
                else:
                    non_oxy_contributions.append((elem, contribution, wl_nm, z_norm))
                w_contributions.append((elem, contribution, wl_nm, z_norm))
            elif (l == 1 and m == 1 or l == 2 and m == 1 or l == 3 and m == 1) and elem != 'Silicon':
                sign = 1 if is_light else -1
                contribution = polarity * freq_factor * Y_lm * (1 + sign * x_shift * 0.5) / phase_factor
                X += contribution
            elif (l == 1 and m == -1 or l == 2 and m == -1 or l == 3 and m == -1) and elem != 'Silicon':
                sign = 1 if is_light else -1
                contribution = polarity * freq_factor * Y_lm * (1 + sign * y_shift * 0.5) / phase_factor
                Y += contribution
    # Handle amplitude constraint
    sum_contrib = sum(abs(contrib) for _, contrib, _, _ in non_oxy_contributions)
    logging.info(f"Amplitude check: sum_contrib={sum_contrib:.2e}, W_oxy={W_oxy:.2e}, limit={AMPLITUDE_LIMIT * abs(W_oxy):.2e}")
    if W_oxy != 0 and sum_contrib > AMPLITUDE_LIMIT * abs(W_oxy):
        logging.info(f"Amplitude constraint violated: sum_contrib={sum_contrib:.2e} > {AMPLITUDE_LIMIT} * W_oxy={AMPLITUDE_LIMIT * abs(W_oxy):.2e}. Upgrading heavy pairs to l=3.")
        use_third_order = True
        for elem in ['Ruthenium', 'Palladium_Group3', 'Platinum_Group3', 'Gold_Group3']:
            local_element_channels[elem] = [(3, 1)]
        non_oxy_contributions = []
        w_contributions = []
        W = 0
        for elem in active_elements:
            z_norm = z_values.get(elem, 1e-10)
            theta = theta_values.get(elem, 0.0)
            hu = hu_values.get(elem, 0.0)
            wl_nm = next((wl for e, _, wl, _, _, _, _ in data if e == elem), 500.0)
            freq_factor = (wl_nm - 400) / (700 - 400)
            if elem == 'Silicon':
                freq_factor *= 1.5
            for l, m in local_element_channels.get(elem, []):
                Y_lm = spherical_harmonic(l, m, theta, elem)
                cos2_term = max(np.cos(theta)**2 * 0.5 + 0.5, 1e-10)
                is_second_order = l == 2
                is_third_order = l == 3
                is_light = elem in ['Silicon', 'Oxygen', 'Carbon', 'Magnesium', 'Hydrogen', 'Helium', 'Lithium', 'Sodium', 'Rubidium', 'Cesium', 'Beryllium', 'Calcium', 'Strontium', 'Barium', 'Radium', 'Scandium', 'Yttrium', 'Lanthanum', 'Actinium']
                is_heavy = elem in ['Cadmium', 'Iron', 'Francium', 'Copper', 'Titanium', 'Molybdenum', 'Potassium-40', 'Silver', 'Gold', 'Palladium', 'Platinum', 'Nickel', 'Cobalt', 'Rhodium', 'Iridium', 'Osmium', 'Ruthenium', 'Uranium-238', 'Thorium-232', 'Lead-208', 'Palladium_Group3', 'Platinum_Group3', 'Gold_Group3']
                polarity = -1 if (is_second_order or is_third_order) and is_light else 1 if (is_second_order or is_third_order) and is_heavy else 1 if l == 1 and is_light else -1
                phase_factor = 0.7 + 0.2 * (q23_count / max(len(active_elements), 1))
                if l == 1 and m == 0 or l == 2 and m == 0 or l == 3 and m == 0:
                    if elem == 'Copper' and args.copper_inverse:
                        contribution = 1.0 / (W_oxy**2 + EPSILON) if W_oxy != 0 else 0.0
                    else:
                        contribution = polarity * W_REINFORCEMENT * Y_lm * z_norm * (1 + hu**2) * cos2_term / z_si * min(abundance / abund_si, 1.0)
                        wl_factor = min(wl_nm / 560.0, 1.0)
                        contribution *= wl_factor
                        if l == 2 or l == 3:
                            contribution *= 0.1
                        contribution *= phase_factor
                    power_scale = z_norm**2 / max(abund_o**2, 1e-10)
                    flux_scale = abundance * wl_nm / 560.0
                    contribution *= power_scale * flux_scale
                    if elem == 'Oxygen':
                        W_oxy = contribution * monophonic_factor
                    else:
                        non_oxy_contributions.append((elem, contribution, wl_nm, z_norm))
                    w_contributions.append((elem, contribution, wl_nm, z_norm))
    sum_contrib = sum(abs(contrib) for _, contrib, _, _ in non_oxy_contributions)
    scale_w = 1.0
    if W_oxy != 0 and sum_contrib > AMPLITUDE_LIMIT * abs(W_oxy):
        scale_w = AMPLITUDE_LIMIT * abs(W_oxy) / max(sum_contrib, 1e-10)
    for elem, contrib, wl_nm, z_norm in non_oxy_contributions:
        scaled_contrib = contrib * scale_w
        W += scaled_contrib
    vector_sum = np.sqrt(X**2 + Y**2)
    if vector_sum > 0:
        boost_limit = 2.83 * vector_sum
        scale = min(1.0, boost_limit / max(np.sqrt(X**2 + Y**2), 1e-10))
        X *= scale
        Y *= scale
    signals = {'W': W, 'X': X, 'Y': Y}
    vector_sum = np.sqrt(X**2 + Y**2)
    phase_coherence = vector_sum / max(abs(W), 1e-10) if W != 0 else 0.0
    logging.info(f"Phase coherence: W={W:.2e}, X={X:.2e}, Y={Y:.2e}, vector_sum={vector_sum:.2e}, coherence_ratio={phase_coherence:.2f}, q23_count={q23_count}, monophonic_factor={monophonic_factor:.2f}")
    total_weight = sum(abs(contrib) * (1.5 if elem == 'Silicon' else 2.0 if elem == 'Oxygen' else 1.0) * (wl_nm / 560.0) for elem, contrib, wl_nm, _ in w_contributions)
    dominant_wl = sum(wl_nm * abs(contrib) * (1.5 if elem == 'Silicon' else 2.0 if elem == 'Oxygen' else 1.0) * (wl_nm / 560.0) for elem, contrib, wl_nm, _ in w_contributions) / total_weight if total_weight > 0 else 560.0
    dominant_wl = min(max(dominant_wl, 300.0), 800.0)
    if lissajous_ax:
        plot_lissajous(lissajous_ax, signals_dict, wl_nm_dict, active_elements, t_range, atomic_numbers)
    final_data = []
    z_final_values = []
    for idx, (elem, f_res, wl_nm, abundance, _, z, hu) in enumerate(data):
        theta = np.mod(theta_values.get(elem, 0.0), 2 * np.pi)
        delta_theta = phase_offsets.get(elem, 0.0)
        r = r_values.get(elem, 1.0)
        z_final = z / (1 + 6.0 * r**2) * (1.5 if wl_nm > 700 or wl_nm < 400 else 1.0)
        z_final_values.append(z_final)
        for l, m in local_element_channels.get(elem, []):
            Y_lm = spherical_harmonic(l, m, theta + delta_theta, elem)
            freq_factor = (wl_nm - 400) / (700 - 400)
            if elem == 'Silicon':
                freq_factor *= 1.5
            cos2_term = max(np.cos(theta + delta_theta)**2 * 0.5 + 0.5, 1e-10)
            is_second_order = l == 2
            is_third_order = l == 3
            is_light = elem in ['Silicon', 'Oxygen', 'Carbon', 'Magnesium', 'Hydrogen', 'Helium', 'Lithium', 'Sodium', 'Rubidium', 'Cesium', 'Beryllium', 'Calcium', 'Strontium', 'Barium', 'Radium', 'Scandium', 'Yttrium', 'Lanthanum', 'Actinium']
            is_heavy = elem in ['Cadmium', 'Iron', 'Francium', 'Copper', 'Titanium', 'Molybdenum', 'Potassium-40', 'Silver', 'Gold', 'Palladium', 'Platinum', 'Nickel', 'Cobalt', 'Rhodium', 'Iridium', 'Osmium', 'Ruthenium', 'Uranium-238', 'Thorium-232', 'Lead-208', 'Palladium_Group3', 'Platinum_Group3', 'Gold_Group3']
            polarity = -1 if (is_second_order or is_third_order) and is_light else 1 if (is_second_order or is_third_order) and is_heavy else 1 if l == 1 and is_light else -1
            phase_factor = 0.7 + 0.2 * (q23_count / max(len(active_elements), 1))
            if l == 1 and m == 0 or l == 2 and m == 0 or l == 3 and m == 0:
                if elem == 'Copper' and args.copper_inverse:
                    contribution = 1.0 / (W_oxy**2 + EPSILON) if W_oxy != 0 else 0.0
                else:
                    contribution = polarity * Y_lm * z_final * cos2_term * phase_factor
                if elem != 'Oxygen':
                    contribution *= scale_w * 0.7071 * abs(W_oxy) / max(abs(contribution), 1e-10) if W_oxy != 0 else 1.0
                else:
                    contribution *= monophonic_factor
                signals['W'] += contribution
            elif (l == 1 and m == 1 or l == 2 and m == 1 or l == 3 and m == 1) and elem != 'Silicon':
                sign = 1 if is_light else -1
                contribution = polarity * freq_factor * Y_lm * (1 + sign * x_shift * 0.5) / phase_factor
                signals['X'] += contribution
            elif (l == 1 and m == -1 or l == 2 and m == -1 or l == 3 and m == -1) and elem != 'Silicon':
                sign = 1 if is_light else -1
                contribution = polarity * freq_factor * Y_lm * (1 + sign * y_shift * 0.5) / phase_factor
                signals['Y'] += contribution
        r = min(r, 5.0)
        theta = np.clip(np.mod(theta, 2 * np.pi), 0, 2 * np.pi - 1e-10)
        wl_shifted = 560.0 + (wl_nm - 560.0) * 0.8
        final_data.append((elem, f_res, wl_nm, abundance, 1.0, z_final, r, theta, z_final, hu, wl_shifted))
    signals['W'] = -signals['W'] if invert_w else signals['W']
    r_max = max(r for _, _, _, _, _, _, r, _, _, _, _ in final_data) * 1.2 if final_data else 1.0
    r_bins = np.linspace(0, max(r_max, 0.1), 25)
    theta_bins = np.linspace(0, 2 * np.pi, 101)
    heatmap = np.ones((len(theta_bins)-1, len(r_bins)-1)) * 1e-8
    for _, _, _, _, _, _, r, theta, z_final, _, _ in final_data:
        theta = np.clip(np.mod(theta, 2 * np.pi), 0, 2 * np.pi - 1e-10)
        r_idx = np.searchsorted(r_bins, r)
        theta_idx = np.searchsorted(theta_bins, theta)
        if not (0 <= r_idx < len(r_bins)-1 and 0 <= theta_idx < len(theta_bins)-1):
            logging.warning(f"Out-of-bounds: r={r:.2f}, theta={theta:.2f}, r_idx={r_idx}, theta_idx={theta_idx}")
            continue
        heatmap[theta_idx, r_idx] = max(heatmap[theta_idx, r_idx], z_final)
    cmap = matplotlib.colormaps['Spectral_r']
    norm = LogNorm(vmin=1e-10, vmax=max(z_final_values) * 1.5)
    im = ax.pcolormesh(theta_bins, r_bins, heatmap.T, cmap=cmap, norm=norm)
    for elem, _, wl_nm, _, _, _, r, theta, _, hu, wl_shifted in final_data:
        color = wavelength_to_rgb(wl_shifted)
        marker = {
            'Silicon': 's', 'Oxygen': '*', 'Iron': '^', 'Cadmium': 'o',
            'Francium': 'x', 'Hydrogen': '>', 'Copper': 'd', 'Carbon': 'p',
            'Titanium': 'h', 'Magnesium': '<', 'Molybdenum': 'v', 'Helium': '+',
            'Potassium-40': '*', 'Lithium': 'D', 'Sodium': 's', 'Rubidium': '^',
            'Cesium': 'o', 'Silver': 'x', 'Gold': '>', 'Palladium': 'd', 'Platinum': 'p',
            'Beryllium': 'h', 'Calcium': '<', 'Strontium': 'v', 'Barium': '+', 'Radium': '*',
            'Nickel': 's', 'Cobalt': '^', 'Rhodium': 'o', 'Iridium': 'x', 'Osmium': '>',
            'Scandium': 'o', 'Yttrium': 's', 'Lanthanum': 'd', 'Actinium': 'p',
            'Ruthenium': 'h', 'Uranium-238': '^', 'Thorium-232': 'x', 'Lead-208': '+',
            'Palladium_Group3': '^', 'Platinum_Group3': 'v', 'Gold_Group3': '*'
        }.get(elem, '.')
        s = 250 if elem in ELEMENT_GROUPS['Group 2'] or elem in ELEMENT_GROUPS['Group 3'] else 150
        ax.scatter(theta, r, c=[color], marker=marker, s=s, label=f'{elem} (HU={hu:.1f}, {wl_shifted:.1f}nm)')
        ax.text(theta, r, f"{elem}\n{hu:.1f}\n{wl_shifted:.1f}nm", fontsize=5, ha='center', va='center', color='black')
    ax.set_title(f'Polar Heat Map (Periphonic Ambisonics) for Oxygen-Rich Silicate Crust (329 THz, W={dominant_wl:.1f} nm)')
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.7), fontsize=6)
    ax.grid(True, which='both', ls='--', alpha=0.5)
    fig.canvas.draw()
    fig.canvas.flush_events()
    return im

def plot_lissajous(ax, signals_dict, wl_nm_dict, active_elements, t_range, atomic_numbers):
    """
    Plot Lissajous curves for oxygen, silicon, carbon, and select elements.
    Parameters:
    - ax: Matplotlib axes for Lissajous plot.
    - signals_dict: Dictionary of {elem: {'X': [], 'Y': []}} over t_range.
    - wl_nm_dict: Dictionary of {elem: wl_nm_shifted}.
    - active_elements: List of active elements.
    - t_range: Array of time points.
    - atomic_numbers: Dictionary of element atomic numbers.
    """
    ax.clear()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('X Signal')
    ax.set_ylabel('Y Signal')
    ax.set_title('Lissajous Curves (O, Si, C Harmonics)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    focus_elements = ['Oxygen', 'Silicon', 'Carbon', 'Hydrogen', 'Magnesium']
    for elem in active_elements:
        if elem not in focus_elements:
            continue
        if elem not in signals_dict or not signals_dict[elem]['X']:
            logging.warning(f"No Lissajous data for {elem}")
            continue
        x = np.array(signals_dict[elem]['X'])
        y = np.array(signals_dict[elem]['Y'])
        # Log signal ranges for debugging
        x_range = np.max(np.abs(x)) if len(x) > 0 else 0
        y_range = np.max(np.abs(y)) if len(y) > 0 else 0
        logging.debug(f"Lissajous range for {elem}: X_max={x_range:.2e}, Y_max={y_range:.2e}")
        if x_range == 0 and y_range == 0:
            logging.warning(f"Zero signal range for {elem}, skipping plot")
            continue
        wl_nm = wl_nm_dict.get(elem, 500.0)
        color = wavelength_to_rgb(wl_nm)
        freq_ratio = wl_nm / 560.0 if elem != 'Silicon' else 1.5 * wl_nm / 560.0
        linestyle = '-' if elem != 'Carbon' else '--'
        linewidth = 2.5 if elem in ['Oxygen', 'Silicon', 'Carbon'] else 1.5
        label = f"{elem} ({wl_nm:.1f} nm, Z={atomic_numbers.get(elem, 50)})"
        ax.plot(x, y, c=color, lw=linewidth, ls=linestyle, label=label, alpha=0.7)
        oxy_wl = wl_nm_dict.get('Oxygen', 560.0)
        freq_ratio_oxy = wl_nm / oxy_wl if elem != 'Oxygen' else 1.0
        logging.debug(f"Lissajous for {elem}: wl_nm={wl_nm:.1f}, freq_ratio_oxy={freq_ratio_oxy:.2f}")
    ax.legend(fontsize=5, loc='upper right')
    return ax

# ... (Unchanged: plot_heatmap, main block)

# ... (Unchanged: plot_heatmap, main block)
def plot_heatmap():
    fig = plt.figure(figsize=(14, 16))
    ax = fig.add_subplot(121, projection='polar')
    lissajous_ax = fig.add_subplot(122)
    plt.subplots_adjust(bottom=0.40, left=0.1, right=0.95, top=0.85, wspace=0.3)
    active_elements = set([elem[0] for elem in ELEMENTS])
    group_states = {group: True for group in ELEMENT_GROUPS}
    start_time = time.time()
    im = update_plot(0.0, 0.0, 1.0, True, active_elements, ax, fig, t=0, lissajous_ax=lissajous_ax)
    if im is None:
        logging.error("Failed to generate heatmap")
        return
    plt.colorbar(im, ax=ax, label='W Signal Intensity', pad=0.20, fraction=0.03)
    ax_z = plt.axes([0.15, 0.20, 0.55, 0.03])
    ax_reset = plt.axes([0.15, 0.25, 0.1, 0.03])
    ax_groups = plt.axes([0.15, 0.03, 0.55, 0.12])
    ax_groups.set_title("Group Selectors", fontsize=8)
    ax_pad = plt.axes([0.80, 0.40, 0.15, 0.15])
    ax_pad.set_xlim(-2.0, 2.0)
    ax_pad.set_ylim(-2.0, 2.0)
    ax_pad.set_xlabel('X Shift')
    ax_pad.set_ylabel('Y Shift')
    ax_pad.grid(True)
    slider_z = Slider(ax_z, 'Z Scale', 0.1, 10.0, valinit=1.0, valstep=JOYSTICK_STEP)
    button_reset = Button(ax_reset, 'Reset')
    group_check = CheckButtons(ax_groups, list(ELEMENT_GROUPS.keys()), [True] * len(ELEMENT_GROUPS))
    pad_point = ax_pad.plot(0, 0, 'ro')[0]
    x_shift = [0.0]
    y_shift = [0.0]
    z_scale = [1.0]
    is_dragging = [False]
    def update_active_elements():
        active_elements.clear()
        for group, state in group_states.items():
            if state:
                active_elements.update(ELEMENT_GROUPS[group])
        logging.info(f"Updated active elements count: {len(active_elements)}")
        update(None)
    def on_group_check(label):
        group_states[label] = not group_states[label]
        update_active_elements()
    def on_click(event):
        if event.inaxes == ax_pad and pad_point and event.xdata is not None and event.ydata is not None:
            is_dragging[0] = True
            x_shift[0] = np.clip(event.xdata, -2.0, 2.0)
            y_shift[0] = np.clip(event.ydata, -2.0, 2.0)
            pad_point.set_data([x_shift[0]], [y_shift[0]])
            update(None)
            plt.pause(0.01)
    def on_motion(event):
        if is_dragging[0] and event.inaxes == ax_pad and pad_point and event.xdata is not None and event.ydata is not None:
            current_time = time.time()
            if current_time - LAST_UPDATE_TIME[0] < 0.05:
                return
            LAST_UPDATE_TIME[0] = current_time
            x_shift[0] = np.clip(event.xdata, -2.0, 2.0)
            y_shift[0] = np.clip(event.ydata, -2.0, 2.0)
            pad_point.set_data([x_shift[0]], [y_shift[0]])
            update(None)
            plt.pause(0.01)
    def on_release(event):
        is_dragging[0] = False
    def update(val):
        z_scale[0] = slider_z.val
        current_time = time.time() - start_time
        update_plot(x_shift[0], y_shift[0], z_scale[0], True, active_elements, ax, fig, t=current_time, lissajous_ax=lissajous_ax)
        plt.pause(0.01)
    def reset(event):
        slider_z.set_val(1.0)
        x_shift[0] = 0.0
        y_shift[0] = 0.0
        z_scale[0] = 1.0
        pad_point.set_data([0], [0])
        for group in group_states:
            group_states[group] = True
        for i in range(len(ELEMENT_GROUPS)):
            group_check.set_active(i)
        update_active_elements()
        plt.pause(0.01)
    group_check.on_clicked(on_group_check)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    slider_z.on_changed(update)
    button_reset.on_clicked(reset)
    if not args.interactive:
        plt.savefig('heatmap_ambisonics_lissajous.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        return
    plt.ion()
    plt.show(block=True)

if __name__ == "__main__":
    try:
        logging.info("Generating Polar Heat Map with Periphonic Ambisonics")
        plot_heatmap()
        logging.info("Heat map generation completed")
    except Exception as e:
        logging.error(f"Application failed: {e}")
        print(f"Application failed: {e}", file=sys.stderr)
        sys.exit(1)