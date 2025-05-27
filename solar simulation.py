import math
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Slider, Button
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import Counter
import itertools
import time

# Provided 8x8 table
original_table = [
    [1, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
    [1.143, 1, 0.857, 0.714, 0.571, 0.429, 0.286, 0.143],
    [1.333, 1.167, 1, 0.833, 0.667, 0.5, 0.333, 0.167],
    [1.6, 1.4, 1.2, 1, 0.8, 0.6, 0.4, 0.2],
    [2, 1.75, 1.5, 1.25, 1, 0.75, 0.5, 0.25],
    [2.667, 2.333, 2, 1.667, 1.333, 1, 0.667, 0.333],
    [4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5],
    [8, 7, 6, 5, 4, 3, 2, 1]
]

def flip_table_columns(table):
    n = len(table)
    flipped_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            flipped_table[i][n-1-j] = table[i][j]
    return flipped_table

# Flip the table
original_table = flip_table_columns(original_table)

def scale_table(table, origin_row, origin_col, target_value):
    origin_value = table[origin_row][origin_col]
    if origin_value == 0:
        raise ValueError("Origin value cannot be zero.")
    scaling_factor = target_value / origin_value
    scaled_table = [[value * scaling_factor for value in row] for row in table]
    return scaled_table, scaling_factor

def solve_kepler(M, e, tol=1e-6, max_iter=100):
    M = M % (2 * math.pi)
    E = M if e < 0.8 else math.pi
    for _ in range(max_iter):
        delta = (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
        E -= delta
        if abs(delta) < tol:
            return E
    return E

def get_true_anomaly(M_deg, e):
    M = math.radians(M_deg)
    E = solve_kepler(M, e)
    nu = 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2), math.sqrt(1 - e) * math.cos(E / 2))
    return nu

# Initial true anomalies at J2000 and real-world semi-major axes
initial_true_anomalies = []
orbital_data = [
    (0.387, 0.205630, 174.796),  # Mercury
    (0.723, 0.006772, 50.416),   # Venus
    (1.000, 0.016708, 358.617),  # Earth
    (1.524, 0.093400, 19.373),   # Mars
    (5.204, 0.048498, 20.020),   # Jupiter
    (9.583, 0.055508, 317.020),  # Saturn
    (19.200, 0.046381, 141.050), # Uranus
    (30.017, 0.008678, 256.228)  # Neptune
]
real_a = [0.387, 0.723, 1.000, 1.524, 5.204, 9.583, 19.200, 30.017]  # Real-world AU
for a, e, M in orbital_data:
    nu = get_true_anomaly(M, e)
    initial_true_anomalies.append(nu)

# Planet masses (Earth masses)
planet_masses = [0.0553, 0.815, 1.000, 0.107, 317.83, 95.16, 14.54, 17.15]

def get_orbital_parameters(i, scaled_table, t, multiplier, period, origin_row, origin_col):
    actual_parameters = [
        (57.9e6 / 149.6e6, 0.205630),  # Mercury
        (108.2e6 / 149.6e6, 0.006772), # Venus
        (149.6e6 / 149.6e6, 0.016708), # Earth
        (227.9e6 / 149.6e6, 0.093400), # Mars
        (778.5e6 / 149.6e6, 0.048498), # Jupiter
        (1433.5e6 / 149.6e6, 0.055508),# Saturn
        (2872.5e6 / 149.6e6, 0.046381),# Uranus
        (4495.1e6 / 149.6e6, 0.008678) # Neptune
    ]
    actual_a, actual_e = actual_parameters[i]
    # Scale real-world a based on matrix value
    matrix_scale = original_table[origin_row][origin_col] / original_table[2][2]  # Relative to (3D, 3D)
    base_a = real_a[i] * matrix_scale
    target_a = 1.0
    if base_a == 0:
        return base_a, 0.0
    if actual_a > target_a:
        factor = 1.0 / actual_a
        oscillation = (factor - 1.0) * (1 - math.sin(2 * math.pi * t / period))
        compression_factor = 1.0 + multiplier * oscillation
    else:
        factor = 1.0 / actual_a
        oscillation = (factor - 1.0) * (1 + math.sin(2 * math.pi * t / period))
        compression_factor = 1.0 + multiplier * oscillation
    a = base_a * max(0.1, compression_factor)
    e = actual_e * abs(compression_factor - 1.0) / abs(factor - 1.0) if factor != 1.0 else 0.0
    e = max(0.0, min(e, 0.999))
    return a, e

def update_plot(t):
    global pi, c, origin_row, origin_col, multiplier, base_period, flash_time
    try:
        origin_row = int(origin_row_str) - 1
        origin_col = int(origin_col_str) - 1
        if not (0 <= origin_row <= 7 and 0 <= origin_col <= 7):
            raise ValueError("Origin must be between 1 and 8.")
    except ValueError:
        print("Invalid origin input; using default (3D, 3D).")
        origin_row, origin_col = 2, 2

    try:
        period = float(base_period)
        if period <= 0.1:
            period = 2 * math.pi
            base_period = str(period)
            print("Invalid period; using default 2π.")
    except ValueError:
        period = 2 * math.pi
        base_period = str(period)
        print("Invalid period input; using default 2π.")

    pi = math.pi
    c = 299792
    pi_c_squared = pi * c * c
    target_value = 149.6e6

    scaled_table, scaling_factor = scale_table(original_table, origin_row, origin_col, target_value)

    semi_major_axes = []
    eccentricities = []
    angles = []
    for i in range(8):
        a, e = get_orbital_parameters(i, scaled_table, t, multiplier, period, origin_row, origin_col)
        semi_major_axes.append(a)
        eccentricities.append(e)
        theta = (initial_true_anomalies[i] + pi * t) % (2 * pi)
        angles.append(theta)

    # Compute alignment factors with weights
    alignment_factors = []
    weights = [1.0 if i < 3 else 0.5 for i in range(8)]  # Inner planets: 1.0, outer: 0.5
    n = len(planets)
    for i in range(n):
        weighted_cos_sum = 0
        weight_sum = 0
        for j in range(n):
            if i != j:
                weighted_cos_sum += weights[j] * math.cos(angles[i] - angles[j])
                weight_sum += weights[j]
        alignment_factor = (weighted_cos_sum / weight_sum + 1) / 2
        alignment_factors.append(alignment_factor)

    # Dynamic detectability, normalized to Earth at a=1 AU
    detectability = []
    earth_ref_det = 1.0
    venus_idx = 1
    earth_idx = 2
    for i, a in enumerate(semi_major_axes):
        if a == 0:
            det = 0.0
        else:
            proximity_factor = math.exp(-abs(a - 1.0) / 1.2)
            phase_boost = 1.0
            if i == earth_idx:
                angle_diff = abs(angles[earth_idx] - angles[venus_idx])
                min_diff = min(angle_diff, 2 * pi - angle_diff)
                if min_diff < math.radians(10):
                    phase_boost = 2.5 if origin_row == 5 and origin_col == 0 else 1.8
            det = (planet_masses[i] * a**2) * (1 + 0.5 * math.sin(2 * math.pi * t / (2 * period))) * alignment_factors[i] * proximity_factor * phase_boost / earth_ref_det
        detectability.append(det)
    total_det = sum(detectability)

    # Flash background if detectability is high
    current_time = time.time()
    if total_det > 0.8 * 5000:
        flash_time = current_time
        ax.set_facecolor('mistyrose')
    elif current_time - flash_time > 0.5:
        ax.set_facecolor('white')

    ax.clear()

    phi = math.pi * t
    orbit_lines = []
    planet_scatters = []
    distances = []
    for i, (a, e, planet, color, nu_0, det) in enumerate(zip(semi_major_axes, eccentricities, planets, colors, initial_true_anomalies, detectability)):
        b = a * math.sqrt(1 - e**2) if e < 1 else a
        theta = np.linspace(0, 2 * np.pi, 100)
        x_orbit = a * np.cos(theta)
        y_orbit = b * np.sin(theta)
        total_phi = phi + nu_0
        x_rot = x_orbit * np.cos(total_phi) - y_orbit * np.sin(total_phi)
        y_rot = x_orbit * np.sin(total_phi) + y_orbit * np.cos(total_phi)
        line_alpha = min(1.0, max(0.1, det / 1000))
        line, = ax.plot(x_rot, y_rot, color=color, label=planet, linewidth=1, alpha=line_alpha)
        marker_size = 30 if det > 1.0 else 10
        marker_alpha = min(1.0, max(0.1, det / 1000)) if planet not in ['Earth', 'Mars'] else 0.3
        x_planet = a * math.cos(total_phi)
        y_planet = b * math.sin(total_phi)
        scatter = ax.scatter([x_planet], [y_planet], color=color, s=marker_size, alpha=marker_alpha)
        orbit_lines.append(line)
        planet_scatters.append(scatter)
        perihelion = a * (1 - e)
        aphelion = a * (1 + e)
        distances.append((perihelion, aphelion))

    sun_scatter = ax.scatter([0], [0], color='yellow', s=200, label='Sun', zorder=10)

    distances_text = "\n".join(
        f"{planet}: {peri:.3f}-{aph:.3f} AU (Det: {det:.3f})"
        for planet, (peri, aph), det in zip(planets, distances, detectability)
    )
    text = ax.text(0.95, 0.5, f"Distances (Peri-Aphelion, AU) & Detectability:\n{distances_text}",
                   transform=ax.transAxes, fontsize=8, verticalalignment='center',
                   horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_aspect('equal')
    ax.set_xlim(-35, 35)
    ax.set_ylim(-35, 35)
    ax.set_xlabel("X (AU)")
    ax.set_ylabel("Y (AU)")
    ax.set_title(f"Solar System (Origin: ({origin_row + 1}D, {origin_col + 1}D), Target = 1 AU, πc² ≈ {pi_c_squared:.0f} km²/s²)")
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    ax.grid(True)

    return orbit_lines + planet_scatters + [sun_scatter, text]

def submit_origin_row(text):
    global origin_row_str
    origin_row_str = text.strip() if text.strip() else "3"

def submit_origin_col(text):
    global origin_col_str
    origin_col_str = text.strip() if text.strip() else "3"

def update_multiplier(val):
    global multiplier
    multiplier = slider.val

def submit_base_period(text):
    global base_period
    base_period = text.strip() if text.strip() else str(2 * math.pi)
    try:
        period = float(base_period)
        if period <= 0.1:
            base_period = str(2 * math.pi)
            print("Period too small; using default 2π.")
    except ValueError:
        base_period = str(2 * math.pi)
        print("Invalid period input; using default 2π.")

def reset_simulation(event):
    global origin_row_str, origin_col_str, multiplier, base_period, ani, flash_time
    origin_row_str = "3"
    origin_col_str = "3"
    multiplier = 0.1
    base_period = str(2 * math.pi)
    text_box_row.set_val("3")
    text_box_col.set_val("3")
    text_box_period.set_val(str(2 * math.pi))
    slider.set_val(0.1)
    flash_time = 0
    ani.frame_seq = ani.new_frame_seq()

# Initialize global variables
pi = math.pi
c = 299792
unit = "AU"
origin_row_str = "3"
origin_col_str = "3"
multiplier = 0.1
base_period = str(2 * math.pi)
flash_time = 0
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
colors = ['gray', 'orange', 'blue', 'red', 'brown', 'gold', 'cyan', 'darkblue']

# Create figure
fig = plt.figure(figsize=(12, 10))
ax = plt.axes()
ax.set_aspect('equal')

# TextBox for origin
ax_origin_row = plt.axes([0.1, 0.02, 0.15, 0.05])
ax_origin_col = plt.axes([0.3, 0.02, 0.15, 0.05])
text_box_row = TextBox(ax_origin_row, 'Origin Row (1-8): ', initial="3")
text_box_col = TextBox(ax_origin_col, 'Origin Col (1-8): ', initial="3")
text_box_row.on_submit(submit_origin_row)
text_box_col.on_submit(submit_origin_col)

# TextBox for base oscillator period
ax_period = plt.axes([0.5, 0.08, 0.15, 0.05])
text_box_period = TextBox(ax_period, 'Base Period (s): ', initial=str(2 * math.pi))
text_box_period.on_submit(submit_base_period)

# Slider for multiplier
ax_slider = plt.axes([0.5, 0.02, 0.3, 0.03])
slider = Slider(ax_slider, 'Compression Multiplier', 0.0, 1.0, valinit=0.1)
slider.on_changed(update_multiplier)

# Reset button
ax_reset = plt.axes([0.85, 0.02, 0.1, 0.05])
button = Button(ax_reset, 'Reset')
button.on_clicked(reset_simulation)

# Animation
def frame_generator():
    t = 0
    dt = 0.05
    while True:
        yield t
        t += dt

ani = FuncAnimation(fig, update_plot, frames=frame_generator, interval=50, blit=False, cache_frame_data=False)

plt.show()