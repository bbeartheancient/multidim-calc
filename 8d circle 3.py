import pygame
import math
import random
import sys

print("Starting script...")

# Initialize Pygame
try:
    pygame.init()
    print("Pygame initialized successfully")
except Exception as e:
    print(f"Pygame initialization failed: {e}")
    sys.exit(1)

# Constants
canvas_width = 800
canvas_height = 800
max_dimension = 8
base_radius = 300
wave_speed = 5.0
simulation_fps = 60
amplitude_threshold = 0.01
resonance_threshold = 129.24
resonance_lock_threshold = 0.25
resonance_persist_threshold = 0.18
resonance_decay_buffer = 100
four_d_resonance_threshold = 192.3
observer_feedback_threshold = 162.6
observer_feedback_factor = 3.0
min_8d_radius = 280
trans_d_radius = (162.6 + 192.3) / 2
desync_penalty = 0.9
decoherence_threshold = 0.5
saturation_threshold = 500.0

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GRAY = (150, 150, 150)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GLOW = (255, 255, 0, 128)
NEAR_RESONANCE_GLOW = (255, 215, 0, 128)
ORBIT_GLOW = (200, 200, 200, 64)
SPARK_COLOR = (255, 255, 255, 192)
EXPANSION_WAVE = (100, 100, 255, 128)

# Carrier wave contributions
carrier_wave_contributions = {
    1: 0.7980, 2: 0.3990, 3: 0.1995, 4: 0.09975,
    5: 0.049875, 6: 0.0249375, 7: 0.01246875, 8: 0.006234375
}

# Logarithmic radii
try:
    radii = {}
    for d in range(1, max_dimension + 1):
        log_radius = 0 if d == 1 else math.log10(d)
        max_log_radius = math.log10(max_dimension)
        radii[d] = (log_radius / max_log_radius) * base_radius if max_log_radius != 0 else 0
        if d == 1:
            radii[d] = 0
    print("Radii calculated successfully")
except Exception as e:
    print(f"Error calculating radii: {e}")
    sys.exit(1)

# Oscillation parameters
frequency = 0.5
amplitude = 5.0
reflection_coeff = 0.8
reflection_phase = 5.09
base_amplitude = 0.18
time = 0
time_step = 1 / simulation_fps
modulate_frequency = True
modulation_period = 5.0
modulation_high = 100.0
modulation_low = 0.5

# 2D oscillation points
try:
    two_d_log = (math.log10(2) + math.log10(3)) / 2
    two_d_radius = (two_d_log / math.log10(max_dimension)) * base_radius
    three_d_radius = (math.log10(3) / math.log10(max_dimension)) * base_radius
    four_d_radius = (math.log10(4) / math.log10(max_dimension)) * base_radius
    two_d_radial_dist = two_d_radius / radii[max_dimension]
    two_d_angles = [i * math.pi / 4 for i in range(8)]  # 8 points at 45° intervals
    two_d_base_positions = [
        (canvas_width/2 + two_d_radius * math.cos(angle), canvas_height/2 + two_d_radius * math.sin(angle))
        for angle in two_d_angles
    ]
    two_d_colors = [BLUE, GREEN, PURPLE, ORANGE, BLUE, GREEN, PURPLE, ORANGE]
    two_d_labels = [f"Q{i+1}: {angle * 180 / math.pi}°" for i, angle in enumerate(two_d_angles)]
    two_d_densities = [random.uniform(0.8, 1.2) for _ in range(8)]
    two_d_frequencies = [random.uniform(0.9, 1.1) * frequency for _ in range(8)]
    phase_offsets = [random.uniform(0, 0.05) for _ in range(8)]
    log_time_scale = [1.0 + i * 0.2 for i in range(8)]
    print("2D points initialized successfully")
except Exception as e:
    print(f"Error initializing 2D points: {e}")
    sys.exit(1)

# 1D energy cluster
num_1d_energies = 128
one_d_densities = [random.uniform(0.8, 1.2) for _ in range(num_1d_energies)]
one_d_frequencies = [random.uniform(0.9, 1.1) for _ in range(num_1d_energies)]
one_d_phases = [random.uniform(0, 0.1) for _ in range(num_1d_energies)]

# Additional oscillation parameters
remove_zero_points = False
crossing_counts = [0] * 8
resonance_counts = [0] * 8
four_d_resonance_counts = [0] * 8
observer_feedback_counts = [0] * 8
dimensional_states = ["2D"] * 8
state_transition_steps = [0] * 8
total_steps = 0
max_ripple_amplitudes = {d: 0.0 for d in range(2, max_dimension + 1)}
max_osc_amplitudes = {d: 0.0 for d in range(1, max_dimension + 1)}
max_quad_osc_amplitudes = [0.0] * 8
energy_focus = {"points": 0.0, "boundaries": 0.0}
quad_variance = 0.0
cycle_state = "1D"
cycle_start_time = 0.0
cycle_duration = 1 / frequency
near_resonance_flags = [False] * 8
global_resonance = 0.0
decoherence = 0.0
saturation = 0.0
singularity_mode = False
expansion_wave_radius = 0.0
expansion_wave_active = False

# Slider parameters
slider_x = 650
slider_width = 10
slider_height = 100
slider_y_freq = 100
slider_y_amp = 220
slider_y_reflect = 340
slider_y_reflect_phase = 460
freq_range = (0.1, 2000)
amp_range = (0, 5.0)
reflect_range = (0, 1.0)
reflect_phase_range = (0, 2 * math.pi)

try:
    sliders = [
        {"type": "freq", "x": slider_x, "y": slider_y_freq, "handle_y": slider_y_freq + slider_height * (1 - (math.log10(frequency) - math.log10(freq_range[0])) / (math.log10(freq_range[1]) - math.log10(freq_range[0])))},
        {"type": "amp", "x": slider_x, "y": slider_y_amp, "handle_y": slider_y_amp + slider_height * (1 - (amplitude - amp_range[0]) / (amp_range[1] - amp_range[0]))},
        {"type": "reflect", "x": slider_x, "y": slider_y_reflect, "handle_y": slider_y_reflect + slider_height * (1 - (reflection_coeff - reflect_range[0]) / (reflect_range[1] - reflect_range[0]))},
        {"type": "reflect_phase", "x": slider_x, "y": slider_y_reflect_phase, "handle_y": slider_y_reflect_phase + slider_height * (1 - (reflection_phase - reflect_phase_range[0]) / (reflect_phase_range[1] - reflect_phase_range[0]))}
    ]
    print("Sliders initialized successfully")
except Exception as e:
    print(f"Error initializing sliders: {e}")
    sys.exit(1)

# Setup display
try:
    screen = pygame.display.set_mode((canvas_width, canvas_height))
    pygame.display.set_caption("1D-8D Dark Matter Oscillation with Cosmic Evolution")
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 18)
    print("Display set up successfully")
except Exception as e:
    print(f"Error setting up display: {e}")
    sys.exit(1)

# Phase plot surface
plot_width = 200
plot_height = 100
plot_surface = pygame.Surface((plot_width, plot_height))
plot_x = 10
plot_y = canvas_height - plot_height - 10
print("Phase plot surface created")

# Slider dragging state
dragging_slider = None

# Main loop
print("Entering main loop...")
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            for slider in sliders:
                sx, sy = slider["x"], slider["y"]
                if sx <= x <= sx + slider_width and sy <= y <= sy + slider_height:
                    dragging_slider = slider
                    dragging_slider["handle_y"] = max(sy, min(sy + slider_height, y))
        elif event.type == pygame.MOUSEBUTTONUP:
            dragging_slider = None
        elif event.type == pygame.MOUSEMOTION and dragging_slider:
            x, y = event.pos
            dragging_slider["handle_y"] = max(dragging_slider["y"], min(dragging_slider["y"] + slider_height, y))
            norm_pos = 1 - (dragging_slider["handle_y"] - dragging_slider["y"]) / slider_height
            if dragging_slider["type"] == "freq" and not modulate_frequency:
                log_min, log_max = math.log10(freq_range[0]), math.log10(freq_range[1])
                log_value = log_min + norm_pos * (log_max - log_min)
                frequency = 10 ** log_value
                cycle_duration = 1 / frequency
                for i in range(8):
                    two_d_frequencies[i] = random.uniform(0.9, 1.1) * frequency
            elif dragging_slider["type"] == "amp":
                amplitude = amp_range[0] + norm_pos * (amp_range[1] - amp_range[0])
            elif dragging_slider["type"] == "reflect":
                reflection_coeff = reflect_range[0] + norm_pos * (reflect_range[1] - reflect_range[0])
            elif dragging_slider["type"] == "reflect_phase":
                reflection_phase = reflect_phase_range[0] + norm_pos * (reflect_phase_range[1] - reflect_phase_range[0])
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                remove_zero_points = not remove_zero_points
            elif event.key == pygame.K_c:
                dimensional_states = ["2D"] * 8
                crossing_counts = [0] * 8
                resonance_counts = [0] * 8
                four_d_resonance_counts = [0] * 8
                observer_feedback_counts = [0] * 8
                state_transition_steps = [0] * 8
                cycle_state = "1D"
                cycle_start_time = time
                phase_offsets = [random.uniform(0, 0.05) for _ in range(8)]
                one_d_phases = [random.uniform(0, 0.1) for _ in range(num_1d_energies)]
                singularity_mode = False
            elif event.key == pygame.K_m:
                modulate_frequency = not modulate_frequency
                if not modulate_frequency:
                    frequency = modulation_low
                    cycle_duration = 1 / frequency
                    for i in range(8):
                        two_d_frequencies[i] = random.uniform(0.9, 1.1) * frequency
                    for slider in sliders:
                        if slider["type"] == "freq":
                            slider["handle_y"] = slider["y"] + slider_height * (1 - (math.log10(frequency) - math.log10(freq_range[0])) / (math.log10(freq_range[1]) - math.log10(freq_range[0])))
            elif event.key == pygame.K_s:
                singularity_mode = not singularity_mode
                if singularity_mode:
                    one_d_phases = [0.0] * num_1d_energies  # Synchronize for singularity
                    cycle_start_time = time
                    expansion_wave_radius = 0.0
                    expansion_wave_active = True

    # Frequency modulation
    if modulate_frequency:
        t_mod = (time % modulation_period) / modulation_period
        frequency = modulation_high * (1 - t_mod) + modulation_low * t_mod
        cycle_duration = 1 / frequency
        for i in range(8):
            two_d_frequencies[i] = random.uniform(0.9, 1.1) * frequency
        for slider in sliders:
            if slider["type"] == "freq":
                slider["handle_y"] = slider["y"] + slider_height * (1 - (math.log10(frequency) - math.log10(freq_range[0])) / (math.log10(freq_range[1]) - math.log10(freq_range[0])))

    # Calculate 1D cluster oscillation
    one_d_amplitude = 0.0
    one_d_wave = 0.0
    phase_diffs = []
    for i in range(num_1d_energies):
        phase = 2 * math.pi * frequency * one_d_frequencies[i] * (time - cycle_start_time) + one_d_phases[i]
        one_d_amplitude += base_amplitude * one_d_densities[i] / num_1d_energies
        one_d_wave += math.sin(phase) / num_1d_energies
        phase_diffs.append(phase % (2 * math.pi))
    mean_phase = sum(phase_diffs) / num_1d_energies
    decoherence = sum(abs((p - mean_phase) % (2 * math.pi) - (0 if abs((p - mean_phase) % (2 * math.pi)) < math.pi else 2 * math.pi)) for p in phase_diffs) / num_1d_energies
    if singularity_mode:
        decoherence += 0.01  # Accelerate decoherence

    # Update cycle state
    cycle_progress = (time - cycle_start_time) % cycle_duration
    cycle_phase = 2 * math.pi * cycle_progress / cycle_duration
    if cycle_progress < cycle_duration:
        if math.sin(cycle_phase) >= 0:
            cycle_state = "1D"
        else:
            cycle_state = "8D"
    else:
        cycle_start_time += cycle_duration
        cycle_state = "1D"

    # Check for 1D-to-2D transition
    if decoherence > decoherence_threshold and random.random() < decoherence / math.pi:
        dimensional_states = ["2D"] * 8
        phase_offsets = [random.uniform(0, 0.05) for _ in range(8)]
        expansion_wave_active = True
        expansion_wave_radius = 0.0

    # Clear screen
    screen.fill(WHITE)

    # Draw simulation
    surface = pygame.Surface((canvas_width, canvas_height), pygame.SRCALPHA)

    # Draw expansion wave
    if expansion_wave_active:
        pygame.draw.circle(surface, EXPANSION_WAVE, (canvas_width/2, canvas_height/2), expansion_wave_radius, 3)
        expansion_wave_radius += 2.0
        if expansion_wave_radius > base_radius:
            expansion_wave_active = False

    # Draw orbital rings
    pygame.draw.circle(surface, ORBIT_GLOW, (canvas_width/2, canvas_height/2), three_d_radius, 1)
    pygame.draw.circle(surface, ORBIT_GLOW, (canvas_width/2, canvas_height/2), four_d_radius, 1)
    if any(near_resonance_flags):
        pulse_alpha = int(128 * (1 + math.sin(2 * math.pi * time * 2)) / 2)
        orbit_color = (200, 200, 200, pulse_alpha)
        pygame.draw.circle(surface, orbit_color, (canvas_width/2, canvas_height/2), three_d_radius, 2)
        pygame.draw.circle(surface, orbit_color, (canvas_width/2, canvas_height/2), four_d_radius, 2)

    # Draw quadrant markers
    quadrant_labels = ["Q1: 0°", "Q2: 90°", "Q3: 180°", "Q4: 270°"]
    quadrant_colors = [BLUE, GREEN, PURPLE, ORANGE]
    for i, angle in enumerate([0, math.pi/2, math.pi, 3*math.pi/2]):
        x_end = canvas_width/2 + base_radius * 1.1 * math.cos(angle)
        y_end = canvas_height/2 + base_radius * 1.1 * math.sin(angle)
        pygame.draw.line(surface, quadrant_colors[i], (canvas_width/2, canvas_height/2), (x_end, y_end), 2)
        label = font.render(quadrant_labels[i], True, quadrant_colors[i])
        label_rect = label.get_rect(center=(canvas_width/2 + (base_radius + 30) * math.cos(angle), canvas_height/2 + (base_radius + 30) * math.sin(angle)))
        screen.blit(label, label_rect)

    # Draw coordinate guide
    guide_label = small_font.render("Upper-Right: Q1 (Blue), Upper-Left: Q2 (Green)", True, BLACK)
    screen.blit(guide_label, (10, 70))
    guide_label2 = small_font.render("Lower-Left: Q3 (Purple), Lower-Right: Q4 (Orange)", True, BLACK)
    screen.blit(guide_label2, (10, 90))

    # Draw 3D boundary glow
    glow_active = False
    two_d_displacements = []
    resonance_transitions = [False] * 8
    saturation = 0.0
    for i, (base_x, base_y) in enumerate(two_d_base_positions):
        scaled_time = (time - cycle_start_time) / log_time_scale[i]
        two_d_wave_value = amplitude * math.sin(2 * math.pi * two_d_frequencies[i] * scaled_time + phase_offsets[i])
        two_d_amplitude = base_amplitude * abs(two_d_wave_value) * (carrier_wave_contributions[1] / carrier_wave_contributions[max_dimension]) * two_d_densities[i]
        two_d_displacement = two_d_amplitude * math.sin(2 * math.pi * two_d_frequencies[i] * scaled_time + phase_offsets[i])
        two_d_reflect = reflection_coeff * two_d_amplitude * math.sin(2 * math.pi * two_d_frequencies[i] * scaled_time + wave_speed * two_d_radial_dist + reflection_phase)
        two_d_displacement = (two_d_displacement + two_d_reflect) / (1 + reflection_coeff)
        two_d_displacements.append(two_d_displacement)
        near_resonance_flags[i] = (abs(two_d_displacement) >= 0.9 * resonance_threshold and abs(two_d_displacement) < resonance_threshold) or \
                                  (abs(two_d_displacement) >= 0.9 * four_d_resonance_threshold and abs(two_d_displacement) < four_d_resonance_threshold)
        saturation += abs(two_d_displacement) * two_d_densities[i]
        if abs(two_d_displacement) >= observer_feedback_threshold:
            glow_active = True
    if glow_active:
        pygame.draw.circle(surface, GLOW, (canvas_width/2, canvas_height/2), three_d_radius, 3)

    # Apply desynchronization penalty
    resonating_points = sum(1 for d in two_d_displacements if abs(d) >= resonance_threshold)
    amplitude_factor = desync_penalty if resonating_points >= 5 else 1.0

    # Draw 1D radial oscillation
    one_d_radius = one_d_amplitude * one_d_wave * amplitude_factor if cycle_state == "1D" else 0
    one_d_radius = min(one_d_radius, radii[2])
    max_osc_amplitudes[1] = max(max_osc_amplitudes[1], one_d_radius)
    pygame.draw.circle(surface, BLACK, (canvas_width/2, canvas_height/2), max(5, int(one_d_radius)), 1)

    # Draw 2D oscillation points
    total_steps += 1
    point_energy = 0.0
    quad_amplitudes = []
    global_resonance = 0.0
    for i, (base_x, base_y) in enumerate(two_d_base_positions):
        two_d_displacement = two_d_displacements[i] * amplitude_factor
        base_radius_i = two_d_radius
        prev_state = dimensional_states[i]
        if dimensional_states[i] == "3D+":
            base_radius_i = trans_d_radius
            origin_color = list(two_d_colors[i]) + [128]
            origin_x = canvas_width/2 + trans_d_radius * math.cos(two_d_angles[i])
            origin_y = canvas_height/2 + trans_d_radius * math.sin(two_d_angles[i])
            pygame.draw.circle(surface, origin_color, (origin_x, origin_y), 4)
        elif dimensional_states[i] == "4D+":
            base_radius_i = four_d_radius
            origin_color = list(two_d_colors[i]) + [64]
            origin_x = canvas_width/2 + four_d_radius * math.cos(two_d_angles[i])
            origin_y = canvas_height/2 + four_d_radius * math.sin(two_d_angles[i])
            pygame.draw.circle(surface, origin_color, (origin_x, origin_y), 5)
        else:
            origin_x = base_x
            origin_y = base_y
        radial_distance = base_radius_i + two_d_displacement * 1.5
        radial_distance = max(min(radial_distance, base_radius_i * 2.0), base_radius_i * 0.3)
        if dimensional_states[i] == "3D+":
            radial_distance = max(radial_distance, three_d_radius)
        elif dimensional_states[i] == "4D+":
            radial_distance = max(radial_distance, four_d_radius)
        x = canvas_width/2 + radial_distance * math.cos(two_d_angles[i])
        y = canvas_height/2 + radial_distance * math.sin(two_d_angles[i])
        max_quad_osc_amplitudes[i] = max(max_quad_osc_amplitudes[i], abs(two_d_displacement))
        quad_amplitudes.append(max_quad_osc_amplitudes[i])
        point_energy += two_d_displacement ** 2
        global_resonance += abs(two_d_displacement) * two_d_densities[i]
        two_d_wave_value = amplitude * math.sin(2 * math.pi * two_d_frequencies[i] * (time - cycle_start_time) + phase_offsets[i])
        if abs(two_d_displacement) >= resonance_threshold:
            resonance_counts[i] += 1
        if abs(two_d_displacement) >= observer_feedback_threshold:
            resonance_counts[i] += observer_feedback_factor - 1
            observer_feedback_counts[i] += 1
        if abs(two_d_displacement) >= four_d_resonance_threshold:
            four_d_resonance_counts[i] += 1
        if abs(two_d_displacement) > 0.99 * two_d_radius:
            crossing_counts[i] += 1
        if total_steps > 0:
            resonance_ratio = resonance_counts[i] / total_steps
            four_d_resonance_ratio = four_d_resonance_counts[i] / total_steps
            if dimensional_states[i] == "2D" and resonance_ratio > resonance_lock_threshold:
                dimensional_states[i] = "3D+"
                resonance_transitions[i] = True
                state_transition_steps[i] = 0
            elif dimensional_states[i] == "3D+" and four_d_resonance_ratio > resonance_lock_threshold:
                dimensional_states[i] = "4D+"
                resonance_transitions[i] = True
                state_transition_steps[i] = 0
            elif dimensional_states[i] == "4D+" and four_d_resonance_ratio < resonance_persist_threshold:
                state_transition_steps[i] += 1
                if state_transition_steps[i] > resonance_decay_buffer:
                    dimensional_states[i] = "3D+"
                    state_transition_steps[i] = 0
            elif dimensional_states[i] == "3D+" and resonance_ratio < resonance_persist_threshold:
                state_transition_steps[i] += 1
                if state_transition_steps[i] > resonance_decay_buffer:
                    dimensional_states[i] = "2D"
                    state_transition_steps[i] = 0
        if near_resonance_flags[i]:
            phase_offsets[i] += 0.001 * two_d_densities[i]
            phase_offsets[i] = phase_offsets[i] % (2 * math.pi)
        point_size = 8 if dimensional_states[i] == "3D+" else 10 if dimensional_states[i] == "4D+" else 5
        if not remove_zero_points or abs(two_d_wave_value) >= amplitude_threshold:
            if near_resonance_flags[i]:
                pygame.draw.circle(surface, NEAR_RESONANCE_GLOW, (x, y), point_size + 3, 2)
            if resonance_transitions[i]:
                pygame.draw.circle(surface, SPARK_COLOR, (x, y), point_size + 5, 3)
                resonance_transitions[i] = False
            pygame.draw.circle(surface, two_d_colors[i], (x, y), point_size)
            if dimensional_states[i] in ["3D+", "4D+"]:
                label = small_font.render(dimensional_states[i], True, two_d_colors[i])
                label_rect = label.get_rect(center=(x + 15, y))
                surface.blit(label, label_rect)

    # Desynchronization
    if quad_amplitudes:
        mean_amp = sum(quad_amplitudes) / len(quad_amplitudes)
        quad_variance = sum((amp - mean_amp) ** 2 for amp in quad_amplitudes) / len(quad_amplitudes)

    # Draw 2D to 8D circles
    boundary_energy = 0.0
    for d in range(2, max_dimension + 1):
        base_r = radii[d]
        color = RED if d == max_dimension else BLACK
        scaled_amplitude = base_amplitude * (carrier_wave_contributions[d] / carrier_wave_contributions[max_dimension]) * amplitude_factor
        points = []
        ripple_amplitude = 0.0
        osc_amplitude = 0.0
        for angle in range(0, 3600, 2):
            angle_rad = math.radians(angle / 10)
            radial_dist = base_r / radii[max_dimension]
            wave_effect = math.sin(2 * math.pi * frequency * (time - cycle_start_time) - wave_speed * radial_dist) + \
                          reflection_coeff * math.sin(2 * math.pi * frequency * (time - cycle_start_time) + wave_speed * radial_dist + reflection_phase)
            wave_effect /= (1 + reflection_coeff)
            if d == max_dimension and cycle_state == "8D":
                dark_matter_effect = 2 * one_d_amplitude * one_d_wave * (carrier_wave_contributions[1] / carrier_wave_contributions[max_dimension])
                wave_effect += dark_matter_effect
            r = base_r + scaled_amplitude * wave_effect
            if d == max_dimension:
                r = max(r, min_8d_radius)
            ripple_amplitude = max(ripple_amplitude, abs(wave_effect * scaled_amplitude))
            osc_amplitude = max(osc_amplitude, abs(r - base_r))
            boundary_energy += (r - base_r) ** 2
            x = r * math.cos(angle_rad) + canvas_width / 2
            y = r * math.sin(angle_rad) + canvas_height / 2
            points.append((x, y))
        max_ripple_amplitudes[d] = max(max_ripple_amplitudes[d], ripple_amplitude)
        max_osc_amplitudes[d] = max(max_osc_amplitudes[d], osc_amplitude)
        pygame.draw.polygon(surface, color, points, 1)
        label = font.render(f"{d}D", True, BLACK)
        label_rect = label.get_rect(center=(canvas_width / 2 + base_r + 20, canvas_height / 2))
        screen.blit(label, label_rect)
        if d == max_dimension and cycle_state == "8D":
            dm_label = small_font.render("8D Dark Matter", True, RED)
            dm_label_rect = label.get_rect(center=(canvas_width/2 + base_r + 20, canvas_height/2 + 20))
            surface.blit(dm_label, dm_label_rect)

    # Energy focus
    energy_focus["points"] = point_energy / 8
    energy_focus["boundaries"] = boundary_energy / (3600 // 2 * (max_dimension - 1))

    # Draw 1D label
    label = font.render("1D", True, BLACK)
    label_rect = label.get_rect(center=(canvas_width / 2 + 20, canvas_height / 2))
    screen.blit(label, label_rect)

    # Draw cycle state
    label = font.render(f"Cycle State: {cycle_state}", True, BLACK)
    screen.blit(label, (10, 110))

    # Draw sliders
    for slider in sliders:
        sx, sy = slider["x"], slider["y"]
        pygame.draw.rect(screen, GRAY, (sx, sy, slider_width, slider_height))
        pygame.draw.rect(screen, BLUE, (sx, slider["handle_y"] - 5, slider_width, 10))
        if slider["type"] == "freq":
            label = font.render(f"Freq: {frequency:.1f} Hz", True, BLACK)
        elif slider["type"] == "amp":
            label = font.render(f"Amp: {amplitude:.2f}", True, BLACK)
        elif slider["type"] == "reflect":
            label = font.render(f"Reflect: {reflection_coeff:.2f}", True, BLACK)
        elif slider["type"] == "reflect_phase":
            label = font.render(f"Refl Phase: {reflection_phase:.2f} rad", True, BLACK)
        screen.blit(label, (sx + 15, sy + slider_height + 10))

    # Draw toggle status
    toggle_label = font.render(f"Remove Zero Points: {'ON' if remove_zero_points else 'OFF'} (Press R)", True, BLACK)
    screen.blit(toggle_label, (10, 10))
    reset_label = font.render("Reset States: Press C", True, BLACK)
    screen.blit(reset_label, (10, 30))
    mod_label = font.render(f"Modulate Freq: {'ON' if modulate_frequency else 'OFF'} (Press M)", True, BLACK)
    screen.blit(mod_label, (10, 50))
    singularity_label = font.render(f"Singularity Mode: {'ON' if singularity_mode else 'OFF'} (Press S)", True, BLACK)
    screen.blit(singularity_label, (10, 130))

    # Draw metrics
    for i, count in enumerate(crossing_counts):
        prob = (count / total_steps * 100) if total_steps > 0 else 0
        label = font.render(f"{two_d_labels[i]} Crossing: {prob:.1f}%", True, two_d_colors[i])
        screen.blit(label, (220, 690 + i * 15))
    for i, d in enumerate(range(1, max_dimension + 1)):
        label = font.render(f"{d}D Osc: {max_osc_amplitudes[d]:.3f}", True, BLACK)
        screen.blit(label, (220, 610 + i * 15))
    for i, d in enumerate(range(2, max_dimension + 1)):
        label = font.render(f"{d}D Ripple: {max_ripple_amplitudes[d]:.3f}", True, BLACK)
        screen.blit(label, (220, 650 + i * 15))
    for i in range(8):
        label = font.render(f"{two_d_labels[i]} Osc: {max_quad_osc_amplitudes[i]:.3f}", True, two_d_colors[i])
        screen.blit(label, (350, 690 + i * 15))
    for i, count in enumerate(resonance_counts):
        prob = (count / total_steps * 100) if total_steps > 0 else 0
        label = font.render(f"{two_d_labels[i]} 3D Res: {prob:.1f}%", True, two_d_colors[i])
        screen.blit(label, (480, 690 + i * 15))
    for i, count in enumerate(four_d_resonance_counts):
        prob = (count / total_steps * 100) if total_steps > 0 else 0
        label = font.render(f"{two_d_labels[i]} 4D Res: {prob:.1f}%", True, two_d_colors[i])
        screen.blit(label, (480, 750 + i * 15))
    for i in range(8):
        label = font.render(f"{two_d_labels[i]} State: {dimensional_states[i]}", True, two_d_colors[i])
        screen.blit(label, (610, 690 + i * 15))
    for i, count in enumerate(resonance_counts):
        prob = (count / total_steps * 100 / resonance_lock_threshold) if total_steps > 0 else 0
        prob = min(prob, 100.0)
        label = font.render(f"{two_d_labels[i]} Res Prog: {prob:.1f}%", True, two_d_colors[i])
        screen.blit(label, (610, 750 + i * 15))
    for i, count in enumerate(observer_feedback_counts):
        prob = (count / total_steps * 100) if total_steps > 0 else 0
        label = font.render(f"{two_d_labels[i]} Obs Feedback: {prob:.1f}%", True, two_d_colors[i])
        screen.blit(label, (740, 690 + i * 15))
    potential_3d_points = sum(8 if state == "3D+" or state == "4D+" else 0 for state in dimensional_states)
    potential_4d_points = sum(16 if state == "4D+" else 0 for state in dimensional_states)
    label = font.render(f"Potential 3D Points: {potential_3d_points}", True, BLACK)
    screen.blit(label, (220, 570))
    label = font.render(f"Potential 4D Points: {potential_4d_points}", True, BLACK)
    screen.blit(label, (220, 590))
    label = font.render(f"Global Resonance: {global_resonance:.2f}", True, BLACK)
    screen.blit(label, (220, 550))
    label = font.render(f"1D Decoherence: {decoherence:.2f}", True, BLACK)
    screen.blit(label, (220, 530))
    label = font.render(f"2D Saturation: {saturation:.2f}", True, BLACK)
    screen.blit(label, (220, 510))

    # Draw phase plot
    plot_surface.fill(WHITE)
    pygame.draw.line(plot_surface, BLACK, (0, plot_height/2), (plot_width, plot_height/2), 1)
    pygame.draw.line(plot_surface, BLACK, (plot_width/2, 0), (plot_width/2, plot_height), 1)
    prev_y_two_d = [0] * 8
    time_window = 2.0
    max_plot_amplitude = max(one_d_amplitude, base_amplitude * amplitude * (carrier_wave_contributions[1] / carrier_wave_contributions[max_dimension]), four_d_radius)
    for t in range(plot_width):
        plot_time = time - time_window * (1 - t / plot_width)
        one_d_value = one_d_wave * one_d_amplitude / max_plot_amplitude
        y_one_d = plot_height/2 - one_d_value * (plot_height/2 - 5)
        if t > 0:
            pygame.draw.line(plot_surface, BLACK, (t-1, prev_y_one_d), (t, y_one_d), 2)
        prev_y_one_d = y_one_d
        for i in range(8):
            scaled_plot_time = (plot_time - cycle_start_time) / log_time_scale[i]
            two_d_wave_value = amplitude * math.sin(2 * math.pi * two_d_frequencies[i] * scaled_plot_time + phase_offsets[i])
            if abs(two_d_wave_value) < amplitude_threshold:
                continue
            two_d_amplitude = base_amplitude * abs(two_d_wave_value) * (carrier_wave_contributions[1] / carrier_wave_contributions[max_dimension]) * two_d_densities[i]
            two_d_reflect = reflection_coeff * two_d_amplitude * math.sin(2 * math.pi * two_d_frequencies[i] * scaled_plot_time + wave_speed * two_d_radial_dist + reflection_phase)
            two_d_value = (math.sin(2 * math.pi * two_d_frequencies[i] * scaled_plot_time + phase_offsets[i]) * two_d_amplitude + two_d_reflect) / (1 + reflection_coeff)
            if dimensional_states[i] == "3D+":
                two_d_value = trans_d_radius + two_d_value
                two_d_value = max(two_d_value, three_d_radius)
            elif dimensional_states[i] == "4D+":
                two_d_value = four_d_radius + two_d_value
                two_d_value = max(two_d_value, four_d_radius)
            two_d_value /= max_plot_amplitude
            y_two_d = plot_height/2 - two_d_value * (plot_height/2 - 5)
            if t > 0:
                pygame.draw.line(plot_surface, two_d_colors[i], (t-1, prev_y_two_d[i]), (t, y_two_d), 1)
            prev_y_two_d[i] = y_two_d
    label_1d = font.render("1D", True, BLACK)
    plot_surface.blit(label_1d, (5, 5))
    for i, label_text in enumerate(two_d_labels):
        label = font.render(label_text, True, two_d_colors[i])
        plot_surface.blit(label, (5, 20 + i * 15))
    screen.blit(plot_surface, (plot_x, plot_y))

    screen.blit(surface, (0, 0))

    # Update time
    time += time_step

    # Update display
    try:
        pygame.display.flip()
    except Exception as e:
        print(f"Error updating display: {e}")
        running = False

    clock.tick(simulation_fps)

# Cleanup
print("Exiting script...")
pygame.quit()
sys.exit()