import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import RadioButtons, TextBox
from collections import Counter
import numpy as np
import matplotlib.colors as mcolors

# Define the provided 8x8 table
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
    """Flip the columns of the table: (i,j) -> (i, n-1-j)."""
    n = len(table)
    flipped_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            flipped_table[i][n-1-j] = table[i][j]
    return flipped_table

# Use flipped table
original_table = flip_table_columns(original_table)

def scale_table(table, origin_row, origin_col, target_value):
    """Scale the table to set the origin point to the target value."""
    try:
        origin_value = table[origin_row][origin_col]
        if origin_value == 0:
            raise ValueError("Origin value cannot be zero.")
        scaling_factor = target_value / origin_value
        scaled_table = [[value * scaling_factor for value in row] for row in table]
        return scaled_table, scaling_factor
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid target value or origin: {e}")

def get_dominant_values(scaled_table):
    """Find the most dominant value (highest frequency or largest if tied) per dimension."""
    dominant_values = []
    for i, row in enumerate(scaled_table):
        value_counts = Counter(round(value, 6) for value in row)
        max_freq = max(value_counts.values())
        max_freq_values = [value for value, count in value_counts.items() if count == max_freq]
        dominant_value = max(max_freq_values)
        for j, value in enumerate(row):
            if round(value, 6) == dominant_value:
                dominant_values.append((i + 1, j + 1, dominant_value))
                break
    return dominant_values

def get_frequency_sum(scaled_table):
    """Get the sum of the value with the highest frequency."""
    flat_values = [round(value, 6) for row in scaled_table for value in row]
    value_counts = Counter(flat_values)
    most_common_value = max(value_counts.items(), key=lambda x: x[1])[0]
    frequency = value_counts[most_common_value]
    return most_common_value * frequency

def lerp(a, b, t):
    """Linear interpolation between a and b by factor t."""
    return a + t * (b - a)

def get_color_and_linewidth(i, j, scaled_value, scaled_origin_value, origin_row, origin_col, 
                           scaled_table, origin_color, larger_color, smaller_color, is_origin2=False):
    """Determine color with fade from origin_color to larger/smaller_color, darker farther away."""
    scaled_value_rounded = round(scaled_value, 6)
    scaled_origin_rounded = round(scaled_origin_value, 6)
    
    if i == origin_row and j == origin_col:
        return origin_color, True
    
    distance = abs(scaled_value - scaled_origin_value)
    max_distance = max(abs(v - scaled_origin_value) for row in scaled_table for v in row)
    norm_distance = 0 if max_distance == 0 else distance / max_distance
    brightness = 1 - norm_distance
    
    if scaled_value_rounded > scaled_origin_rounded:
        r = lerp(origin_color[0], larger_color[0], norm_distance) * brightness
        g = lerp(origin_color[1], larger_color[1], norm_distance) * brightness
        b = lerp(origin_color[2], larger_color[2], norm_distance) * brightness
    else:
        r = lerp(origin_color[0], smaller_color[0], norm_distance) * brightness
        g = lerp(origin_color[1], smaller_color[1], norm_distance) * brightness
        b = lerp(origin_color[2], smaller_color[2], norm_distance) * brightness
    return (r, g, b), is_origin2

def update(val=None):
    """Update the plots based on current origins and color selections."""
    try:
        target_value1 = float(target_value1_str)
        target_value2 = float(target_value2_str)
    except ValueError:
        target_value1 = math.pi
        target_value2 = math.pi
        print("Invalid target value(s); using pi as default.")
    
    ax1.clear()
    
    scaled_table1, scaling_factor1 = scale_table(original_table, origin_row1, origin_col1, target_value1)
    scaled_table2, scaling_factor2 = scale_table(original_table, origin_row2, origin_col2, target_value2)
    scaled_origin_value1 = target_value1
    scaled_origin_value2 = target_value2
    
    # Compute differences for the plot
    differences = []
    for i in range(8):
        for j in range(8):
            diff = abs(scaled_table1[i][j] - scaled_table2[i][j])
            differences.append(diff)
    
    # Polar plot
    max_value1 = max(max(row) for row in scaled_table1)
    freq_sum1 = get_frequency_sum(scaled_table1)
    radius_ref1 = max_value1 if max_value1 > freq_sum1 else freq_sum1
    radius_scale1 = 10 / radius_ref1 if radius_ref1 != 0 else 1
    
    max_value2 = max(max(row) for row in scaled_table2)
    freq_sum2 = get_frequency_sum(scaled_table2)
    radius_ref2 = max_value2 if max_value2 > freq_sum2 else freq_sum2
    radius_scale2 = 10 / radius_ref2 if radius_ref2 != 0 else 1
    
    circles1 = []
    circles2 = []
    origin_radius1 = None
    origin_radius2 = None
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    
    for i in range(8):
        for j in range(8):
            idx = i * 8 + j
            angle = theta[idx]
            
            # Origin 1 circles
            scaled_value1 = scaled_table1[i][j]
            color1, is_origin1 = get_color_and_linewidth(i, j, scaled_value1, scaled_origin_value1, 
                                                        origin_row1, origin_col1, scaled_table1,
                                                        color_assignments['='], 
                                                        color_assignments['>'], 
                                                        color_assignments['<'], False)
            radius1 = scaled_value1 * radius_scale1
            if is_origin1:
                origin_radius1 = radius1
            circles1.append({
                'theta': angle,
                'radius': radius1,
                'color': color1,
                'label': f'({i+1}D, {j+1}D): {scaled_value1:.6f}'
            })
            
            # Origin 2 circles
            scaled_value2 = scaled_table2[i][j]
            color2, is_origin2 = get_color_and_linewidth(i, j, scaled_value2, scaled_origin_value2, 
                                                        origin_row2, origin_col2, scaled_table2,
                                                        color_assignments['='], 
                                                        color_assignments['>'], 
                                                        color_assignments['<'], True)
            radius2 = scaled_value2 * radius_scale2
            if is_origin2:
                origin_radius2 = radius2
            circles2.append({
                'theta': angle,
                'radius': radius2,
                'color': color2,
                'label': f'({i+1}D, {j+1}D): {scaled_value2:.6f}'
            })
    
    circles1.sort(key=lambda c: c['radius'], reverse=True)
    circles2.sort(key=lambda c: c['radius'], reverse=True)
    
    for circle in circles1:
        ax1.add_patch(plt.Circle((0, 0), circle['radius'], 
                                facecolor=circle['color'], edgecolor=None, alpha=0.6))
    
    for circle in circles2:
        ax1.add_patch(plt.Circle((0, 0), circle['radius'], 
                                facecolor=circle['color'], edgecolor='black', 
                                linestyle='--', alpha=0.3))
    
    ax1.plot(theta, differences, color='magenta', linewidth=2, label='|Origin 1 - Origin 2|')
    
    if origin_radius1 is not None:
        ax1.add_patch(plt.Circle((0, 0), origin_radius1, 
                                facecolor=None, edgecolor='magenta', linewidth=1.5, fill=False))
    if origin_radius2 is not None:
        ax1.add_patch(plt.Circle((0, 0), origin_radius2, 
                                facecolor=None, edgecolor='cyan', linewidth=1.5, linestyle='--', fill=False))
    
    # Rotate polar plot so 0 degrees points up
    ax1.set_theta_offset(np.pi / 2)
    
    ax1.set_title(f"Polar Plot: Origin 1 ({origin_row1+1}D, {origin_col1+1}D, {target_value1:.2f}), "
                  f"Origin 2 ({origin_row2+1}D, {origin_col2+1}D, {target_value2:.2f}) (in {unit})")
    ax1.set_xlabel("Angle (radians)")
    ax1.set_ylabel("Radius / Difference")
    ax1.legend()
    
    # Dominant values with sum and difference
    dominant_values1 = get_dominant_values(scaled_table1)
    dominant_values2 = get_dominant_values(scaled_table2)
    
    # Compute sum and difference of dominant values
    sums = []
    differences_dom = []
    for (i1, j1, val1), (i2, j2, val2) in zip(dominant_values1, dominant_values2):
        sum_val = val1 + val2
        diff_val = val1 - val2
        sums.append((i1, j1, sum_val))
        differences_dom.append((i1, j1, diff_val))
    
    text = ("Dominant Values (Origin 1):\n" + "\n".join(
        f"{i}D to {j}D: {value:.6f}" for i, j, value in dominant_values1
    ) + "\n\nDominant Values (Origin 2):\n" + "\n".join(
        f"{i}D to {j}D: {value:.6f}" for i, j, value in dominant_values2
    ) + "\n\nSum of Dominant Values:\n" + "\n".join(
        f"{i}D to {j}D: {value:.6f}" for i, j, value in sums
    ) + "\n\nDifference of Dominant Values (1 - 2):\n" + "\n".join(
        f"{i}D to {j}D: {value:.6f}" for i, j, value in differences_dom
    ))
    ax1.text(1.25, 0.5, text, transform=ax1.transAxes, fontsize=8,
             verticalalignment='center', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.canvas.draw_idle()

def update_origin1(label):
    """Update Origin 1 row and column to the same dimension."""
    global origin_row1, origin_col1
    dim = int(label.split('D')[0]) - 1
    origin_row1 = dim
    origin_col1 = dim
    update()

def update_origin2(label):
    """Update Origin 2 row and column to the same dimension."""
    global origin_row2, origin_col2
    dim = int(label.split('D')[0]) - 1
    origin_row2 = dim
    origin_col2 = dim
    update()

def update_larger_color(label):
    """Update color for > category."""
    color_assignments['>'] = color_options[label]
    update()

def update_origin_color(label):
    """Update color for = category."""
    color_assignments['='] = color_options[label]
    update()

def update_smaller_color(label):
    """Update color for < category."""
    color_assignments['<'] = color_options[label]
    update()

def submit_unit(text):
    """Update the unit from TextBox input."""
    global unit
    unit = text.strip() if text.strip() else "unitless"
    update()

def submit_target1(text):
    """Update target value for Origin 1."""
    global target_value1_str
    target_value1_str = text.strip() if text.strip() else str(math.pi)
    update()

def submit_target2(text):
    """Update target value for Origin 2."""
    global target_value2_str
    target_value2_str = text.strip() if text.strip() else str(math.pi)
    update()

# Main execution
try:
    # Initialize global variables
    unit = "unitless"
    target_value1_str = str(math.pi)  # Default to pi
    target_value2_str = str(math.pi)  # Default to pi
    origin_row1, origin_col1 = 0, 0  # Default Origin 1 (1D, 1D)
    origin_row2, origin_col2 = 2, 2  # Default Origin 2 (3D, 3D)
    
    color_options = {
        'Red': (1, 0, 0),
        'Blue': (0, 0, 1),
        'Green': (0, 1, 0),
        'Yellow': (1, 1, 0),
        'Cyan': (0, 1, 1),
        'Magenta': (1, 0, 1),
        'White': (1, 1, 1),
        'Black': (0, 0, 0)
    }
    
    color_assignments = {
        '>': (1, 0, 0),
        '=': (1, 1, 0),
        '<': (0, 0, 1)
    }
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    ax1 = fig.add_axes([0.1, 0.35, 0.5, 0.5], polar=True)  # Combined polar plot
    
    # Radio buttons for color selection
    ax_larger = plt.axes([0.1, 0.22, 0.15, 0.1])
    ax_origin = plt.axes([0.3, 0.22, 0.15, 0.1])
    ax_smaller = plt.axes([0.5, 0.22, 0.15, 0.1])
    radio_larger = RadioButtons(ax_larger, list(color_options.keys()), active=0)
    radio_origin = RadioButtons(ax_origin, list(color_options.keys()), active=3)
    radio_smaller = RadioButtons(ax_smaller, list(color_options.keys()), active=1)
    
    ax_larger.text(0.5, 1.2, 'Larger', transform=ax_larger.transAxes, 
                   ha='center', va='bottom', fontsize=10)
    ax_origin.text(0.5, 1.2, 'Origin', transform=ax_origin.transAxes, 
                   ha='center', va='bottom', fontsize=10)
    ax_smaller.text(0.5, 1.2, 'Smaller', transform=ax_smaller.transAxes, 
                    ha='center', va='bottom', fontsize=10)
    
    radio_larger.on_clicked(update_larger_color)
    radio_origin.on_clicked(update_origin_color)
    radio_smaller.on_clicked(update_smaller_color)
    
    # Radio buttons for Origin 1 and Origin 2 (single dimension)
    dimensions = [f"{i}D" for i in range(1, 9)]
    ax_origin1 = plt.axes([0.1, 0.12, 0.15, 0.08])
    ax_origin2 = plt.axes([0.3, 0.12, 0.15, 0.08])
    
    radio_origin1 = RadioButtons(ax_origin1, dimensions, active=0)  # 1D
    radio_origin2 = RadioButtons(ax_origin2, dimensions, active=2)  # 3D
    
    ax_origin1.text(0.5, 1.2, 'Origin 1', transform=ax_origin1.transAxes, 
                    ha='center', va='bottom', fontsize=10)
    ax_origin2.text(0.5, 1.2, 'Origin 2', transform=ax_origin2.transAxes, 
                    ha='center', va='bottom', fontsize=10)
    
    radio_origin1.on_clicked(update_origin1)
    radio_origin2.on_clicked(update_origin2)
    
    # TextBox for unit and target values
    ax_unit = plt.axes([0.1, 0.02, 0.15, 0.05])
    ax_target1 = plt.axes([0.3, 0.02, 0.15, 0.05])
    ax_target2 = plt.axes([0.5, 0.02, 0.15, 0.05])
    
    text_box_unit = TextBox(ax_unit, 'Unit: ', initial="unitless")
    text_box_target1 = TextBox(ax_target1, 'Origin 1 Value: ', initial=str(math.pi))
    text_box_target2 = TextBox(ax_target2, 'Origin 2 Value: ', initial=str(math.pi))
    
    text_box_unit.on_submit(submit_unit)
    text_box_target1.on_submit(submit_target1)
    text_box_target2.on_submit(submit_target2)
    
    update()
    
    plt.show()
    
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")