import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import RadioButtons, TextBox, Button
from collections import Counter
import numpy as np
import matplotlib.colors as mcolors
import os


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
def generate_blender_script(circles1, circles2, origin_radius1, origin_radius2, theta, differences, unit, target_value1, target_value2):
    """Generate a Blender Python script to animate 8 concentric spheres centered at the origin with scaled, overlapping radii, XYZ rotation, and custom material properties over 320 frames."""
    import numpy as np
    import math
    import os
    print("NumPy version:", np.__version__)
    
    # Validate inputs
    if not circles1 or not circles2 or len(circles1) != len(circles2) or len(circles1) < 8:
        raise ValueError("Invalid input: circles1 and circles2 must have at least 8 entries and equal length.")
    if len(theta) != len(differences):
        raise ValueError("theta and differences arrays must have equal length.")
    if origin_radius1 is None or origin_radius2 is None:
        print("Warning: origin_radius1 or origin_radius2 is None. Origin sphere may not be created.")
    
    # Convert differences to string for script
    differences_str = str(differences.tolist() if isinstance(differences, np.ndarray) else differences)
    
    # Check for zero differences
    if all(d == 0.0 for d in differences):
        print("Warning: All differences are zero. Animation will be static. Check Origin 1 and Origin 2 settings.")
    
    # Animation parameters
    osc_freq = abs(target_value1 - target_value2)
    osc_amplitude = 0.1
    fps = 24  # Integer for Blender
    light_intensity_factor = 100.0
    scale_factor = 0.5
    base_angular_speed = abs(target_value1) / 10 if target_value1 != 0 else 1.0
    print(f"Oscillation frequency: {osc_freq} Hz")
    print(f"Base angular speed: {base_angular_speed}")
    
    # Extract base radii from circles1
    base_radii = [c['radius'] for c in circles1]
    
    # Scale radii to ensure concentric overlap
    min_radius = min(base_radii)
    max_radius = max(base_radii)
    if max_radius == min_radius:
        scaled_radii = [1.0 + i * 0.5 for i in range(8)]  # Linear spacing if radii are equal
    else:
        # Normalize radii between 1.0 and 4.0 for visible overlap
        scaled_radii = [1.0 + 3.0 * (r - min_radius) / (max_radius - min_radius) for r in base_radii]
    scaled_radii = [r * scale_factor for r in scaled_radii]  # Apply scale_factor
    
    # Calculate target phase
    target_phase = abs(target_value1) % (2 * math.pi) if target_value1 != 0 else 0.0
    print(f"Target phase: {target_phase} radians")
    
    # Define material properties for each sphere
    material_properties = [
        {"base_color": (0.8, 0.8, 0.8, 1.0), "metallic": 1.0, "roughness": 0.1, "alpha": 0.499},  # Sphere_0
        {"base_color": (0.8, 0.8, 0.8, 1.0), "metallic": 1.0, "roughness": 0.2, "alpha": 0.141},  # Sphere_1
        {"base_color": (0.8, 0.8, 0.8, 1.0), "metallic": 1.0, "roughness": 0.3, "alpha": 0.202},  # Sphere_2
        {"base_color": (0.8, 0.8, 0.8, 1.0), "metallic": 1.0, "roughness": 0.4, "alpha": 0.3},    # Sphere_3
        {"base_color": (0.8, 0.8, 0.8, 1.0), "metallic": 1.0, "roughness": 0.5, "alpha": 0.4},    # Sphere_4
        {"base_color": (0.8, 0.8, 0.8, 1.0), "metallic": 1.0, "roughness": 0.6, "alpha": 0.499},  # Sphere_5
        {"base_color": (0.8, 0.8, 0.8, 1.0), "metallic": 1.0, "roughness": 0.141, "alpha": 0.202}, # Sphere_6
        {"base_color": (1.0, 0.0, 1.0, 1.0), "metallic": 1.0, "roughness": 0.104, "alpha": 0.9},  # Sphere_7 (magenta)
    ]
    
    # Generate Blender script
    script_content = f"""
import bpy
import math

def create_translucent_emissive_material(name, base_color, metallic, roughness, alpha):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Create Principled BSDF
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    bsdf.inputs['Base Color'].default_value = base_color
    bsdf.inputs['Metallic'].default_value = metallic
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Alpha'].default_value = alpha
    if 'Transmission' in bsdf.inputs:
        bsdf.inputs['Transmission'].default_value = 0.0  # Fully metallic, no transmission
    if 'Emission Color' in bsdf.inputs:
        bsdf.inputs['Emission Color'].default_value = (0.0, 0.0, 0.0, 1.0)  # No emission
    bsdf.inputs['Emission Strength'].default_value = 0.0
    
    # Add Texture Coordinate and Normal Map for reflection
    tex_coord = nodes.new('ShaderNodeTexCoord')
    tex_coord.location = (-400, 0)
    normal_map = nodes.new('ShaderNodeNormalMap')
    normal_map.location = (-200, 0)
    links.new(tex_coord.outputs['Reflection'], normal_map.inputs['Color'])
    links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])
    
    # Output
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    mat.blend_method = 'BLEND'
    return mat

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Animation settings
fps = {int(fps)}
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 320
bpy.context.scene.render.fps = fps

# Sphere creation
radii = {scaled_radii}
differences = {differences}
osc_freq = {osc_freq}
osc_amplitude = {osc_amplitude}
base_angular_speed = {base_angular_speed}
material_props = {material_properties}
for i in range(8):
    print(f"Creating Blender sphere {{i}}")
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radii[i], location=(0, 0, 0))
    sphere = bpy.context.object
    sphere.name = f'Sphere_{{i}}'
    
    # Assign custom material
    props = material_props[i]
    material = create_translucent_emissive_material(
        f'Sphere_Mat_{{i}}',
        props['base_color'],
        props['metallic'],
        props['roughness'],
        props['alpha']
    )
    sphere.data.materials.append(material)
    
    # Animation: XYZ rotation with oscillation
    sphere.rotation_mode = 'XYZ'
    for frame in range(1, 321):
        t = frame / fps
        osc = osc_amplitude * math.sin(2 * math.pi * osc_freq * t)
        angle = base_angular_speed * t * (1.0 + differences[i] * osc)
        sphere.rotation_euler = (
            angle,           # X rotation
            angle * 0.5,     # Y rotation
            angle * 0.25     # Z rotation
        )
        sphere.keyframe_insert(data_path="rotation_euler", frame=frame)

# Origin sphere
if {origin_radius1 is not None}:
    bpy.ops.mesh.primitive_uv_sphere_add(radius={origin_radius1 * scale_factor * 0.5}, location=(0, 0, 0))
    origin_sphere = bpy.context.object
    origin_sphere.name = 'Origin_Sphere'
    origin_material = create_translucent_emissive_material('Origin_Mat', (1.0, 1.0, 1.0, 1.0), 0.0, 0.0, 0.9)
    origin_sphere.data.materials.append(origin_material)

# Lighting
bpy.ops.object.light_add(type='POINT', location=(0, 8, 0))
light = bpy.context.object
light.data.energy = {light_intensity_factor}

# Camera
bpy.ops.object.camera_add(location=(0, -8, 0), rotation=(math.radians(90), 0, 0))
camera = bpy.context.object
bpy.context.scene.camera = camera

print("Blender scene setup complete.")
"""
    # Save script
    script_path = os.path.join(os.getcwd(), "polar_plot_blender.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    print(f"Blender script saved to: {script_path}")
    print("To render in Blender:")
    print("1. Open Blender 4.2 and create a new General project.")
    print("2. Go to the Scripting workspace.")
    print("3. Open 'polar_plot_blender.py' in the Text Editor.")
    print("4. Click 'Run Script' to generate the scene.")
    print("5. Play animation (Spacebar) or render (F12) to see 8 concentric spheres rotating in XYZ with custom reflective materials.")
    print("6. If only one sphere appears, check Console and ensure all spheres are visible in the Outliner.")
    print(f"Oscillation frequency: {osc_freq} Hz")
def render_in_blender(event):
    try:
        target_value1 = float(target_value1_str)
        target_value2 = float(target_value2_str)
    except ValueError:
        target_value1 = math.pi
        target_value2 = 1.0
        print("Invalid target value(s); using defaults: pi, 1.0")
    
    scaled_table1, scaling_factor1 = scale_table(original_table, origin_row1, origin_col1, target_value1)
    scaled_table2, scaling_factor2 = scale_table(original_table, origin_row2, origin_col2, target_value2)
    
    dominant_values1 = get_dominant_values(scaled_table1)
    dominant_values2 = get_dominant_values(scaled_table2)
    
    circles1 = []
    circles2 = []
    theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    differences = []
    
    max_value1 = max(max(row) for row in scaled_table1)
    freq_sum1 = get_frequency_sum(scaled_table1)
    radius_ref1 = max_value1 if max_value1 > freq_sum1 else freq_sum1
    radius_scale = 2 / max(1, radius_ref1)  # Unified scale for consistency
    
    origin_radius1 = None
    origin_radius2 = None
    
    for idx, ((dim1, j1, value1), (dim2, j2, value2)) in enumerate(zip(dominant_values1, dominant_values2)):
        angle = theta[idx]
        color1, is_origin1 = get_color_and_linewidth(dim1-1, j1-1, value1, target_value1, 
                                                    origin_row1, origin_col1, scaled_table1,
                                                    color_assignments['='], 
                                                    color_assignments['>'], 
                                                    color_assignments['<'], False)
        color2, is_origin2 = get_color_and_linewidth(dim2-1, j2-1, value2, target_value2, 
                                                    origin_row2, origin_col2, scaled_table2,
                                                    color_assignments['='], 
                                                    color_assignments['>'], 
                                                    color_assignments['<'], True)
        # Scale lower value to match higher
        if value1 < value2:
            scaled_value1 = value2
            scaled_value2 = value2
        else:
            scaled_value1 = value1
            scaled_value2 = value1
        # Radius = (sum of scaled values) / 2
        radius = (scaled_value1 + scaled_value2) / 2 * radius_scale
        if is_origin1:
            origin_radius1 = radius
        if is_origin2:
            origin_radius2 = radius
        circles1.append({
            'theta': angle,
            'radius': radius,
            'color': color1,
            'dimension': idx,
            'label': f'({dim1}D, {j1}D): {value1:.6f}'
        })
        circles2.append({
            'theta': angle,
            'radius': radius,
            'color': color2,
            'dimension': idx,
            'label': f'({dim2}D, {j2}D): {value2:.6f}'
        })
        differences.append(abs(value1 - value2))
    
    if origin_radius1 is None:
        origin_radius1 = circles1[0]['radius']
    if origin_radius2 is None:
        origin_radius2 = circles2[0]['radius']
    
    max_diff = max(differences) if differences else 1.0
    differences = [d / max_diff for d in differences]
    
    print("circles1:", [{"radius": c['radius'], "color": c['color'], "dimension": c['dimension']} for c in circles1])
    print("circles2:", [{"radius": c['radius'], "color": c['color'], "dimension": c['dimension']} for c in circles2])
    print("origin_radius1:", origin_radius1, "origin_radius2:", origin_radius2)
    print("theta:", theta.tolist())
    print("differences:", differences)
    print("unit:", unit)
    print("target_value1:", target_value1, "target_value2:", target_value2)
    
    generate_blender_script(circles1, circles2, origin_radius1, origin_radius2, theta, differences, unit, target_value1, target_value2)

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
    
    # Add Render in Blender button
    ax_blender = plt.axes([0.7, 0.02, 0.15, 0.05])
    button_blender = Button(ax_blender, 'Render in Blender')
    button_blender.on_clicked(render_in_blender)
    
    update()
    
    plt.show()
    
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")