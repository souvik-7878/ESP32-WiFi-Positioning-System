import socket
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
from datetime import datetime
from scipy.optimize import minimize  # <-- NEW IMPORT for least-squares optimization

# NOTE: This script now requires the 'scipy' library: pip install scipy

# ===============================
# ⚙️ CONFIGURATION
# ===============================
UDP_PORT = 4210  # Must match ESP32's port
LOG_FILE = "esp32_trilateration_log.txt"
SMOOTHING_FACTOR = 0.5  # Alpha for EMA filter (0.0 = no update, 1.0 = no smoothing)
CLOSE_RANGE_THRESHOLD = 0.8 # <-- If dist < this, snap to AP (in meters)
CLOSE_RANGE_JITTER = 0.1  # <-- NEW: Random jitter std dev (in meters) when snapped

# AP coordinates (meters)
AP_COORDS = {
    "narzo": np.array([3.0, 0.0]),
    "laptop": np.array([0.0, 0.0]),
    "hello": np.array([0.0, 3.0])
}
AP_NAMES = ["narzo", "laptop", "hello"]
P1, P2, P3 = AP_COORDS[AP_NAMES[0]], AP_COORDS[AP_NAMES[1]], AP_COORDS[AP_NAMES[2]]

# ===============================
# 🧩 UDP SETUP
# ===============================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.setblocking(False)  # Non-blocking mode
    print(f"✅ Listening for UDP data on port {UDP_PORT}...")
except OSError as e:
    print(f"❌ FAILED to bind to port {UDP_PORT}. Is another instance running?")
    print(f"Error: {e}")
    exit()


# ===============================
# 📊 GLOBAL VARIABLES
# ===============================
esp32_position = np.array([0.0, 0.0]) # This will store the SMOOTHED position
distances = [0.0, 0.0, 0.0]
_history = []
last_data_time = time.time()
snapping_status = "Mode: Initializing" # <-- NEW: For GUI feedback

# ===============================
# 🔢 TRILATERATION FUNCTIONS
# ===============================
def solve_trilateration_algebraic(p1, p2, p3, r1, r2, r3):
    """
    (Original Function)
    Calculates the (x, y) position from three points and their distances
    using a direct algebraic solution. Can be unstable with noisy data.
    Returns np.array([x, y]) or None if no solution.
    """
    # Check for invalid radii
    if r1 <= 0 or r2 <= 0 or r3 <= 0:
        return None
        
    A = 2*(p2[0]-p1[0])
    B = 2*(p2[1]-p1[1])
    C = r1**2 - r2**2 + p2[0]**2 - p1[0]**2 + p2[1]**2 - p1[1]**2
    D = 2*(p3[0]-p2[0])
    E = 2*(p3[1]-p2[1])
    F = r2**2 - r3**2 + p3[0]**2 - p2[0]**2 + p3[1]**2 - p2[1]**2
    
    denom = A*E - B*D
    
    # Check for collinear points (denominator is zero)
    if abs(denom) < 1e-9:
        # print("Warning: Trilateration denominator is near zero. APs may be collinear.")
        return None
        
    x = (C*E - F*B)/denom
    y = (A*F - D*C)/denom
    
    return np.array([x, y])

def solve_trilateration_least_squares(points, radii, initial_guess):
    """
    (New Function)
    Calculates the (x, y) position by minimizing the squared error
    using scipy.optimize.minimize. This is more robust to noisy data.
    """
    
    # Define the error function (cost function)
    def cost_function(pos):
        x, y = pos
        total_error = 0
        for p, r in zip(points, radii):
            # Calculate distance from (x,y) to AP
            dist = np.sqrt((x - p[0])**2 + (y - p[1])**2)
            # Add squared error
            total_error += (dist - r)**2
        return total_error
        
    # Run the optimizer
    result = minimize(
        cost_function,
        initial_guess,
        method='L-BFGS-B',
        bounds=None  # We can add bounds here later if needed, e.g., ((-10, 10), (-10, 10))
    )
    
    if result.success:
        return result.x  # This is the [x, y] position
    else:
        # Optimization failed, return None
        # print(f"Warning: Optimizer failed: {result.message}")
        return None

# ===============================
# 🧩 PARSE INCOMING UDP DATA
# ===============================
def parse_data_line(line):
    """
    Parses a 'DATA:r1,r2,r3' string into a list of floats.
    """
    if not line.startswith("DATA:"):
        return None
    parts = line[len("DATA:"):].strip().split(',')
    if len(parts) != 3:
        # print(f"Warning: Received malformed data: {line}")
        return None
    try:
        return [float(p) for p in parts]
    except ValueError:
        # print(f"Warning: Could not parse numbers in: {line}")
        return None

# ===============================
# 🧾 LOGGING
# ===============================
def log_to_file(distances, position):
    """
    Appends a new data row to the log file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] # Add milliseconds
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp} | Distances: {distances[0]:.2f}, {distances[1]:.2f}, {distances[2]:.2f} | Position: ({position[0]:.2f}, {position[1]:.2f})\n")

# ===============================
# 🔄 UPDATE DATA
# ===============================
def update_data():
    """
    Reads all available data from the UDP socket and updates global state.
    """
    global distances, esp32_position, _history, last_data_time, snapping_status
    data_received = False
    try:
        while True:
            # Keep reading until BlockingIOError (no more data)
            data, _ = sock.recvfrom(1024)
            line = data.decode().strip()
            parsed = parse_data_line(line)
            
            if parsed is not None:
                # Use the most recent valid data
                new_dists = [0.0 if v <= 0.001 else v for v in parsed]
                distances[:] = new_dists
                data_received = True
                
    except BlockingIOError:
        pass  # No new data yet

    if data_received:
        # If we got new data, solve for position
        last_data_time = time.time()
        
        # --- MODIFIED SOLVER AND SMOOTHING LOGIC ---
        
        measured_pos = None # This will hold the "raw" calculated position
        
        # 1. --- NEW "CLOSE RANGE" LOGIC ---
        # Find the minimum valid distance
        valid_dists = [d for d in distances if d > 0.001]
        if valid_dists:
            min_dist = min(valid_dists)
            
            if min_dist < CLOSE_RANGE_THRESHOLD:
                # We are very close to an AP. Trust this one measurement.
                min_index = distances.index(min_dist)
                ap_name = AP_NAMES[min_index]
                ap_pos = AP_COORDS[ap_name].copy()
                last_pos = esp32_position.copy()
                
                # --- MODIFIED LOGIC as requested ---
                # Find direction from AP to last known position
                v = last_pos - ap_pos
                norm_v = np.linalg.norm(v)
                
                if norm_v > 1e-6:
                    # We have a valid direction
                    unit_v = v / norm_v
                    # Set target to be on the small circle, in direction of last pos
                    base_pos = ap_pos + unit_v * min_dist
                else:
                    # We are already at the AP. Just pick an arbitrary direction (e.g., +x)
                    # to place the point *on* the small measured circle.
                    base_pos = ap_pos + np.array([min_dist, 0])
                
                # --- NEW JITTER LOGIC ---
                # Add a small random jitter (e.g., mean 0, std dev 10cm)
                jitter = np.random.randn(2) * CLOSE_RANGE_JITTER
                measured_pos = base_pos + jitter
                # --- END NEW JITTER LOGIC ---
                
                snapping_status = f"Mode: Snapped near {ap_name}"

            else:
                # We are not close to any single AP. Use trilateration.
                snapping_status = "Mode: Trilateration"
        else:
            # No valid distances at all
            snapping_status = "Mode: No Valid Data"

        # 2. --- REGULAR TRILATERATION (if not snapped) ---
        if measured_pos is None and all(d > 0.001 for d in distances):
            # 1. Get an initial guess.
            initial_pos_guess = esp32_position.copy()
            
            # 2. Run the more accurate least-squares optimizer
            points_list = [P1, P2, P3]
            measured_pos = solve_trilateration_least_squares(points_list, distances, initial_pos_guess)
        
        # 3. --- SMOOTHING ---
        if measured_pos is not None:
            # Apply Exponential Moving Average (EMA) smoothing
            # This blends the new measurement with the last known position
            esp32_position[:] = (1 - SMOOTHING_FACTOR) * esp32_position + SMOOTHING_FACTOR * measured_pos
            
            # 4. Log and store history (using the final smoothed position)
            _history.append(esp32_position.copy())
            if len(_history) > 100:
                _history.pop(0)
            
            # Log the raw distances and the final smoothed position
            log_to_file(distances, esp32_position)
        
        # If measured_pos is None (optimizer failed or no valid data), we do nothing
        # and just keep the old esp32_position.

# ===============================
# 📈 MATPLOTLIB GUI SETUP
# ===============================
fig = plt.figure(figsize=(10, 7)) # Made figure slightly larger
gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.2]) # Gave info panel more space
ax = fig.add_subplot(gs[0])
ax_info = fig.add_subplot(gs[1])
ax_info.axis('off')
ax.set_aspect('equal', adjustable='box')

# --- Plot Access Points ---
ap_scatter = ax.scatter([P1[0], P2[0], P3[0]], [P1[1], P2[1], P3[1]],
                        s=100, c='blue', label='Access Points (APs)', zorder=5)
for i, name in enumerate(AP_NAMES):
    ax.text(AP_COORDS[name][0] + 0.3, AP_COORDS[name][1] + 0.3, f"{name}\n({AP_COORDS[name][0]}, {AP_COORDS[name][1]})", color='blue', fontsize=9)

# --- Plot ESP32 ---
esp32_dot, = ax.plot([], [], 'ro', markersize=10, label='ESP32 (Smoothed)', zorder=10) # <-- Label updated

# --- Plot Circles ---
# Circles for AP range
circle1 = plt.Circle(P1, 0, fill=False, linestyle='--', alpha=0.8, color='#1f77b4', label=f'"{AP_NAMES[0]}" raw dist') # <-- Label updated
circle2 = plt.Circle(P2, 0, fill=False, linestyle='--', alpha=0.8, color='#ff7f0e', label=f'"{AP_NAMES[1]}" raw dist') # <-- Label updated
circle3 = plt.Circle(P3, 0, fill=False, linestyle='--', alpha=0.8, color='#2ca02c', label=f'"{AP_NAMES[2]}" raw dist') # <-- Label updated
ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)

# --- Plot Uncertainty Visualizations ---
# This was the orange circle, let's make it more prominent
uncertainty_radius_circle = plt.Circle((0, 0), 0, fill=False, color='red', linestyle=':', lw=2, alpha=0.7, label='Mean Error Radius')
ax.add_artist(uncertainty_radius_circle)

# --- Plot History Trail ---
trail, = ax.plot([], [], 'r-', alpha=0.3, lw=2, label='History (100 pts)')

# --- Plot Text Info ---
info_text = ax_info.text(0.0, 0.95, '', va='top', ha='left',
                         fontsize=11, fontfamily='monospace',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', alpha=1.0))

live_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

status_text = ax.text(0.98, 0.98, 'Waiting for data...', transform=ax.transAxes,
                      fontsize=10, verticalalignment='top', horizontalalignment='right',
                      color='red', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


# ===============================
# 🧭 INIT FUNCTION
# ===============================
def init():
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("ESP32 Wi-Fi Trilateration Tracker (UDP)")
    ax.grid(True, linestyle=':', alpha=0.7)
    
    all_pts = np.array([P1, P2, P3])
    minc = np.min(all_pts, axis=0) - 5
    maxc = np.max(all_pts, axis=0) + 5
    ax.set_xlim(minc[0], maxc[0])
    ax.set_ylim(minc[1], maxc[1])
    
    esp32_dot.set_data([], [])
    trail.set_data([], [])
    
    # Re-order legend
    handles, labels = ax.get_legend_handles_labels()
    # Updated order to match new label
    order = [0, 1, 5, 2, 3, 4, 6] 
    if len(handles) == len(order):
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower left', fontsize=8)
    else:
        # Fallback if handle count is off
        ax.legend(loc='lower left', fontsize=8)

    return esp32_dot, uncertainty_radius_circle, info_text, live_text, trail, status_text

# ===============================
# 🔁 ANIMATION LOOP
# ===============================
def animate(i):
    update_data()
    
    # esp32_position is now the smoothed position
    esp32_dot.set_data([esp32_position[0]], [esp32_position[1]])

    # --- Update status text ---
    time_since_data = time.time() - last_data_time
    if time_since_data > 2.0:
        status_text.set_text(f"NO DATA ({time_since_data:.0f}s)")
        status_text.set_color('red')
    else:
        status_text.set_text("● LIVE")
        status_text.set_color('green')


    # --- Update circle visibility (using raw 'distances') ---
    circles = [circle1, circle2, circle3]
    valid_dists = [d > 0.001 for d in distances]
    
    for idx, (circle, dist, valid) in enumerate(zip(circles, distances, valid_dists)):
        if valid:
            circle.set_visible(True)
            circle.set_radius(dist) # Dashed circles always show raw data
        else:
            circle.set_visible(False)

    # --- Trail update ---
    if len(_history) > 1:
        hist_array = np.array(_history)
        trail.set_data(hist_array[:, 0], hist_array[:, 1])

    # --- NEW: Uncertainty and Display Logic ---
    # We now calculate this every frame, regardless of mode,
    # to show the most accurate info in the text panel.
    
    est = esp32_position # Use the final smoothed position
    
    # 1. Calculate the *actual* distances from the plotted dot to the APs
    dist_to_p1 = np.linalg.norm(est - P1)
    dist_to_p2 = np.linalg.norm(est - P2)
    dist_to_p3 = np.linalg.norm(est - P3)
    
    # This list holds the *plotted* distances, which we will display
    plotted_distances = [dist_to_p1, dist_to_p2, dist_to_p3]
    
    # 2. Calculate errors (difference between *plotted* and *raw* signal)
    # Use raw `distances` list (global) for this comparison.
    error1 = abs(dist_to_p1 - distances[0]) if valid_dists[0] else 0.0
    error2 = abs(dist_to_p2 - distances[1]) if valid_dists[1] else 0.0
    error3 = abs(dist_to_p3 - distances[2]) if valid_dists[2] else 0.0
    dist_errors = [error1, error2, error3]

    # 3. Calculate Mean Error
    valid_errors = [err for err, valid in zip(dist_errors, valid_dists) if valid]
    if valid_errors:
        mean_error = np.mean(valid_errors)
        uncertainty_radius_circle.center = (est[0], est[1])
        uncertainty_radius_circle.set_radius(mean_error)
        uncertainty_radius_circle.set_visible(True)
        error_text = f"Mean Error: {mean_error:6.2f} m"
    else:
        uncertainty_radius_circle.set_visible(False)
        error_text = "Mean Error: N/A"
    # --- END NEW LOGIC ---


    live_text.set_text(f"ESP32: ({esp32_position[0]:.2f}, {esp32_position[1]:.2f})")

    info_lines = [
        "📡 LIVE DATA",
        "---------------------",
        f"ESP32 (Smoothed):",
        f"  X = {esp32_position[0]:6.2f} m",
        f"  Y = {esp32_position[1]:6.2f} m",
        "",
        "AP Distances (Error):",
        # --- THIS IS THE KEY CHANGE ---
        # `dist` now comes from `plotted_distances`
        # `err` is the difference between plotted and raw signal
        *(f"  {name:<6}: {dist:5.2f}m ({err:4.2f}m)" for name, dist, err in zip(AP_NAMES, plotted_distances, dist_errors)),
        # --- END KEY CHANGE ---
        "",
        error_text,
        f"\nSmoothing: {SMOOTHING_FACTOR*100:.0f}%",
        snapping_status
    ]
    info_text.set_text("\n".join(info_lines))
    
    return esp32_dot, uncertainty_radius_circle, info_text, live_text, trail, status_text

# ===============================
# 🚀 MAIN
# ===============================
if __name__ == "__main__":
    # Reset log file
    with open(LOG_FILE, "w") as f:
        f.write("=== ESP32 Wi-Fi Trilateration Log (UDP Version) ===\n")
        f.write(f"Timestamp | Distances [{AP_NAMES[0]}, {AP_NAMES[1]}, {AP_NAMES[2]}] | Position (X,Y)\n\n")

    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=50, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

    # Clean up socket
    sock.close()
    print("Socket closed. Exiting.")

