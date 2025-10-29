import socket
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
from datetime import datetime

# ===============================
# ⚙️ CONFIGURATION
# ===============================
UDP_PORT = 4210  # Must match ESP32's port
LOG_FILE = "esp32_trilateration_log.txt"

# AP coordinates (meters)
AP_COORDS = {
    "narzo": np.array([3.0, 0.0]),
    "laptop": np.array([3.0, 3.0]),
    "A": np.array([0.0, 0.0])
}
AP_NAMES = ["narzo", "A", "laptop"]
P1, P2, P3 = AP_COORDS[AP_NAMES[0]], AP_COORDS[AP_NAMES[1]], AP_COORDS[AP_NAMES[2]]

# ===============================
# 🧩 UDP SETUP
# ===============================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", UDP_PORT))
sock.setblocking(False)  # Non-blocking mode
print(f"✅ Listening for UDP data on port {UDP_PORT}...")

# ===============================
# 📊 GLOBAL VARIABLES
# ===============================
esp32_position = np.array([0.0, 0.0])
distances = [0.0, 0.0, 0.0]
_history = []

# ===============================
# 🔢 TRILATERATION FUNCTION
# ===============================
def solve_trilateration(p1, p2, p3, r1, r2, r3):
    if r1 <= 0 or r2 <= 0 or r3 <= 0:
        return None
    A = 2*(p2[0]-p1[0])
    B = 2*(p2[1]-p1[1])
    C = r1**2 - r2**2 + p2[0]**2 - p1[0]**2 + p2[1]**2 - p1[1]**2
    D = 2*(p3[0]-p2[0])
    E = 2*(p3[1]-p2[1])
    F = r2**2 - r3**2 + p3[0]**2 - p2[0]**2 + p3[1]**2 - p2[1]**2
    denom = A*E - B*D
    if abs(denom) < 1e-9:
        return None
    x = (C*E - F*B)/denom
    y = (A*F - D*C)/denom
    return np.array([x, y])

# ===============================
# 🧩 PARSE INCOMING UDP DATA
# ===============================
def parse_data_line(line):
    if not line.startswith("DATA:"):
        return None
    parts = line[len("DATA:"):].strip().split(',')
    if len(parts) != 3:
        return None
    try:
        return [float(p) for p in parts]
    except:
        return None

# ===============================
# 🧾 LOGGING
# ===============================
def log_to_file(distances, position):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp} | Distances: {distances} | Position: ({position[0]:.2f}, {position[1]:.2f})\n")

# ===============================
# 🔄 UPDATE DATA
# ===============================
def update_data():
    global distances, esp32_position, _history
    try:
        while True:
            data, _ = sock.recvfrom(1024)
            line = data.decode().strip()
            parsed = parse_data_line(line)
            if parsed is not None:
                new_dists = [0.0 if v <= 0.001 else v for v in parsed]
                distances[:] = new_dists
                pos = solve_trilateration(P1, P2, P3, distances[0], distances[1], distances[2])
                if pos is not None:
                    esp32_position[:] = pos
                    _history.append(pos.copy())
                    if len(_history) > 100:
                        _history.pop(0)
                    log_to_file(distances, esp32_position)
    except BlockingIOError:
        pass  # No new data yet

# ===============================
# 📈 MATPLOTLIB GUI SETUP
# ===============================
fig = plt.figure(figsize=(9, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
ax = fig.add_subplot(gs[0])
ax_info = fig.add_subplot(gs[1])
ax_info.axis('off')
ax.set_aspect('equal', adjustable='box')

# --- Plot Access Points ---
ap_scatter = ax.scatter([P1[0], P2[0], P3[0]], [P1[1], P2[1], P3[1]],
                        s=100, c='blue', label='Access Points')
for i, name in enumerate(AP_NAMES):
    ax.text(AP_COORDS[name][0] + 0.3, AP_COORDS[name][1] + 0.3, name, color='blue')

esp32_dot, = ax.plot([], [], 'ro', markersize=8, label='ESP32')

# Circles for AP range
circle1 = plt.Circle(P1, 0, fill=False, linestyle='--', alpha=0.6)
circle2 = plt.Circle(P2, 0, fill=False, linestyle='--', alpha=0.6)
circle3 = plt.Circle(P3, 0, fill=False, linestyle='--', alpha=0.6)
ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)

trail, = ax.plot([], [], 'r-', alpha=0.5, lw=1)
intersection_circle = plt.Circle((0, 0), 0, fill=False, linestyle='--', color='green', lw=1.5, alpha=0.7)
ax.add_artist(intersection_circle)

uncertainty_ellipse = plt.Circle((0, 0), 0, fill=False, color='orange', linestyle=':', alpha=0.5)
ax.add_artist(uncertainty_ellipse)

info_text = ax_info.text(0.05, 0.95, '', va='top', ha='left',
                         fontsize=10, fontfamily='monospace',
                         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

live_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ===============================
# 🧭 INIT FUNCTION
# ===============================
def init():
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("ESP32 Wi-Fi Trilateration Tracker (UDP)")
    ax.grid(True)
    ax.legend()
    all_pts = np.array([P1, P2, P3])
    minc = np.min(all_pts, axis=0) - 5
    maxc = np.max(all_pts, axis=0) + 5
    ax.set_xlim(minc[0], maxc[0])
    ax.set_ylim(minc[1], maxc[1])
    esp32_dot.set_data([], [])
    trail.set_data([], [])
    return esp32_dot, trail, uncertainty_ellipse, info_text, live_text

# ===============================
# 🔁 ANIMATION LOOP
# ===============================
def animate(i):
    update_data()
    esp32_dot.set_data([esp32_position[0]], [esp32_position[1]])

    # --- Update circle visibility ---
    circles = [circle1, circle2, circle3]
    for idx, (circle, dist) in enumerate(zip(circles, distances)):
        if dist > 0.001:  # Signal ON
            circle.set_visible(True)
            circle.set_radius(dist)
        else:  # Signal OFF
            circle.set_visible(False)

    # --- Draw green intersection circle only if 3 valid distances ---
    if all(d > 0.001 for d in distances):
        intersection_circle.set_visible(True)
        intersection_circle.center = esp32_position
        intersection_circle.set_radius(np.mean(distances))
    else:
        intersection_circle.set_visible(False)

    # --- Trail update ---
    if len(_history) > 1:
        hist_array = np.array(_history)
        trail.set_data(hist_array[:, 0], hist_array[:, 1])

    # --- Uncertainty visualization ---
    if len(distances) == 3 and all(d > 0.001 for d in distances):
        est = esp32_position
        rad = np.mean([abs(np.linalg.norm(est - P1) - distances[0]),
                       abs(np.linalg.norm(est - P2) - distances[1]),
                       abs(np.linalg.norm(est - P3) - distances[2])])
        uncertainty_ellipse.center = (est[0], est[1])
        uncertainty_ellipse.set_radius(rad)

    live_text.set_text(f"ESP32: ({esp32_position[0]:.2f}, {esp32_position[1]:.2f})")

    info_lines = [
        "📡 LIVE DATA",
        "----------------------",
        f"ESP32 Position:",
        f"   X = {esp32_position[0]:6.2f} m",
        f"   Y = {esp32_position[1]:6.2f} m",
        "",
        "Distances from APs:",
        *(f"   {name:<6}: {dist:.2f} m" for name, dist in zip(AP_NAMES, distances))
    ]
    info_text.set_text("\n".join(info_lines))
    return esp32_dot, trail, uncertainty_ellipse, info_text, live_text

# ===============================
# 🚀 MAIN
# ===============================
if __name__ == "__main__":
    # Reset log file
    with open(LOG_FILE, "w") as f:
        f.write("=== ESP32 Wi-Fi Trilateration Log (UDP Version) ===\n")
        f.write("Timestamp | Distances [narzo, laptop, A] | Position (X,Y)\n\n")

    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=200, blit=False)
    plt.tight_layout()
    plt.show()
