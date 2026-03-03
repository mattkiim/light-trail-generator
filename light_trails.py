#!/usr/bin/env python3
"""
Light Trail Generator — Desktop App
=====================================
1. Load a video of turtlebots
2. Click on each robot in the first frame to initialize tracking
3. Adjust detection/style parameters
4. Run tracking + render glowing light trails on the last frame

Requirements:
    pip install opencv-python numpy pillow
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import math
import os


# ── Detection ────────────────────────────────────────────────

def detect_robots(frame, threshold=80, min_area=800, max_area=15000):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            detections.append((cx, cy))
    return detections


# ── Tracker ──────────────────────────────────────────────────

class SimpleTracker:
    def __init__(self, initial_positions, max_dist=80):
        self.n = len(initial_positions)
        self.max_dist = max_dist
        self.tracks = [[pos] for pos in initial_positions]
        self.velocities = [(0.0, 0.0)] * self.n

    def _predict(self, i):
        lx, ly = self.tracks[i][-1]
        vx, vy = self.velocities[i]
        return (lx + vx, ly + vy)

    def update(self, detections):
        if len(detections) == 0:
            for i in range(self.n):
                self.tracks[i].append(self.tracks[i][-1])
            return

        predictions = [self._predict(i) for i in range(self.n)]

        used = set()
        assignments = [None] * self.n
        for _ in range(self.n):
            best_cost, best_i, best_j = float("inf"), -1, -1
            for i in range(self.n):
                if assignments[i] is not None:
                    continue
                for j, d in enumerate(detections):
                    if j in used:
                        continue
                    dist = math.hypot(predictions[i][0] - d[0], predictions[i][1] - d[1])
                    if dist < best_cost:
                        best_cost, best_i, best_j = dist, i, j
            if best_i >= 0 and best_cost < self.max_dist:
                assignments[best_i] = best_j
                used.add(best_j)

        for i in range(self.n):
            if assignments[i] is not None:
                d = detections[assignments[i]]
                old = self.tracks[i][-1]
                self.velocities[i] = (
                    0.7 * self.velocities[i][0] + 0.3 * (d[0] - old[0]),
                    0.7 * self.velocities[i][1] + 0.3 * (d[1] - old[1]),
                )
                self.tracks[i].append(d)
            else:
                self.tracks[i].append(self._predict(i))


# ── Smoothing ────────────────────────────────────────────────

def smooth_trajectory(points, window=7):
    if len(points) < window:
        return points
    pts = np.array(points)
    k = np.ones(window) / window
    sx = np.convolve(pts[:, 0], k, mode="same")
    sy = np.convolve(pts[:, 1], k, mode="same")
    hw = window // 2
    sx[:hw], sx[-hw:] = pts[:hw, 0], pts[-hw:, 0]
    sy[:hw], sy[-hw:] = pts[:hw, 1], pts[-hw:, 1]
    return list(zip(sx.tolist(), sy.tolist()))


# ── Glow Drawing ─────────────────────────────────────────────

def draw_glow_trail(base, points, color, width=4, glow=25, fade_mode="none"):
    if len(points) < 2:
        return base
    r, g, b = color
    result = base.copy()
    n = len(points)

    def alpha_at(i):
        """Return alpha multiplier 0.0-1.0 for point index i."""
        if fade_mode == "none":
            return 1.0
        t = i / max(n - 1, 1)
        if fade_mode == "fade_in":
            # bright at start, fades out toward end
            return 1.0 - 0.85 * t
        elif fade_mode == "fade_out":
            # dim at start, brightens toward end
            return 0.15 + 0.85 * t
        return 1.0

    # for "none" mode, draw all at once (fast path)
    if fade_mode == "none":
        layers = [
            (glow * 3, 25,  glow * 2),
            (glow,     60,  glow),
            (8,        140, 6),
        ]
        for extra_w, alpha, blur_r in layers:
            layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer)
            for i in range(1, n):
                draw.line([points[i-1], points[i]], fill=(r, g, b, alpha), width=width + extra_w)
            layer = layer.filter(ImageFilter.GaussianBlur(radius=blur_r))
            result = Image.alpha_composite(result, layer)

        # bright core
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        cr, cg, cb = min(255, r+120), min(255, g+120), min(255, b+120)
        for i in range(1, n):
            draw.line([points[i-1], points[i]], fill=(cr, cg, cb, 240), width=width)
        layer = layer.filter(ImageFilter.GaussianBlur(radius=1))
        result = Image.alpha_composite(result, layer)

        # white center
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for i in range(1, n):
            draw.line([points[i-1], points[i]], fill=(255, 255, 255, 180), width=max(1, width - 1))
        layer = layer.filter(ImageFilter.GaussianBlur(radius=1))
        result = Image.alpha_composite(result, layer)

        return result

    # fading mode: draw in chunks for performance (groups of ~20 segments)
    chunk = max(1, n // 50)
    for start in range(0, n - 1, chunk):
        end = min(start + chunk + 1, n)
        seg = points[start:end]
        if len(seg) < 2:
            continue
        mid_idx = (start + end) // 2
        a = alpha_at(mid_idx)

        layers = [
            (glow * 3, int(25 * a),  glow * 2),
            (glow,     int(60 * a),  glow),
            (8,        int(140 * a), 6),
        ]
        for extra_w, alpha_val, blur_r in layers:
            if alpha_val < 1:
                continue
            layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer)
            for i in range(1, len(seg)):
                draw.line([seg[i-1], seg[i]], fill=(r, g, b, alpha_val), width=width + extra_w)
            layer = layer.filter(ImageFilter.GaussianBlur(radius=blur_r))
            result = Image.alpha_composite(result, layer)

        # core
        cr, cg, cb = min(255, r+120), min(255, g+120), min(255, b+120)
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for i in range(1, len(seg)):
            draw.line([seg[i-1], seg[i]], fill=(cr, cg, cb, int(240 * a)), width=width)
        layer = layer.filter(ImageFilter.GaussianBlur(radius=1))
        result = Image.alpha_composite(result, layer)

        # white
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for i in range(1, len(seg)):
            draw.line([seg[i-1], seg[i]], fill=(255, 255, 255, int(180 * a)), width=max(1, width - 1))
        layer = layer.filter(ImageFilter.GaussianBlur(radius=1))
        result = Image.alpha_composite(result, layer)

    return result


def draw_head_glow(base, pos, color, radius=28):
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    r, g, b = color
    x, y = pos
    for ring in range(radius, 0, -1):
        a = int(12 * (radius - ring) / radius)
        draw.ellipse([x-ring, y-ring, x+ring, y+ring], fill=(r, g, b, a))
    draw.ellipse([x-5, y-5, x+5, y+5], fill=(min(255,r+120), min(255,g+120), min(255,b+120), 240))
    draw.ellipse([x-2, y-2, x+2, y+2], fill=(255, 255, 255, 255))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=3))
    return Image.alpha_composite(base, layer)


def draw_start_marker(base, pos, color, radius=20):
    """Draw a diamond-shaped start marker with glow."""
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    r, g, b = color
    x, y = pos

    # outer glow ring
    for ring in range(radius, 0, -1):
        a = int(10 * (radius - ring) / radius)
        draw.ellipse([x-ring, y-ring, x+ring, y+ring], fill=(r, g, b, a))

    # diamond shape
    s = 7
    diamond = [(x, y-s), (x+s, y), (x, y+s), (x-s, y)]
    draw.polygon(diamond, fill=(min(255,r+100), min(255,g+100), min(255,b+100), 220),
                 outline=(255, 255, 255, 240))

    layer = layer.filter(ImageFilter.GaussianBlur(radius=2))
    return Image.alpha_composite(base, layer)


# ── App ──────────────────────────────────────────────────────

TRAIL_COLORS = [
    (255, 51, 102),   # pink
    (51, 204, 255),   # cyan
    (102, 255, 136),  # green
    (255, 136, 51),   # orange
    (170, 102, 255),  # purple
]

COLOR_NAMES = ["Pink", "Cyan", "Green", "Orange", "Purple"]


class LightTrailApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Light Trail Generator")
        self.root.configure(bg="#0a0b10")

        # state
        self.video_path = None
        self.first_frame = None       # original BGR
        self.last_frame = None        # original BGR
        self.display_frame = None     # resized for display
        self.scale = 1.0              # display scale factor
        self.init_positions = []      # clicked robot positions (in original frame coords)
        self.n_robots = tk.IntVar(value=3)
        self.result_image = None

        # params
        self.threshold = tk.IntVar(value=80)
        self.min_area = tk.IntVar(value=800)
        self.max_area = tk.IntVar(value=15000)
        self.max_dist = tk.IntVar(value=80)
        self.skip_frames = tk.IntVar(value=2)
        self.smooth_window = tk.IntVar(value=7)
        self.trail_width = tk.IntVar(value=4)
        self.glow_size = tk.IntVar(value=25)
        self.trail_end_pct = tk.IntVar(value=100)
        self.show_start_marker = tk.BooleanVar(value=True)
        self.fade_mode = tk.StringVar(value="none")  # "none", "fade_in", "fade_out"
        self.start_frame_pct = tk.IntVar(value=0)  # trail start = robot selection frame

        self.build_ui()

    def build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TFrame", background="#0a0b10")
        style.configure("Dark.TLabel", background="#0a0b10", foreground="#888", font=("Courier", 9))
        style.configure("Header.TLabel", background="#0a0b10", foreground="#555", font=("Courier", 9, "bold"))
        style.configure("Status.TLabel", background="#0a0b10", foreground="#33ccff", font=("Courier", 10))
        style.configure("Dark.TButton", font=("Courier", 10))
        style.configure("Accent.TButton", font=("Courier", 10, "bold"))

        # main layout
        main = ttk.Frame(self.root, style="Dark.TFrame")
        main.pack(fill=tk.BOTH, expand=True)

        # sidebar
        sidebar = ttk.Frame(main, style="Dark.TFrame", width=260)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(12, 0), pady=12)
        sidebar.pack_propagate(False)

        ttk.Label(sidebar, text="LIGHT TRAIL GENERATOR", style="Header.TLabel").pack(anchor="w", pady=(0, 16))

        # step 1: load video
        ttk.Label(sidebar, text="① LOAD VIDEO", style="Header.TLabel").pack(anchor="w", pady=(8, 4))
        ttk.Button(sidebar, text="Open Video", command=self.load_video).pack(fill=tk.X, pady=(0, 4))
        self.video_label = ttk.Label(sidebar, text="No video loaded", style="Dark.TLabel")
        self.video_label.pack(anchor="w")

        # step 2: mark robots
        ttk.Label(sidebar, text="② MARK ROBOTS", style="Header.TLabel").pack(anchor="w", pady=(12, 4))
        f = ttk.Frame(sidebar, style="Dark.TFrame")
        f.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(f, text="Count:", style="Dark.TLabel").pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=1, to=5, textvariable=self.n_robots, width=4, font=("Courier", 10)).pack(side=tk.LEFT, padx=4)

        sf = ttk.Frame(sidebar, style="Dark.TFrame")
        sf.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(sf, text="Start Frame %", style="Dark.TLabel", width=16).pack(side=tk.LEFT)
        ttk.Scale(sf, from_=0, to=100, variable=self.start_frame_pct, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(sf, textvariable=self.start_frame_pct, style="Dark.TLabel", width=4).pack(side=tk.RIGHT)

        ttk.Button(sidebar, text="Go to Frame", command=self.go_to_start_frame).pack(fill=tk.X, pady=(0, 4))
        ttk.Button(sidebar, text="Reset Clicks", command=self.reset_clicks).pack(fill=tk.X, pady=(0, 4))
        self.click_label = ttk.Label(sidebar, text="Click robots on the image", style="Dark.TLabel")
        self.click_label.pack(anchor="w")

        # color legend
        self.legend_frame = ttk.Frame(sidebar, style="Dark.TFrame")
        self.legend_frame.pack(anchor="w", pady=(4, 0))

        # step 3: parameters
        ttk.Label(sidebar, text="③ PARAMETERS", style="Header.TLabel").pack(anchor="w", pady=(12, 4))

        params = [
            ("Threshold", self.threshold, 20, 200),
            ("Min Area", self.min_area, 100, 30000),
            ("Max Area", self.max_area, 1000, 50000),
            ("Max Track Dist", self.max_dist, 20, 300),
            ("Frame Skip", self.skip_frames, 1, 10),
            ("Smooth Window", self.smooth_window, 1, 21),
            ("Trail Width", self.trail_width, 1, 20),
            ("Glow Size", self.glow_size, 5, 60),
            ("End Frame %", self.trail_end_pct, 0, 100),
        ]
        for label, var, lo, hi in params:
            pf = ttk.Frame(sidebar, style="Dark.TFrame")
            pf.pack(fill=tk.X, pady=1)
            ttk.Label(pf, text=label, style="Dark.TLabel", width=16).pack(side=tk.LEFT)
            ttk.Scale(pf, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Label(pf, textvariable=var, style="Dark.TLabel", width=6).pack(side=tk.RIGHT)

        # step 4: trail emphasis
        ttk.Label(sidebar, text="④ TRAIL EMPHASIS", style="Header.TLabel").pack(anchor="w", pady=(12, 4))

        ttk.Checkbutton(sidebar, text="Start marker (bright dot)",
                        variable=self.show_start_marker).pack(anchor="w", pady=2)

        fade_frame = ttk.Frame(sidebar, style="Dark.TFrame")
        fade_frame.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(fade_frame, text="Fade:", style="Dark.TLabel").pack(side=tk.LEFT)
        for val, label in [("none", "None"), ("fade_in", "Fade in"), ("fade_out", "Fade out")]:
            ttk.Radiobutton(fade_frame, text=label, variable=self.fade_mode, value=val).pack(side=tk.LEFT, padx=4)

        # step 5: generate
        ttk.Label(sidebar, text="⑤ GENERATE", style="Header.TLabel").pack(anchor="w", pady=(12, 4))
        ttk.Button(sidebar, text="▶  Run Tracking", command=self.run_tracking, style="Accent.TButton").pack(fill=tk.X, pady=(0, 4))
        ttk.Button(sidebar, text="Save Result", command=self.save_result).pack(fill=tk.X, pady=(0, 4))

        self.status = ttk.Label(sidebar, text="Ready", style="Status.TLabel")
        self.status.pack(anchor="w", pady=(8, 0))

        self.progress = ttk.Progressbar(sidebar, mode="determinate")
        self.progress.pack(fill=tk.X, pady=(4, 0))

        # canvas area
        canvas_frame = ttk.Frame(main, style="Dark.TFrame")
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=12, pady=12)

        self.canvas = tk.Canvas(canvas_frame, bg="#111218", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def load_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv"), ("All", "*.*")]
        )
        if not path:
            return

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open {path}")
            return

        self.video_path = path
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        ret, self.first_frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Cannot read first frame")
            cap.release()
            return

        # read last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, self.total_frames - 1))
        ret, self.last_frame = cap.read()
        if not ret:
            self.last_frame = self.first_frame.copy()
        cap.release()

        self.video_label.config(text=f"{os.path.basename(path)}\n{self.total_frames} frames, {self.fps:.0f}fps")
        self.init_positions = []
        self.result_image = None
        self.start_frame_pct.set(0)
        self.update_click_label()
        self.show_frame(self.first_frame)
        self.status.config(text="Adjust Start Frame %, then click each robot")

    def go_to_start_frame(self):
        """Seek to the start frame and display it for robot selection."""
        if self.video_path is None:
            return
        cap = cv2.VideoCapture(self.video_path)
        target_idx = int(self.start_frame_pct.get() / 100.0 * self.total_frames)
        target_idx = max(0, min(target_idx, self.total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.first_frame = frame
            self.init_positions = []
            self.update_click_label()
            self.update_legend()
            self.show_frame(frame)
            self.status.config(text=f"Showing frame {target_idx}/{self.total_frames}. Click robots.")

    def show_frame(self, frame_bgr):
        """Display a BGR frame on the canvas, scaled to fit."""
        h, w = frame_bgr.shape[:2]
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 800, 600

        self.scale = min(cw / w, ch / h)
        nw, nh = int(w * self.scale), int(h * self.scale)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((nw, nh), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil)

        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor=tk.CENTER)
        self._img_offset = ((cw - nw) // 2, (ch - nh) // 2)

        # redraw markers
        self.draw_markers()

    def draw_markers(self):
        self.canvas.delete("marker")
        for i, (ox, oy) in enumerate(self.init_positions):
            # convert original coords to canvas coords
            dx = self._img_offset[0] + ox * self.scale
            dy = self._img_offset[1] + oy * self.scale
            c = TRAIL_COLORS[i % len(TRAIL_COLORS)]
            hex_c = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
            r = 10
            self.canvas.create_oval(dx - r, dy - r, dx + r, dy + r,
                                    outline=hex_c, width=2, tags="marker")
            self.canvas.create_line(dx - r, dy, dx + r, dy, fill=hex_c, width=1, tags="marker")
            self.canvas.create_line(dx, dy - r, dx, dy + r, fill=hex_c, width=1, tags="marker")
            self.canvas.create_text(dx + r + 6, dy - r, text=str(i + 1),
                                    fill=hex_c, font=("Courier", 11, "bold"), anchor=tk.SW, tags="marker")

    def on_canvas_click(self, event):
        if self.first_frame is None:
            return
        if len(self.init_positions) >= self.n_robots.get():
            return

        # convert canvas coords to original frame coords
        ox = (event.x - self._img_offset[0]) / self.scale
        oy = (event.y - self._img_offset[1]) / self.scale

        h, w = self.first_frame.shape[:2]
        if 0 <= ox <= w and 0 <= oy <= h:
            self.init_positions.append((ox, oy))
            self.draw_markers()
            self.update_click_label()
            self.update_legend()

            n = len(self.init_positions)
            total = self.n_robots.get()
            if n >= total:
                self.status.config(text=f"All {total} robots marked. Ready to run!")
            else:
                self.status.config(text=f"Click robot {n + 1} of {total}")

    def on_canvas_resize(self, event):
        if self.result_image is not None:
            self.show_pil_image(self.result_image)
        elif self.first_frame is not None:
            self.show_frame(self.first_frame)

    def reset_clicks(self):
        self.init_positions = []
        self.update_click_label()
        self.update_legend()
        if self.first_frame is not None:
            self.show_frame(self.first_frame)
        self.status.config(text="Click on each robot to mark it")

    def update_click_label(self):
        n = len(self.init_positions)
        total = self.n_robots.get()
        self.click_label.config(text=f"{n}/{total} robots marked")

    def update_legend(self):
        for w in self.legend_frame.winfo_children():
            w.destroy()
        for i, (ox, oy) in enumerate(self.init_positions):
            c = TRAIL_COLORS[i % len(TRAIL_COLORS)]
            hex_c = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
            name = COLOR_NAMES[i % len(COLOR_NAMES)]
            ttk.Label(self.legend_frame,
                      text=f"  ● Robot {i+1} ({name}): ({int(ox)}, {int(oy)})",
                      foreground=hex_c, background="#0a0b10",
                      font=("Courier", 9)).pack(anchor="w")

    def show_pil_image(self, pil_img):
        """Display a PIL RGB image on the canvas."""
        w, h = pil_img.size
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        self.scale = min(cw / w, ch / h)
        nw, nh = int(w * self.scale), int(h * self.scale)
        resized = pil_img.resize((nw, nh), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor=tk.CENTER)
        self._img_offset = ((cw - nw) // 2, (ch - nh) // 2)

    def run_tracking(self):
        if self.video_path is None:
            messagebox.showwarning("No video", "Load a video first.")
            return
        if len(self.init_positions) < self.n_robots.get():
            messagebox.showwarning("Mark robots", f"Click on all {self.n_robots.get()} robots first.")
            return

        self.status.config(text="Tracking...")
        self.progress["value"] = 0
        threading.Thread(target=self._tracking_thread, daemon=True).start()

    def _tracking_thread(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            total = self.total_frames
            skip = self.skip_frames.get()
            thresh = self.threshold.get()
            mina = self.min_area.get()
            maxa = self.max_area.get()
            maxd = self.max_dist.get()

            # seek to start frame (where robots were marked)
            start_idx = int(self.start_frame_pct.get() / 100.0 * total)
            start_idx = max(0, min(start_idx, total - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

            tracker = SimpleTracker(self.init_positions, max_dist=maxd)
            bg_frame = None
            idx = start_idx

            end_pct = self.trail_end_pct.get() / 100.0
            end_frame_idx = int(end_pct * total)
            end_frame_idx = max(start_idx + 1, min(end_frame_idx, total))

            frames_to_process = end_frame_idx - start_idx

            print(f"[DEBUG] Tracking frames {start_idx} → {end_frame_idx} ({frames_to_process} frames)")
            print(f"[DEBUG] Init positions: {self.init_positions}")
            print(f"[DEBUG] Params: thresh={thresh} min_area={mina} max_area={maxa} max_dist={maxd} skip={skip}")

            # skip the init frame (already used for marking)
            cap.read()
            idx = start_idx + 1

            det_counts = []
            while idx <= end_frame_idx:
                ret, frame = cap.read()
                if not ret:
                    break
                bg_frame = frame
                if idx % skip == 0:
                    dets = detect_robots(frame, threshold=thresh, min_area=mina, max_area=maxa)
                    tracker.update(dets)
                    det_counts.append(len(dets))
                idx += 1

                if idx % 50 == 0:
                    pct = int(100 * (idx - start_idx) / max(frames_to_process, 1))
                    self.root.after(0, lambda p=pct, i=idx, e=end_frame_idx: self._update_progress(p, f"Frame {i}/{e}"))

            cap.release()

            if bg_frame is None:
                bg_frame = self.last_frame

            print(f"[DEBUG] Processed {idx - start_idx} frames")
            if det_counts:
                print(f"[DEBUG] Detections per frame: min={min(det_counts)} max={max(det_counts)} avg={sum(det_counts)/len(det_counts):.1f}")
            else:
                print(f"[DEBUG] WARNING: No frames were processed for detection!")

            for i, t in enumerate(tracker.tracks):
                print(f"[DEBUG] Robot {i}: {len(t)} tracked points")
                if t:
                    print(f"[DEBUG]   first=({t[0][0]:.1f}, {t[0][1]:.1f}) last=({t[-1][0]:.1f}, {t[-1][1]:.1f})")

            # smooth trajectories
            sw = self.smooth_window.get()
            trajectories = [smooth_trajectory(t, window=sw) for t in tracker.tracks]

            # subsample
            maxpts = 2000
            step = max(1, len(trajectories[0]) // maxpts) if trajectories[0] else 1
            subsampled = [t[::step] for t in trajectories]

            self.root.after(0, lambda: self.status.config(text="Rendering trails..."))
            print(f"[DEBUG] Subsampled step={step}, points per trail: {[len(t) for t in subsampled]}")

            # render
            rgb = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)
            base = Image.fromarray(rgb).convert("RGBA")
            print(f"[DEBUG] Base image size: {base.size}")
            tw = self.trail_width.get()
            gs = self.glow_size.get()

            for i, traj in enumerate(subsampled):
                color = TRAIL_COLORS[i % len(TRAIL_COLORS)]
                pts = [(float(x), float(y)) for x, y in traj]
                print(f"[DEBUG] Drawing trail {i}: {len(pts)} pts, first={pts[0] if pts else 'N/A'}, last={pts[-1] if pts else 'N/A'}")
                fade = self.fade_mode.get()
                base = draw_glow_trail(base, pts, color, width=tw, glow=gs, fade_mode=fade)
                if pts:
                    base = draw_head_glow(base, pts[-1], color)
                    if self.show_start_marker.get():
                        base = draw_start_marker(base, pts[0], color)
                self.root.after(0, lambda i=i: self.status.config(text=f"Rendered trail {i+1}"))

            result = base.convert("RGB")
            self.result_image = result

            self.root.after(0, lambda: self._show_result(result, trajectories))

        except Exception as e:
            self.root.after(0, lambda: self.status.config(text=f"Error: {e}"))

    def _update_progress(self, pct, msg):
        self.progress["value"] = pct
        self.status.config(text=msg)

    def _show_result(self, pil_img, trajectories):
        self.show_pil_image(pil_img)
        pts_info = ", ".join(f"R{i+1}:{len(t)}" for i, t in enumerate(trajectories))
        self.status.config(text=f"Done! Points: {pts_info}")
        self.progress["value"] = 100

    def save_result(self):
        if self.result_image is None:
            messagebox.showwarning("No result", "Run tracking first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
            initialfile="light_trails.png"
        )
        if path:
            self.result_image.save(path, quality=95)
            self.status.config(text=f"Saved: {os.path.basename(path)}")


def main():
    root = tk.Tk()
    root.geometry("1200x750")
    root.minsize(900, 600)
    app = LightTrailApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()