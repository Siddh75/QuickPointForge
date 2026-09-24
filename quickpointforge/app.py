"""
QuickPointForge desktop app.

A single native window (Open3D's gui + rendering modules) with:
  - a control panel on the left (load, sensor, pose, trajectory, export;
    Gaussian filter options live in a separate "Filters..." dialog)
  - TWO live 3D viewports on the right, side by side:
      * "Gaussian splat" view -- raw/filtered splat centers, plus a
        coordinate-frame gizmo showing the sensor's current position and
        orientation, updated live as you edit the pose fields.
      * "Simulated LiDAR" view -- the binned scan result, colorable by
        range / intensity / beam index / RGB / timestamp.
  - a scrub bar spanning both viewports, below them: drags preview a
    Trajectory pose (and, once a trajectory scan has been run, play back
    the sweep by time); keyframes show as dots along it, hover for values.

Run with:  python -m quickpointforge.app   (or the `quickpointforge-gui`
console script once the package is installed)
Requires a display (X11/Wayland/macOS/Windows) -- this will not run in a
headless container.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from matplotlib import colormaps

from .io import load_gaussian_ply, filter_splat, GaussianSplat
from .sensors import (
    SensorModel,
    VELODYNE_VLP16, VELODYNE_VLP32C_APPROX, VELODYNE_HDL32E_APPROX, VELODYNE_HDL64E_APPROX,
    OUSTER_OS0_128_APPROX, OUSTER_OS1_64_APPROX, OUSTER_OS2_128_APPROX,
    HESAI_PANDAR64_APPROX,
    FLASH_LIDAR_EXAMPLE, FLASH_LIDAR_NARROW_LONGRANGE_EXAMPLE,
    generic_uniform_sensor,
)
from .simulate import simulate_lidar_scan, concatenate_scans, ScanResult
from .pose import rotation_from_ypr, ypr_from_rotation, Trajectory
from .export import save_scan


SENSOR_PRESETS = {
    "Velodyne VLP-16": VELODYNE_VLP16,
    "Velodyne VLP-32C (approx)": VELODYNE_VLP32C_APPROX,
    "Velodyne HDL-32E (approx)": VELODYNE_HDL32E_APPROX,
    "Velodyne HDL-64E (approx)": VELODYNE_HDL64E_APPROX,
    "Ouster OS0-128 (approx)": OUSTER_OS0_128_APPROX,
    "Ouster OS1-64 (approx)": OUSTER_OS1_64_APPROX,
    "Ouster OS2-128 (approx)": OUSTER_OS2_128_APPROX,
    "Hesai Pandar64 (approx)": HESAI_PANDAR64_APPROX,
    "Flash LiDAR (example, 60x30 FOV)": FLASH_LIDAR_EXAMPLE,
    "Flash LiDAR (example, 20x10 FOV, long range)": FLASH_LIDAR_NARROW_LONGRANGE_EXAMPLE,
    "Custom...": None,
}

COLOR_MODES = ["Range", "Intensity (opacity)", "Beam index", "RGB (from splat)", "Timestamp"]

_MAX_KEYFRAME_MARKERS = 50  # pre-allocated pool size for scrub-bar keyframe dots
_PLAYBACK_DURATION_S = 4.0  # wall-clock time for one full Play sweep, regardless of trajectory time units


def _values_to_rgb(values: np.ndarray, colormap: str = "turbo") -> np.ndarray:
    v = values.astype(np.float64)
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    if hi <= lo:
        hi = lo + 1e-6
    v = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    return colormaps[colormap](v)[:, :3]


class QuickPointForgeApp:
    def __init__(self):
        self.window = gui.Application.instance.create_window("QuickPointForge", 1600, 900)
        w = self.window
        em = w.theme.font_size

        self.splat: GaussianSplat = None
        self.filtered: GaussianSplat = None
        self.scan: ScanResult = None
        self._splat_path = None
        self.keyframes = []  # list of (t, [x, y, z], yaw, pitch, roll)
        self._playing = False
        self._busy = False

        # --- Two 3D viewports ------------------------------------------------
        self._scene_splat = gui.SceneWidget()
        self._scene_splat.scene = rendering.Open3DScene(w.renderer)
        self._scene_splat.scene.set_background([0.08, 0.08, 0.09, 1.0])
        self._scene_splat.scene.scene.set_sun_light([-0.3, -0.5, -0.85], [1.0, 1.0, 1.0], 45000)
        self._scene_splat.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

        self._scene_lidar = gui.SceneWidget()
        self._scene_lidar.scene = rendering.Open3DScene(w.renderer)
        self._scene_lidar.scene.set_background([0.08, 0.08, 0.09, 1.0])
        self._scene_lidar.scene.scene.set_sun_light([-0.3, -0.5, -0.85], [1.0, 1.0, 1.0], 45000)
        self._scene_lidar.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

        self._title_splat = gui.Label("Gaussian splat  (sensor pose shown as axes: R=X fwd, G=Y left, B=Z up)")
        self._title_lidar = gui.Label("Simulated LiDAR scan")
        for lbl in (self._title_splat, self._title_lidar):
            lbl.text_color = gui.Color(0.75, 0.75, 0.78)

        self._point_mat = rendering.MaterialRecord()
        self._point_mat.shader = "defaultUnlit"
        self._point_mat.point_size = 2.5

        self._gizmo_mat = rendering.MaterialRecord()
        self._gizmo_mat.shader = "defaultUnlit"

        # --- Control panel -----------------------------------------------------
        self._panel = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))

        self._panel.add_child(gui.Label("1. Load splat"))
        load_row = gui.Horiz(0.25 * em)
        self._load_btn = gui.Button("Load .ply...")
        self._load_btn.set_on_clicked(self._on_load_clicked)
        load_row.add_child(self._load_btn)
        self._settings_btn = gui.Button("Filters...")
        self._settings_btn.set_on_clicked(self._on_open_filters)
        load_row.add_child(self._settings_btn)
        self._panel.add_child(load_row)
        self._info_label = gui.Label("No splat loaded.")
        self._panel.add_child(self._info_label)
        self._panel.add_fixed(0.5 * em)

        # Shown centered over the (otherwise empty) splat viewport until a
        # splat is loaded, so there's an obvious call to action there too.
        self._load_overlay_btn = gui.Button("Load Gaussian splat...")
        self._load_overlay_btn.set_on_clicked(self._on_load_clicked)

        # Gaussian filter controls live in their own floating dialog
        # (opened via the "Filters..." button above) rather than inline,
        # to keep the always-visible panel focused on sensor/pose/run.
        self._filter_dialog = gui.Dialog("Filter Gaussians")
        filter_layout = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        self._opacity_slider = self._add_slider(filter_layout, "Min opacity", 0.0, 1.0, 0.2)
        self._anisotropy_slider = self._add_slider(filter_layout, "Max anisotropy ratio (min/max scale)", 0.0, 1.0, 1.0)
        self._maxscale_slider = self._add_slider(filter_layout, "Max scale (0 = off)", 0.0, 2.0, 0.0)
        apply_btn = gui.Button("Apply filters")
        apply_btn.set_on_clicked(self._on_apply_filters)
        filter_layout.add_child(apply_btn)
        self._filtered_label = gui.Label("")
        filter_layout.add_child(self._filtered_label)
        close_filters_btn = gui.Button("Close")
        close_filters_btn.set_on_clicked(self.window.close_dialog)
        filter_layout.add_child(close_filters_btn)
        self._filter_dialog.add_child(filter_layout)

        self._panel.add_child(gui.Label("2. Sensor"))
        self._sensor_combo = gui.Combobox()
        for name in SENSOR_PRESETS:
            self._sensor_combo.add_item(name)
        self._sensor_combo.set_on_selection_changed(self._on_sensor_changed)
        self._panel.add_child(self._sensor_combo)

        self._custom_panel = gui.Vert(0.25 * em)
        self._num_beams = self._add_numedit(self._custom_panel, "Num beams", 32, is_int=True)
        self._fov_min = self._add_numedit(self._custom_panel, "FOV min (deg)", -15.0)
        self._fov_max = self._add_numedit(self._custom_panel, "FOV max (deg)", 15.0)
        self._az_res = self._add_numedit(self._custom_panel, "Azimuth resolution (deg)", 0.2)
        self._max_range = self._add_numedit(self._custom_panel, "Max range (m)", 100.0)
        self._min_range = self._add_numedit(self._custom_panel, "Min range (m)", 0.5)
        self._custom_panel.visible = False
        self._panel.add_child(self._custom_panel)
        self._panel.add_fixed(0.5 * em)

        self._panel.add_child(gui.Label("3. Sensor pose"))
        pose_grid = gui.VGrid(2, 0.25 * em)
        self._pos_x = self._grid_numedit(pose_grid, "X", 0.0)
        self._pos_y = self._grid_numedit(pose_grid, "Y", 0.0)
        self._pos_z = self._grid_numedit(pose_grid, "Z", 0.0)
        self._yaw = self._grid_numedit(pose_grid, "Yaw", 0.0)
        self._pitch = self._grid_numedit(pose_grid, "Pitch", 0.0)
        self._roll = self._grid_numedit(pose_grid, "Roll", 0.0)
        # Live-update the gizmo in the splat view as pose fields change.
        for ne in (self._pos_x, self._pos_y, self._pos_z, self._yaw, self._pitch, self._roll):
            ne.set_on_value_changed(lambda *_: self._update_gizmo())
        self._panel.add_child(pose_grid)
        self._panel.add_fixed(0.5 * em)

        self._run_btn = gui.Button("Simulate scan")
        self._run_btn.set_on_clicked(self._on_simulate_clicked)
        self._panel.add_child(self._run_btn)
        self._panel.add_fixed(0.25 * em)

        self._results_label = gui.Label("")
        self._panel.add_child(self._results_label)
        self._panel.add_fixed(0.5 * em)

        self._panel.add_child(gui.Label("4. Trajectory (optional)"))
        kf_row = gui.Horiz(0.25 * em)
        kf_row.add_child(gui.Label("t"))
        self._kf_time = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._kf_time.double_value = 0.0
        kf_row.add_child(self._kf_time)
        add_kf_btn = gui.Button("Add keyframe at current pose")
        add_kf_btn.set_on_clicked(self._on_add_keyframe)
        kf_row.add_child(add_kf_btn)
        self._panel.add_child(kf_row)

        self._kf_list = gui.ListView()
        self._kf_list.set_max_visible_items(4)
        self._panel.add_child(self._kf_list)

        clear_kf_btn = gui.Button("Clear keyframes")
        clear_kf_btn.set_on_clicked(self._on_clear_keyframes)
        self._panel.add_child(clear_kf_btn)
        self._panel.add_fixed(0.5 * em)

        sweep_row = gui.Horiz(0.25 * em)
        sweep_row.add_child(gui.Label("Samples"))
        self._num_samples = gui.NumberEdit(gui.NumberEdit.INT)
        self._num_samples.int_value = 10
        sweep_row.add_child(self._num_samples)
        self._panel.add_child(sweep_row)

        self._sweep_btn = gui.Button("Simulate over trajectory")
        self._sweep_btn.set_on_clicked(self._on_simulate_trajectory)
        self._panel.add_child(self._sweep_btn)
        self._panel.add_fixed(0.5 * em)

        self._panel.add_child(gui.Label("5. Display / export"))
        self._color_combo = gui.Combobox()
        for name in COLOR_MODES:
            self._color_combo.add_item(name)
        self._color_combo.set_on_selection_changed(lambda *_: self._refresh_lidar_view())
        self._panel.add_child(self._color_combo)
        self._export_btn = gui.Button("Export scan...")
        self._export_btn.set_on_clicked(self._on_export_clicked)
        self._panel.add_child(self._export_btn)

        # --- Scrub bar: spans both viewports, below them -----------------------
        self._play_btn = gui.Button("Play")
        self._play_btn.enabled = False
        self._play_btn.set_on_clicked(self._on_play_clicked)

        self._scrub_label = gui.Label("Scrub time")
        self._scrub_label.text_color = gui.Color(0.75, 0.75, 0.78)
        self._scrub_slider = gui.Slider(gui.Slider.DOUBLE)
        self._scrub_slider.set_limits(0.0, 1.0)
        self._scrub_slider.double_value = 0.0
        self._scrub_slider.enabled = False
        self._scrub_slider.set_on_value_changed(self._on_scrub)

        self._accumulate_checkbox = gui.Checkbox("Accumulate")
        self._accumulate_checkbox.checked = True
        self._accumulate_checkbox.tooltip = "On: show every point scanned up to the scrub time. Off: only the single nearest frame."
        self._accumulate_checkbox.set_on_checked(lambda *_: self._refresh_lidar_view())

        # Loading indicator drawn over the whole scrub bar while a
        # simulation runs in its background thread (see _set_busy).
        self._progress_bar = gui.ProgressBar()
        self._progress_bar.visible = False

        # Fixed pool of keyframe-marker dots (Open3D widgets can't be
        # removed once added, so we pre-allocate and show/hide+reposition
        # instead) drawn on top of the slider; hover shows that keyframe's
        # time/pose via the native tooltip.
        self._kf_markers = []
        for _ in range(_MAX_KEYFRAME_MARKERS):
            marker = gui.Label("*")
            marker.text_color = gui.Color(1.0, 0.65, 0.0)
            marker.visible = False
            self._kf_markers.append(marker)

        w.add_child(self._scene_splat)
        w.add_child(self._scene_lidar)
        w.add_child(self._title_splat)
        w.add_child(self._title_lidar)
        w.add_child(self._play_btn)
        w.add_child(self._scrub_label)
        w.add_child(self._scrub_slider)
        w.add_child(self._accumulate_checkbox)
        for marker in self._kf_markers:
            w.add_child(marker)  # after the slider -> drawn on top of it
        w.add_child(self._progress_bar)  # after markers -> covers the whole bar while busy
        w.add_child(self._panel)
        w.add_child(self._load_overlay_btn)  # added last -> drawn on top of the scene
        w.set_on_layout(self._on_layout)

    # ---- layout -------------------------------------------------------------
    def _on_layout(self, layout_context):
        r = self.window.content_rect
        em = layout_context.theme.font_size
        panel_width = 22 * em
        remaining = r.width - panel_width
        half = remaining // 2
        title_h = int(1.8 * em)
        scrub_h = int(2.6 * em)
        scene_h = r.height - title_h - scrub_h

        self._panel.frame = gui.Rect(r.x, r.y, panel_width, r.height)

        splat_x = r.x + panel_width
        lidar_x = splat_x + half

        self._title_splat.frame = gui.Rect(splat_x + 8, r.y + 4, half - 16, title_h)
        self._scene_splat.frame = gui.Rect(splat_x, r.y + title_h, half, scene_h)

        self._title_lidar.frame = gui.Rect(lidar_x + 8, r.y + 4, remaining - half - 16, title_h)
        self._scene_lidar.frame = gui.Rect(lidar_x, r.y + title_h, remaining - half, scene_h)

        # Scrub bar: spans the full width of both viewports, below them.
        scrub_y = r.y + title_h + scene_h
        play_w = int(4.5 * em)
        label_w = int(7 * em)
        checkbox_w = int(8 * em)
        slider_w = remaining - play_w - label_w - checkbox_w - 32
        play_x = splat_x + 8
        label_x = play_x + play_w + 8
        slider_x = label_x + label_w + 8
        checkbox_x = slider_x + slider_w + 8
        self._play_btn.frame = gui.Rect(play_x, scrub_y, play_w, scrub_h)
        self._scrub_label.frame = gui.Rect(label_x, scrub_y, label_w, scrub_h)
        self._scrub_slider.frame = gui.Rect(slider_x, scrub_y, slider_w, scrub_h)
        self._accumulate_checkbox.frame = gui.Rect(checkbox_x, scrub_y, checkbox_w, scrub_h)
        self._progress_bar.frame = gui.Rect(play_x, scrub_y, checkbox_x + checkbox_w - play_x, scrub_h)
        self._position_keyframe_markers()

        if self._load_overlay_btn.visible:
            constraint = gui.Widget.Constraints()
            constraint.width = half
            constraint.height = scene_h
            pref = self._load_overlay_btn.calc_preferred_size(layout_context, constraint)
            btn_x = self._scene_splat.frame.x + (self._scene_splat.frame.width - pref.width) // 2
            btn_y = self._scene_splat.frame.y + (self._scene_splat.frame.height - pref.height) // 2
            self._load_overlay_btn.frame = gui.Rect(btn_x, btn_y, pref.width, pref.height)

    def _position_keyframe_markers(self):
        """Dots along the scrub slider marking each keyframe's time, drawn
        on top of it (see the pool built in __init__); hover for a tooltip
        with that keyframe's pose."""
        slider_frame = self._scrub_slider.frame
        marker_size = int(0.9 * self.window.theme.font_size)
        times = [k[0] for k in self.keyframes]
        have_range = len(times) >= 2
        lo, span = (min(times), max(times) - min(times)) if have_range else (0.0, 1.0)
        span = span or 1.0
        for i, marker in enumerate(self._kf_markers):
            if not have_range or i >= len(self.keyframes):
                marker.visible = False
                continue
            t, p, yaw, pitch, roll = self.keyframes[i]
            frac = (t - lo) / span
            x = int(slider_frame.x + frac * slider_frame.width - marker_size / 2)
            y = int(slider_frame.y + slider_frame.height / 2 - marker_size / 2)
            marker.frame = gui.Rect(x, y, marker_size, marker_size)
            marker.tooltip = (
                f"t={t:g}\npos=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})\nypr=({yaw:.1f}, {pitch:.1f}, {roll:.1f})"
            )
            marker.visible = True

    # ---- small widget helpers -------------------------------------------------
    def _add_slider(self, parent, label, lo, hi, default):
        parent.add_child(gui.Label(label))
        s = gui.Slider(gui.Slider.DOUBLE)
        s.set_limits(lo, hi)
        s.double_value = default
        parent.add_child(s)
        return s

    def _add_numedit(self, parent, label, default, is_int=False):
        row = gui.Horiz(0.25 * self.window.theme.font_size)
        row.add_child(gui.Label(label))
        ne = gui.NumberEdit(gui.NumberEdit.INT if is_int else gui.NumberEdit.DOUBLE)
        if is_int:
            ne.int_value = int(default)
        else:
            ne.double_value = float(default)
        row.add_child(ne)
        parent.add_child(row)
        return ne

    def _grid_numedit(self, grid, label, default):
        grid.add_child(gui.Label(label))
        ne = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        ne.double_value = default
        grid.add_child(ne)
        return ne

    # ---- pose / gizmo ---------------------------------------------------------
    def _current_pose(self):
        position = np.array([self._pos_x.double_value, self._pos_y.double_value, self._pos_z.double_value])
        rotation = rotation_from_ypr(self._yaw.double_value, self._pitch.double_value, self._roll.double_value)
        return position, rotation

    def _gizmo_size(self) -> float:
        src = self.filtered if self.filtered is not None else self.splat
        if src is None or len(src) == 0:
            return 1.0
        diag = np.linalg.norm(src.centers.max(axis=0) - src.centers.min(axis=0))
        return max(0.3, 0.06 * diag)

    def _update_gizmo(self):
        if self.splat is None:
            return
        position, rotation = self._current_pose()
        size = self._gizmo_size()

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = position
        frame.transform(T)

        # A small sphere at the origin keeps the sensor visible even when
        # zoomed out far enough that the axes themselves are hard to see.
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size * 0.12)
        sphere.paint_uniform_color([1.0, 1.0, 1.0])
        sphere.translate(position)

        if self._scene_splat.scene.has_geometry("sensor_gizmo"):
            self._scene_splat.scene.remove_geometry("sensor_gizmo")
        if self._scene_splat.scene.has_geometry("sensor_origin"):
            self._scene_splat.scene.remove_geometry("sensor_origin")
        self._scene_splat.scene.add_geometry("sensor_gizmo", frame, self._gizmo_mat)
        self._scene_splat.scene.add_geometry("sensor_origin", sphere, self._gizmo_mat)

    def _set_pose_fields(self, position, rotation):
        """Push a (position, rotation_matrix) pose into the pose numedits and
        refresh the gizmo -- used to preview a Trajectory-interpolated pose."""
        self._pos_x.double_value, self._pos_y.double_value, self._pos_z.double_value = (float(v) for v in position)
        self._yaw.double_value, self._pitch.double_value, self._roll.double_value = ypr_from_rotation(rotation)
        self._update_gizmo()

    # ---- trajectory / keyframes ------------------------------------------------
    def _refresh_keyframe_list(self):
        self._kf_list.set_items([
            f"t={t:g}  pos=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})  ypr=({y:.1f}, {pi:.1f}, {r:.1f})"
            for t, p, y, pi, r in self.keyframes
        ])
        has_traj = len(self.keyframes) >= 2
        if has_traj:
            times = [k[0] for k in self.keyframes]
            self._scrub_slider.set_limits(min(times), max(times))
            self._scrub_slider.double_value = min(times)
        else:
            self._stop_playback()
        self._scrub_slider.enabled = has_traj and not self._busy
        self._play_btn.enabled = has_traj and not self._busy
        self.window.set_needs_layout()  # reposition keyframe-marker dots

    def _on_add_keyframe(self):
        t = self._kf_time.double_value
        self.keyframes = [k for k in self.keyframes if k[0] != t]
        self.keyframes.append((
            t,
            [self._pos_x.double_value, self._pos_y.double_value, self._pos_z.double_value],
            self._yaw.double_value, self._pitch.double_value, self._roll.double_value,
        ))
        self.keyframes.sort(key=lambda k: k[0])
        self._refresh_keyframe_list()

    def _on_clear_keyframes(self):
        self.keyframes = []
        self._refresh_keyframe_list()

    def _on_scrub(self, t):
        if len(self.keyframes) >= 2:
            traj = Trajectory.from_keyframes(self.keyframes)
            position, rotation = traj.pose_at(t)
            self._set_pose_fields(position, rotation)
        if self.scan is not None and self.scan.timestamp is not None:
            self._refresh_lidar_view()

    def _set_scrub(self, t):
        """Programmatic scrub (from Play) -- setting double_value alone
        doesn't fire the slider's on_value_changed callback."""
        self._scrub_slider.double_value = t
        self._on_scrub(t)

    def _stop_playback(self):
        self._playing = False
        self._play_btn.text = "Play"

    def _on_play_clicked(self):
        if self._playing:
            self._stop_playback()
            return
        if not self._scrub_slider.enabled:
            return
        self._playing = True
        self._play_btn.text = "Stop"
        lo = self._scrub_slider.get_minimum_value
        hi = self._scrub_slider.get_maximum_value
        start_wall = time.time()

        def work():
            while self._playing:
                frac = min(1.0, (time.time() - start_wall) / _PLAYBACK_DURATION_S)
                t = lo + frac * (hi - lo)
                gui.Application.instance.post_to_main_thread(self.window, lambda t=t: self._set_scrub(t))
                if frac >= 1.0:
                    break
                time.sleep(1.0 / 30)
            gui.Application.instance.post_to_main_thread(self.window, self._stop_playback)

        threading.Thread(target=work, daemon=True).start()

    def _set_busy(self, busy: bool):
        """Grey out the load/filter/simulate/export/play controls and show
        a loading bar over the scrub bar while a simulation runs in its
        background thread."""
        self._busy = busy
        for btn in (self._load_btn, self._settings_btn, self._run_btn, self._sweep_btn, self._export_btn):
            btn.enabled = not busy
        self._play_btn.enabled = (not busy) and len(self.keyframes) >= 2
        self._scrub_slider.enabled = (not busy) and len(self.keyframes) >= 2
        self._progress_bar.visible = busy
        if busy:
            self._progress_bar.value = 0.0
            threading.Thread(target=self._animate_progress, daemon=True).start()

    def _animate_progress(self):
        """Indeterminate marquee: sweep the bar 0->1 on a fixed period for
        as long as _busy stays True (there's no real fraction-done to show
        for a single simulate_lidar_scan call)."""
        start = time.time()
        period_s = 1.0
        while self._busy:
            frac = ((time.time() - start) % period_s) / period_s
            gui.Application.instance.post_to_main_thread(self.window, lambda frac=frac: setattr(self._progress_bar, "value", frac))
            time.sleep(1.0 / 20)

    def _on_simulate_trajectory(self):
        if self.filtered is None:
            self._results_label.text = "Load a splat first."
            return
        if len(self.keyframes) < 2:
            self._results_label.text = "Add at least 2 keyframes first."
            return
        sensor = self._current_sensor()
        traj = Trajectory.from_keyframes(self.keyframes)
        times = np.linspace(traj.times[0], traj.times[-1], max(2, self._num_samples.int_value))

        def scan_at(t):
            position, rotation = traj.pose_at(t)
            return simulate_lidar_scan(
                self.filtered, sensor, position, rotation, return_frame="world", timestamp=float(t),
            )

        def work():
            # Poses are independent and numpy releases the GIL, so scans run
            # in parallel. ponytail: 4 workers ~ 2.7x on a 3.8M-splat scene;
            # each holds ~0.5 GB of temporaries per 4M splats, so raise only
            # if RAM allows (gains flatten past ~6).
            with ThreadPoolExecutor(max_workers=4) as pool:
                merged = concatenate_scans(list(pool.map(scan_at, times)))
            gui.Application.instance.post_to_main_thread(self.window, lambda: self._simulated(merged))

        self._results_label.text = "Simulating over trajectory..."
        self._set_busy(True)
        threading.Thread(target=work, daemon=True).start()

    # ---- actions ----------------------------------------------------------------
    def _on_open_filters(self):
        self.window.show_dialog(self._filter_dialog)

    def _on_load_clicked(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Load Gaussian splat (.ply)", self.window.theme)
        dlg.add_filter(".ply", "3D Gaussian splat PLY")
        dlg.set_on_cancel(self.window.close_dialog)
        dlg.set_on_done(self._do_load)
        self.window.show_dialog(dlg)

    def _do_load(self, path):
        self.window.close_dialog()
        self._splat_path = path

        def work():
            splat = load_gaussian_ply(path)
            gui.Application.instance.post_to_main_thread(self.window, lambda: self._loaded(splat))

        self._info_label.text = "Loading..."
        threading.Thread(target=work, daemon=True).start()

    def _loaded(self, splat):
        self.splat = splat
        self.filtered = splat
        bounds_min = splat.centers.min(axis=0)
        bounds_max = splat.centers.max(axis=0)
        self._info_label.text = (
            f"{len(splat):,} Gaussians\n"
            f"bounds: [{bounds_min[0]:.2f},{bounds_min[1]:.2f},{bounds_min[2]:.2f}] to "
            f"[{bounds_max[0]:.2f},{bounds_max[1]:.2f},{bounds_max[2]:.2f}]"
        )
        self._load_overlay_btn.visible = False
        self._refresh_splat_view()
        self._update_gizmo()
        self.window.set_needs_layout()

    def _on_apply_filters(self):
        if self.splat is None:
            return
        max_scale = self._maxscale_slider.double_value
        self.filtered = filter_splat(
            self.splat,
            min_opacity=self._opacity_slider.double_value,
            max_scale=None if max_scale <= 0 else max_scale,
            max_anisotropy_ratio=None if self._anisotropy_slider.double_value >= 1.0 else self._anisotropy_slider.double_value,
        )
        self._filtered_label.text = f"{len(self.filtered):,} / {len(self.splat):,} remain after filtering"
        self._refresh_splat_view()
        self._update_gizmo()

    def _on_sensor_changed(self, name, idx):
        self._custom_panel.visible = (name == "Custom...")
        self.window.set_needs_layout()

    def _current_sensor(self) -> SensorModel:
        name = self._sensor_combo.selected_text
        preset = SENSOR_PRESETS.get(name)
        if preset is not None:
            return preset
        return generic_uniform_sensor(
            name="Custom",
            num_beams=self._num_beams.int_value,
            fov_min_deg=self._fov_min.double_value,
            fov_max_deg=self._fov_max.double_value,
            azimuth_resolution_deg=self._az_res.double_value,
            max_range_m=self._max_range.double_value,
            min_range_m=self._min_range.double_value,
        )

    def _on_simulate_clicked(self):
        if self.filtered is None:
            self._results_label.text = "Load a splat first."
            return
        sensor = self._current_sensor()
        position, rotation = self._current_pose()

        def work():
            scan = simulate_lidar_scan(self.filtered, sensor, position, rotation, return_frame="world")
            gui.Application.instance.post_to_main_thread(self.window, lambda: self._simulated(scan))

        self._results_label.text = "Simulating..."
        self._set_busy(True)
        threading.Thread(target=work, daemon=True).start()

    def _simulated(self, scan: ScanResult):
        self._set_busy(False)
        self.scan = scan
        hit = scan.hit_rate()
        hit_str = f"{hit:.1%}" if hit is not None else "n/a"
        rng_str = "n/a"
        if scan.num_output_points > 0:
            rng_str = f"min={scan.ranges.min():.2f} mean={scan.ranges.mean():.2f} max={scan.ranges.max():.2f}"
        self._results_label.text = (
            f"input={scan.num_input_points:,}  output={scan.num_output_points:,}\n"
            f"hit rate={hit_str}\nrange (m): {rng_str}"
        )
        self._refresh_lidar_view(reset_camera=True)

    def _refresh_splat_view(self):
        """Left viewport: raw/filtered Gaussian centers (the sensor gizmo is
        managed separately by _update_gizmo so pose edits don't require
        rebuilding the whole point cloud)."""
        self._scene_splat.scene.clear_geometry()
        src = self.filtered if self.filtered is not None else self.splat
        if src is None:
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(src.centers)
        if src.color is not None:
            pcd.colors = o3d.utility.Vector3dVector(src.color)
        else:
            pcd.paint_uniform_color([0.6, 0.6, 0.6])
        self._scene_splat.scene.add_geometry("cloud", pcd, self._point_mat)

        bounds = self._scene_splat.scene.bounding_box
        self._scene_splat.setup_camera(60, bounds, bounds.get_center())

    def _refresh_lidar_view(self, reset_camera: bool = False):
        """Right viewport: the simulated scan, colored per the color-by dropdown.
        If the scan carries timestamps and the scrub slider is active, either
        only the single nearest-time frame or everything scanned up to the
        scrubbed time is shown, per the Accumulate checkbox.

        reset_camera: only True right after a fresh simulation -- scrubbing
        or recoloring an existing scan must not reset the viewer's zoom/pan.
        """
        self._scene_lidar.scene.clear_geometry()
        if self.scan is None or self.scan.num_output_points == 0:
            return
        mode = self._color_combo.selected_text
        points = self.scan.points
        if mode == "Range":
            colors = _values_to_rgb(self.scan.ranges)
        elif mode == "Intensity (opacity)" and self.scan.intensity is not None:
            colors = _values_to_rgb(self.scan.intensity)
        elif mode == "Beam index":
            colors = _values_to_rgb(self.scan.beam_index.astype(np.float64))
        elif mode == "RGB (from splat)" and self.scan.color is not None:
            colors = self.scan.color
        elif mode == "Timestamp" and self.scan.timestamp is not None:
            colors = _values_to_rgb(self.scan.timestamp)
        else:
            colors = np.tile([1.0, 0.6, 0.0], (len(points), 1))

        if self.scan.timestamp is not None and self._scrub_slider.enabled:
            t = self._scrub_slider.double_value
            if self._accumulate_checkbox.checked:
                mask = self.scan.timestamp <= t
            else:
                frame_times = np.unique(self.scan.timestamp)
                nearest = frame_times[np.argmin(np.abs(frame_times - t))]
                mask = self.scan.timestamp == nearest
            points, colors = points[mask], colors[mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        self._scene_lidar.scene.add_geometry("cloud", pcd, self._point_mat)

        if not reset_camera:
            return
        bounds = self._scene_lidar.scene.bounding_box
        self._scene_lidar.setup_camera(60, bounds, bounds.get_center())

    def _on_export_clicked(self):
        if self.scan is None:
            self._results_label.text = "Run a simulation first."
            return
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Export simulated scan", self.window.theme)
        for ext, desc in [(".pcd", "PCD point cloud"), (".ply", "PLY point cloud"), (".npy", "NumPy array")]:
            dlg.add_filter(ext, desc)
        dlg.set_on_cancel(self.window.close_dialog)
        dlg.set_on_done(self._do_export)
        self.window.show_dialog(dlg)

    def _do_export(self, path):
        self.window.close_dialog()
        save_scan(self.scan, path)
        self._results_label.text += f"\nSaved to {path}"


def main():
    gui.Application.instance.initialize()
    app = QuickPointForgeApp()
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
