"""
splat2lidar desktop app.

A single native window (Open3D's gui + rendering modules) with:
  - a control panel on the left (load, filter, sensor, pose, run, export)
  - TWO live 3D viewports on the right, side by side:
      * "Gaussian splat" view -- raw/filtered splat centers, plus a
        coordinate-frame gizmo showing the sensor's current position and
        orientation, updated live as you edit the pose fields.
      * "Simulated LiDAR" view -- the binned scan result, colorable by
        range / intensity / beam index / RGB.

Run with:  python -m splat2lidar.app   (or the `splat2lidar-gui` console
script once the package is installed)
Requires a display (X11/Wayland/macOS/Windows) -- this will not run in a
headless container.
"""

import threading

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from matplotlib import cm

from .io import load_gaussian_ply, filter_splat, GaussianSplat
from .sensors import (
    SensorModel, VELODYNE_VLP16, VELODYNE_HDL64E_APPROX, OUSTER_OS1_64_APPROX,
    generic_uniform_sensor,
)
from .simulate import simulate_lidar_scan, ScanResult
from .pose import rotation_from_ypr
from .export import save_scan


SENSOR_PRESETS = {
    "Velodyne VLP-16": VELODYNE_VLP16,
    "Velodyne HDL-64E (approx)": VELODYNE_HDL64E_APPROX,
    "Ouster OS1-64 (approx)": OUSTER_OS1_64_APPROX,
    "Custom...": None,
}

COLOR_MODES = ["Range", "Intensity (opacity)", "Beam index", "RGB (from splat)"]


def _values_to_rgb(values: np.ndarray, colormap: str = "turbo") -> np.ndarray:
    v = values.astype(np.float64)
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    if hi <= lo:
        hi = lo + 1e-6
    v = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    return cm.get_cmap(colormap)(v)[:, :3]


class Splat2LidarApp:
    def __init__(self):
        self.window = gui.Application.instance.create_window("splat2lidar", 1600, 900)
        w = self.window
        em = w.theme.font_size

        self.splat: GaussianSplat = None
        self.filtered: GaussianSplat = None
        self.scan: ScanResult = None
        self._splat_path = None

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
        load_btn = gui.Button("Load .ply...")
        load_btn.set_on_clicked(self._on_load_clicked)
        self._panel.add_child(load_btn)
        self._info_label = gui.Label("No splat loaded.")
        self._panel.add_child(self._info_label)
        self._panel.add_fixed(0.5 * em)

        self._panel.add_child(gui.Label("2. Filter Gaussians"))
        self._opacity_slider = self._add_slider("Min opacity", 0.0, 1.0, 0.2)
        self._anisotropy_slider = self._add_slider("Max anisotropy ratio (min/max scale)", 0.0, 1.0, 1.0)
        self._maxscale_slider = self._add_slider("Max scale (0 = off)", 0.0, 2.0, 0.0)
        apply_btn = gui.Button("Apply filters")
        apply_btn.set_on_clicked(self._on_apply_filters)
        self._panel.add_child(apply_btn)
        self._filtered_label = gui.Label("")
        self._panel.add_child(self._filtered_label)
        self._panel.add_fixed(0.5 * em)

        self._panel.add_child(gui.Label("3. Sensor"))
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

        self._panel.add_child(gui.Label("4. Sensor pose"))
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

        run_btn = gui.Button("Simulate scan")
        run_btn.set_on_clicked(self._on_simulate_clicked)
        self._panel.add_child(run_btn)
        self._panel.add_fixed(0.25 * em)

        self._results_label = gui.Label("")
        self._panel.add_child(self._results_label)
        self._panel.add_fixed(0.5 * em)

        self._panel.add_child(gui.Label("5. Display / export"))
        self._color_combo = gui.Combobox()
        for name in COLOR_MODES:
            self._color_combo.add_item(name)
        self._color_combo.set_on_selection_changed(lambda *_: self._refresh_lidar_view())
        self._panel.add_child(self._color_combo)
        export_btn = gui.Button("Export scan...")
        export_btn.set_on_clicked(self._on_export_clicked)
        self._panel.add_child(export_btn)

        w.add_child(self._scene_splat)
        w.add_child(self._scene_lidar)
        w.add_child(self._title_splat)
        w.add_child(self._title_lidar)
        w.add_child(self._panel)
        w.set_on_layout(self._on_layout)

    # ---- layout -------------------------------------------------------------
    def _on_layout(self, layout_context):
        r = self.window.content_rect
        em = layout_context.theme.font_size
        panel_width = 22 * em
        remaining = r.width - panel_width
        half = remaining // 2
        title_h = int(1.8 * em)

        self._panel.frame = gui.Rect(r.x, r.y, panel_width, r.height)

        splat_x = r.x + panel_width
        lidar_x = splat_x + half

        self._title_splat.frame = gui.Rect(splat_x + 8, r.y + 4, half - 16, title_h)
        self._scene_splat.frame = gui.Rect(splat_x, r.y + title_h, half, r.height - title_h)

        self._title_lidar.frame = gui.Rect(lidar_x + 8, r.y + 4, remaining - half - 16, title_h)
        self._scene_lidar.frame = gui.Rect(lidar_x, r.y + title_h, remaining - half, r.height - title_h)

    # ---- small widget helpers -------------------------------------------------
    def _add_slider(self, label, lo, hi, default):
        self._panel.add_child(gui.Label(label))
        s = gui.Slider(gui.Slider.DOUBLE)
        s.set_limits(lo, hi)
        s.double_value = default
        self._panel.add_child(s)
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

    # ---- actions ----------------------------------------------------------------
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
        threading.Thread(target=work, daemon=True).start()

    def _simulated(self, scan: ScanResult):
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
        self._refresh_lidar_view()

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

    def _refresh_lidar_view(self):
        """Right viewport: the simulated scan, colored per the color-by dropdown."""
        self._scene_lidar.scene.clear_geometry()
        if self.scan is None or self.scan.num_output_points == 0:
            return
        mode = self._color_combo.selected_text
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.scan.points)
        if mode == "Range":
            pcd.colors = o3d.utility.Vector3dVector(_values_to_rgb(self.scan.ranges))
        elif mode == "Intensity (opacity)" and self.scan.intensity is not None:
            pcd.colors = o3d.utility.Vector3dVector(_values_to_rgb(self.scan.intensity))
        elif mode == "Beam index":
            pcd.colors = o3d.utility.Vector3dVector(_values_to_rgb(self.scan.beam_index.astype(np.float64)))
        elif mode == "RGB (from splat)" and self.scan.color is not None:
            pcd.colors = o3d.utility.Vector3dVector(self.scan.color)
        else:
            pcd.paint_uniform_color([1.0, 0.6, 0.0])
        self._scene_lidar.scene.add_geometry("cloud", pcd, self._point_mat)

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
    app = Splat2LidarApp()
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
