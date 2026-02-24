import pickle

import hydra
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math

from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons
from ware_ops_algos.algorithms import RouteNode

class SimulationVisualizer:
    def __init__(self, layout, log):
        # --- Initialization ------------------------------------------------
        self.log = log
        self.interval = 1
        self.num_frames = len(log)
        self.current_frame = 0
        self.is_playing = False
        self._slider_updating = False
        self._slider_was_playing = False
        self.anim = None

        # --- Graph & positions ----------------------------------------------
        self.G = layout.layout_network.graph
        self.pos = nx.get_node_attributes(self.G, "pos")

        # --- Picker colors (skip default blue) ------------------------------
        default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color_cycle = [c for c in default_cycle if c.lower() not in ("#1f77b4", "b", "blue")]
        first_snapshot = log[0]
        self.picker_colors = {
            p.id: color_cycle[i % len(color_cycle)]
            for i, p in enumerate(first_snapshot["pickers"].resources)
        }

        # --- Matplotlib base figure -----------------------------------------
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.12)
        nx.draw(self.G, self.pos, ax=self.ax, with_labels=True, node_size=300, font_size=4, font_color="w")

        # --- Picker markers & status texts ----------------------------------
        self.picker_markers = {
            p.id: self.ax.plot([], [], "o", color=self.picker_colors[p.id], markersize=13)[0]
            for p in first_snapshot["pickers"].resources
        }
        self.status_texts = {
            p.id: self.ax.text(0, 0, "", fontsize=8, color=self.picker_colors[p.id])
            for p in first_snapshot["pickers"].resources
        }

        # --- Route lines container -----------------------------------------
        self.route_lines: dict[int, list] = {}

        # --- Info texts -----------------------------------------------------
        self.event_text = self.ax.text(0.01, 0.95, "", transform=self.ax.transAxes, fontsize=10)
        self.time_text = self.ax.text(0.01, 0.91, "", transform=self.ax.transAxes, fontsize=10)

        # --- Widgets --------------------------------------------------------
        # Frame slider
        ax_slider = self.fig.add_axes([0.06, 0.08, 0.73, 0.03], facecolor="lightgoldenrodyellow")
        self.slider = Slider(ax_slider, "Frame", 0, self.num_frames - 1, valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider_change)
        self.fig.canvas.mpl_connect("button_release_event", self.on_slider_release)

        # Play/Pause button
        ax_play = self.fig.add_axes([0.06, 0.03, 0.10, 0.04], facecolor="lightgoldenrodyellow")
        self.button_play = Button(ax_play, "Play")
        self.button_play.on_clicked(self.toggle_animation)

        # Previous frame button
        ax_prev = self.fig.add_axes([0.18, 0.03, 0.04, 0.04], facecolor="lightgoldenrodyellow")
        self.button_prev = Button(ax_prev, "<")
        self.button_prev.on_clicked(self.on_button_prev)

        # Next frame button
        ax_next = self.fig.add_axes([0.23, 0.03, 0.04, 0.04], facecolor="lightgoldenrodyellow")
        self.button_next = Button(ax_next, ">")
        self.button_next.on_clicked(self.on_button_next)

        # Speed selection
        ax_speed = self.fig.add_axes([0.84, 0.01, 0.15, 0.10], facecolor="lightgoldenrodyellow")
        self.speed_dropdown = RadioButtons(
            ax_speed,
            ["As fast as possible", "10 Frames/s", "5 Frames/s", "2 Frames/s"]
        )
        self.speed_dropdown.set_active(0)
        self.speed_dropdown.on_clicked(self.on_speed_change)

        # Route visibility toggle
        ax_route = self.fig.add_axes([0.60, 0.01, 0.18, 0.10], facecolor="lightgoldenrodyellow")
        self.route_radio = RadioButtons(ax_route, ["No Routes", "All Routes"])
        self.route_radio.on_clicked(self.on_route_toggle)
        self.route_visibility = "No Routes"

        # Display initial frame
        self.update(0)
        self.fig.canvas.draw()

    # --- Helper functions for route geometry -------------------------------
    def _shift_point_perp(self, u, v, p, offset):
        """Shift point p by 'offset' perpendicular to edge u->v."""
        if offset == 0:
            return p
        x1, y1 = self.pos[u]
        x2, y2 = self.pos[v]
        dx, dy = x2 - x1, y2 - y1
        perp_x, perp_y = dy, -dx
        length = math.hypot(perp_x, perp_y)
        if length == 0:
            return p
        nx_, ny_ = perp_x / length, perp_y / length
        return p[0] + offset * nx_, p[1] + offset * ny_

    @staticmethod
    def _offset_intersection(p1, p2, p3, p4):
        """Compute intersection of lines (p1-p2) and (p3-p4)."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-9:
            return p2
        px = round(((x1 * y2 - y1 * x2) * (x3 - x4)
              - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom, 2)
        py = round(((x1 * y2 - y1 * x2) * (y3 - y4)
              - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom, 2)
        return px, py

    # --- Update route lines based on current snapshot -----------------------
    def _update_routes(self, snapshot):
        # 1) Draw path for each picker
        STEP = 0.05  # base offset for parallel lines
        edge_levels: dict[tuple[int, tuple[int, int]], set[int]] = {}

        for picker in snapshot["pickers"].resources:
            pid = picker.id
            # remove old lines
            for ln in self.route_lines.get(pid, []):
                ln.remove()
            self.route_lines[pid] = []

            tour = None
            # tour = picker.tour
            if tour is None:
                continue

            color = self.picker_colors[pid]

            # split tour into straight segments
            segments = []
            cur_edges = []
            cur_dir = None
            for u, v in zip(tour.route.route_nodes, tour.route.route_nodes[1:]):
                dx, dy = v[0] - u[0], v[1] - u[1]
                if cur_dir is None:
                    cur_dir = (dx, dy)
                    cur_edges = [(u, v)]
                else:
                    cross = cur_dir[0] * dy - cur_dir[1] * dx
                    dot   = cur_dir[0] * dx + cur_dir[1] * dy
                    if abs(cross) < 1e-9 and dot > 0:
                        cur_edges.append((u, v))
                    else:
                        segments.append(cur_edges)
                        cur_dir = (dx, dy)
                        cur_edges = [(u, v)]
            if cur_edges:
                segments.append(cur_edges)

            # assign offset levels
            segments_w_offsets = []
            for seg_idx, edges in enumerate(segments):
                used_levels = set()
                extra_keys = []
                # collect levels from adjacent edges
                for u, v in edges:
                    used_levels |= edge_levels.setdefault((u, (u, v)), set())
                    used_levels |= edge_levels.setdefault((v, (u, v)), set())

                # handle segment transitions to avoid overlaps
                # segment start
                if seg_idx > 0:
                    u, v = edges[0]
                    prev_u, prev_v = segments[seg_idx-1][-1]

                    dx1, dy1 = prev_v[0]-prev_u[0], prev_v[1]-prev_u[1]
                    dx2, dy2 = v[0]-u[0], v[1]-u[1]

                    cross = dx1*dy2 - dy1*dx2

                    if cross > 0:
                        # left turn
                        start_node = u
                        for neighbor in self.G.neighbors(start_node):
                            if neighbor == v:
                                continue
                            dx3, dy3 = (start_node[0] - neighbor[0], start_node[1] - neighbor[1])
                            if abs(dx2*dy3 - dy2*dx3) < 1e-9 and dx2*dx3 + dy2*dy3 > 0:
                                key = (start_node, (neighbor, start_node))
                                used_levels |= edge_levels.get(key, set())
                                #extra_keys.append(key) #is theoretically not needed, as other routes that use the level at this point theoretically do not lead to overlaps
                                break
                # segment end
                if seg_idx < len(segments) - 1:
                    u, v = edges[-1]
                    nxt_u, nxt_v = segments[seg_idx+1][0]

                    dx1, dy1 = v[0]-u[0], v[1]-u[1]
                    dx2, dy2 = nxt_v[0]-nxt_u[0], nxt_v[1]-nxt_u[1]

                    cross = dx1*dy2 - dy1*dx2
                    dot   = dx1*dx2 + dy1*dy2

                    if cross > 0 or (cross < 0 and dot > 0):
                        # left turn or blunt right turn
                        end_node = v
                        for neighbor in self.G.neighbors(end_node):
                            if neighbor == u:
                                continue
                            dx3, dy3 = (neighbor[0] - end_node[0], neighbor[1] - end_node[1])
                            if abs(dx1*dy3 - dy1*dx3) < 1e-9 and dx1*dx3 + dy1*dy3 > 0:
                                key = (end_node, (end_node, neighbor))
                                used_levels |= edge_levels.get(key, set())
                                extra_keys.append(key)
                                break
                
                level = 1
                while level in used_levels:
                    level += 1
                for u, v in edges:
                    edge_levels[(u, (u, v))].add(level)
                    edge_levels[(v, (u, v))].add(level)
                for key in extra_keys:
                    edge_levels.setdefault(key, set()).add(level)
                segments_w_offsets.append((edges, STEP * level))

            # build and smooth points
            seg_pts = []
            for segment, offset in segments_w_offsets:
                pts = [self._shift_point_perp(segment[0][0], segment[0][1], self.pos[segment[0][0]], offset)]
                for u, v in segment:
                    pts.append(self._shift_point_perp(u, v, self.pos[v], offset))
                seg_pts.append(pts)

            merged_pts = [seg_pts[0][0]]
            for i, cur in enumerate(seg_pts):
                if i > 0:
                    prev = seg_pts[i - 1]

                    dx1 = prev[-1][0] - prev[-2][0]
                    dy1 = prev[-1][1] - prev[-2][1]
                    dx2 = cur[1][0] - cur[0][0]
                    dy2 = cur[1][1] - cur[0][1]

                    cross = dx1 * dy2 - dy1 * dx2
                    dot   = dx1 * dx2 + dy1 * dy2

                    # check for U-turns vs. smooth bends
                    if abs(cross) < 1e-9 and dot < 0:
                        merged_pts.append(cur[0])
                    else:
                        inter = self._offset_intersection(prev[-2], prev[-1], cur[0], cur[1])
                        merged_pts[-1] = inter
                        cur[0] = inter
                merged_pts.extend(cur[1:])

            xs, ys = zip(*merged_pts)
            ln, = self.ax.plot(xs, ys, "-", color=color, lw=2, alpha=0.7)
            self.route_lines[pid] = [ln]

        # 2) Toggle visibility of all routes
        visible = (self.route_visibility == "All Routes")
        for lines  in self.route_lines.values():
            for ln in lines:
                ln.set_visible(visible)


    # --- Update display for a given frame ----------------------------------
    def update(self, frame):
        self.current_frame = frame
        snap = self.log[frame]

        self.event_text.set_text(f"Event: {snap['event']}")
        # self.time_text.set_text(f"Time: {snap['time'].strftime('%Y-%m-%d %H:%M:%S')}")
        self.time_text.set_text(f"Time: {snap['time']}")

        pos_counter = {}
        for p in snap["pickers"].resources:
            if isinstance(p.current_location, RouteNode):
                current_location = p.current_location.position
            else:
                current_location = p.current_location

            self.picker_markers[p.id].set_data([current_location[0]], [current_location[1]])
            pos_counter[current_location] = pos_counter.get(current_location, 0) + 1
            offset_y = 0.25 * pos_counter[current_location]
            self.status_texts[p.id].set_position((current_location[0] + 0.06, current_location[1] + offset_y))
            self.status_texts[p.id].set_text(f"P{p.id}: {p.tpe}")

        # refresh routes
        self._update_routes(snap)

        # sync slider position
        if not self._slider_updating and self.slider.val != frame:
            self._slider_updating = True
            self.slider.set_val(frame)
            self._slider_updating = False

        # auto-pause at last frame
        if frame == self.num_frames - 1 and self.anim:
            self.anim.event_source.stop()
            self.button_play.label.set_text("Play")
            self.is_playing = False

        return (
            list(self.picker_markers.values()) +
            list(self.status_texts.values()) +
            [ln for sub in self.route_lines.values() for ln in sub] +
            [self.event_text, self.time_text]
        )

    # --- UI callback functions --------------------------------------------
    def on_slider_change(self, val):
        """Handle manual slider movement."""
        if self._slider_updating:
            return
        if self.is_playing and self.anim:
            self.anim.event_source.stop()
            self._slider_was_playing = True
            self.is_playing = False
        self.update(int(val))
        self.fig.canvas.draw_idle()

    def on_slider_release(self, event):
        """Resume animation if slider released while playing."""
        if event.inaxes == self.slider.ax and self._slider_was_playing:
            if self.current_frame < self.num_frames - 1:
                self.toggle_animation(None)
            self._slider_was_playing = False

    def on_button_prev(self, _):
        """Go back one frame."""
        if self.is_playing and self.anim:
            self.anim.event_source.stop()
            self.button_play.label.set_text("Play")
            self.is_playing = False
        self.slider.set_val(max(self.current_frame - 1, 0))

    def on_button_next(self, _):
        """Advance one frame."""
        if self.is_playing and self.anim:
            self.anim.event_source.stop()
            self.button_play.label.set_text("Play")
            self.is_playing = False
        self.slider.set_val(min(self.current_frame + 1, self.num_frames - 1))

    def on_speed_change(self, label):
        """Adjust animation speed."""
        self.interval = {"As fast as possible": 1,
                         "10 Frames/s": 100,
                         "5 Frames/s": 200,
                         "2 Frames/s": 500}[label]
        if self.anim and self.is_playing:
            self.anim.event_source.stop()
            self.anim = FuncAnimation(
                self.fig, self.update,
                frames=range(self.current_frame, self.num_frames),
                interval=self.interval, blit=False, repeat=False
            )
        self.fig.canvas.draw_idle()

    def on_route_toggle(self, label):
        """Toggle route visibility."""
        self.route_visibility = label
        self._update_routes(self.log[self.current_frame])
        self.fig.canvas.draw_idle()

    def toggle_animation(self, _):
        """Play or pause the animation."""
        if self.is_playing:
            if self.anim:
                self.anim.event_source.stop()
            self.button_play.label.set_text("Play")
            self.is_playing = False
        else:
            if self.current_frame == self.num_frames - 1:
                self.slider.set_val(0)
                self.current_frame = 0
            self.anim = FuncAnimation(
                self.fig, self.update,
                frames=range(self.current_frame, self.num_frames),
                interval=self.interval, blit=False, repeat=False
            )
            self.button_play.label.set_text("Pause")
            self.is_playing = True
        self.fig.canvas.draw_idle()

    # --- Display the animation --------------------------------------------
    def show(self):
        """Show the animation window."""
        plt.show(block=True)


def main():
    with open("../../experiments/outputs/2026-01-27/18-22-19/logs.pkl", "rb") as f:
        log = pickle.load(f)

    with open("../../data/instances/caches/sim/dynamic_info.pkl", "rb") as f:
        domain = pickle.load(f)
    layout = domain.layout
    visualizer = SimulationVisualizer(layout=layout, log=log)
    plt.show()


if __name__ == "__main__":
    main()
