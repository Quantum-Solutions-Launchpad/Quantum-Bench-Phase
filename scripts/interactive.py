import numpy as np
from mpl_toolkits.mplot3d.proj3d import proj_transform


def attach_hover(fig, ax, point_sets):
    all_xs = np.concatenate([np.asarray(ps["xs"], dtype=float) for ps in point_sets])
    all_ys = np.concatenate([np.asarray(ps["ys"], dtype=float) for ps in point_sets])
    all_zs = np.concatenate([np.asarray(ps["zs"], dtype=float) for ps in point_sets])
    all_labels = np.concatenate([
        [ps.get("label", "")] * len(np.asarray(ps["xs"])) for ps in point_sets
    ])

    annotation = ax.text2D(
        0.02, 0.04, "",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        family="sans-serif",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#aaaaaa", alpha=0.85),
    )
    annotation.set_visible(False)

    def on_motion(event):
        if event.inaxes != ax or event.xdata is None:
            if annotation.get_visible():
                annotation.set_visible(False)
                fig.canvas.draw_idle()
            return

        proj = ax.get_proj()
        x2d, y2d, _ = proj_transform(all_xs, all_ys, all_zs, proj)
        try:
            disp = ax.transData.transform(np.column_stack([x2d, y2d]))
        except Exception:
            return
        ev_disp = np.array([event.x, event.y])
        dists = np.linalg.norm(disp - ev_disp, axis=1)
        idx = int(np.argmin(dists))

        if dists[idx] < 30:
            x, y, z = all_xs[idx], all_ys[idx], all_zs[idx]
            label = all_labels[idx]
            xlabel = ax.get_xlabel() or "x"
            ylabel = ax.get_ylabel() or "y"
            zlabel = ax.get_zlabel() or "z"
            lines = []
            if label:
                lines.append(label)
            lines += [f"{xlabel} = {x:.4g}", f"{ylabel} = {y:.4g}", f"{zlabel} = {z:.4g}"]
            annotation.set_text("\n".join(lines))
            annotation.set_visible(True)
        else:
            annotation.set_visible(False)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_motion)


def lock_camera_azimuth(fig, ax):
    fixed_elev = ax.elev
    initial_azim = ax.azim

    ax.disable_mouse_rotation()

    state = {"x": None, "button": None}

    def on_press(event):
        if event.inaxes == ax:
            state["x"] = event.x
            state["button"] = event.button

    def on_release(event):
        state["button"] = None
        state["x"] = None

    def on_move(event):
        if state["button"] != 1 or state["x"] is None:
            return
        dx = event.x - state["x"]
        state["x"] = event.x
        new_azim = ax.azim - dx
        ax.azim = max(initial_azim - 45, min(initial_azim + 45, new_azim))
        ax.elev = fixed_elev
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
