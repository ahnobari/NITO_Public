import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib import markers
from matplotlib.path import Path


def align_marker(marker, halign="center", valign="middle"):

    if isinstance(halign, (str)):
        halign = {
            "right": -1.0,
            "middle": 0.0,
            "center": 0.0,
            "left": 1.0,
        }[halign]

    if isinstance(valign, (str)):
        valign = {
            "top": -1.0,
            "middle": 0.0,
            "center": 0.0,
            "bottom": 1.0,
        }[valign]

    bm = markers.MarkerStyle(marker)

    m_arr = bm.get_path().transformed(bm.get_transform()).vertices

    m_arr[:, 0] += halign / 2
    m_arr[:, 1] += valign / 2

    return Path(m_arr, bm.get_path().codes)


def plot_problem_2D(
    nodes,
    elements,
    c,
    f,
    ax=None,
    face_color="grey",
    edge_color="black",
    x_color="tomato",
    y_color="royalblue",
    f_color="#8e0000",
    **kwargs,
):

    if ax is None:
        ax = plt.gca()

    y = nodes[:, 0]
    z = nodes[:, 1]

    def quatplot(y, z, quatrangles, ax=None, **kwargs):

        if not ax:
            ax = plt.gca()
        yz = np.c_[y, z]
        verts = yz[quatrangles]
        pc = matplotlib.collections.PolyCollection(verts, **kwargs)
        ax.add_collection(pc)
        ax.autoscale()

    ax.set_aspect("equal")

    elements_ = []
    for e in elements:
        if len(e) == 4:
            elements_.append([e[0], e[1], e[2], e[3]])
        else:
            elements_.append([e[0], e[1], e[2], e[2]])

    quatplot(y, z, np.asarray(elements_), ax=ax, color=edge_color, facecolor=face_color)

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")

    x_bc = c[:, 0] != 0
    y_bc = c[:, 1] != 0

    ax.scatter(
        nodes[x_bc][:, 0],
        nodes[x_bc][:, 1],
        marker=align_marker(">", "right", "middle"),
        s=500,
        color=x_color,
        alpha=0.7,
    )
    ax.scatter(
        nodes[y_bc][:, 0],
        nodes[y_bc][:, 1],
        marker=align_marker("^", "middle", "top"),
        s=500,
        color=y_color,
        alpha=0.7,
    )

    force_nodes = (f != 0).sum(1) > 0

    ax.quiver(
        nodes[force_nodes, 0],
        nodes[force_nodes, 1],
        f[force_nodes, 0] / np.abs(f).max(),
        f[force_nodes, 1] / np.abs(f).max(),
        color=f_color,
        scale=15,
        width=0.005,
    )

    return ax


def plot_mesh_2D(
    nodes, elements, ax=None, face_color="grey", edge_color="black", **kwargs
):

    if ax is None:
        ax = plt.gca()

    y = nodes[:, 0]
    z = nodes[:, 1]

    def quatplot(y, z, quatrangles, ax=None, **kwargs):

        if not ax:
            ax = plt.gca()
        yz = np.c_[y, z]
        verts = yz[quatrangles]
        pc = matplotlib.collections.PolyCollection(verts, **kwargs)
        ax.add_collection(pc)
        ax.autoscale()

    ax.set_aspect("equal")

    elements_ = []
    for e in elements:
        if len(e) == 4:
            elements_.append([e[0], e[1], e[2], e[3]])
        else:
            elements_.append([e[0], e[1], e[2], e[2]])

    quatplot(y, z, np.asarray(elements_), ax=ax, color=edge_color, facecolor=face_color)

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")

    return ax


def plot_field_2D(
    node_positions, elements, field, power_scaling=0.2, plotter=None, cmap="jet"
):
    if plotter is None:
        plotter = pv.Plotter()

    elements_ = []
    for e in elements:
        if len(e) == 4:
            elements_.append([4, e[0], e[1], e[2], e[3]])
        else:
            elements_.append([4, e[0], e[1], e[2], e[2]])

    mesh = pv.PolyData(np.pad(node_positions, [[0, 0], [0, 1]]), elements_)
    mesh[f"Field^{power_scaling}"] = field**power_scaling

    plotter.add_mesh(mesh, scalars=f"Field^{power_scaling}", cmap=cmap)
    plotter.view_xy()

    return plotter


def plot_field_3D(
    node_positions,
    element_centroids,
    field,
    power_scaling=0.2,
    plotter=None,
    cmap="jet",
    opacity=0.7,
):

    if plotter is None:
        plotter = pv.Plotter()

    mesh = pv.PolyData(element_centroids)
    mesh[f"Field^{power_scaling}"] = field**power_scaling

    plotter.add_mesh(mesh, scalars=f"Field^{power_scaling}", cmap=cmap, opacity=opacity)

    return plotter


def plot_mesh_3D(
    node_positions,
    elements,
    plotter=None,
    face_color="grey",
    edge_color="black",
    **kwargs,
):
    faces = []

    for e in elements:
        if len(e) == 4:
            faces.append([4, e[0], e[1], e[2], e[2]])
            faces.append([4, e[0], e[1], e[3], e[3]])
            faces.append([4, e[0], e[2], e[3], e[3]])
            faces.append([4, e[1], e[2], e[3], e[3]])
        else:
            faces.append([4, e[0], e[1], e[2], e[3]])
            faces.append([4, e[4], e[5], e[6], e[7]])
            faces.append([4, e[0], e[1], e[5], e[4]])
            faces.append([4, e[1], e[2], e[6], e[5]])
            faces.append([4, e[2], e[3], e[7], e[6]])
            faces.append([4, e[3], e[0], e[4], e[7]])

    faces = np.hstack(faces)

    pv_mesh = pv.PolyData(node_positions, faces)

    if plotter is None:
        plotter = pv.Plotter()

    plotter.add_mesh(
        pv_mesh, show_edges=True, opacity=1.0, color=face_color, edge_color=edge_color
    )
    plotter.show_grid()

    return plotter


def plot_problem_3D(
    node_positions,
    elements,
    c,
    f,
    plotter=None,
    face_color="grey",
    edge_color="black",
    x_color="tomato",
    y_color="royalblue",
    z_color="springgreen",
    f_color="darkorange",
    **kwargs,
):

    # find mean element size
    l_size = np.mean(
        np.linalg.norm(
            node_positions[[e[0] for e in elements]]
            - node_positions[[e[1] for e in elements]],
            axis=-1,
        )
    )

    faces = []
    for e in elements:
        if len(e) == 4:
            faces.append([4, e[0], e[1], e[2], e[2]])
            faces.append([4, e[0], e[1], e[3], e[3]])
            faces.append([4, e[0], e[2], e[3], e[3]])
            faces.append([4, e[1], e[2], e[3], e[3]])
        else:
            faces.append([4, e[0], e[1], e[2], e[3]])
            faces.append([4, e[4], e[5], e[6], e[7]])
            faces.append([4, e[0], e[1], e[5], e[4]])
            faces.append([4, e[1], e[2], e[6], e[5]])
            faces.append([4, e[2], e[3], e[7], e[6]])
            faces.append([4, e[3], e[0], e[4], e[7]])

    faces = np.hstack(faces)

    pv_mesh = pv.PolyData(node_positions, faces)

    # plot the mesh
    if plotter is None:
        plotter = pv.Plotter()
    plotter.add_mesh(
        pv_mesh, show_edges=True, opacity=1.0, color=face_color, edge_color=edge_color
    )

    nodes = []
    faces = []
    for vert in node_positions[c[:, 0] > 0]:
        points = np.zeros((6, 3))
        points[0] = vert + np.array([0, -l_size / 4, 0])
        points[1] = vert + np.array([l_size / 2, -l_size / 4, l_size / 4])
        points[2] = vert + np.array([l_size / 2, -l_size / 4, -l_size / 4])
        points[3] = points[0] + np.array([0, l_size / 2, 0])
        points[4] = points[1] + np.array([0, l_size / 2, 0])
        points[5] = points[2] + np.array([0, l_size / 2, 0])
        nodes += points.tolist()
        faces.append([len(nodes) - 3, len(nodes) - 2, len(nodes) - 1])
        faces.append([len(nodes) - 6, len(nodes) - 5, len(nodes) - 4])
        faces.append([len(nodes) - 3, len(nodes) - 6, len(nodes) - 5])
        faces.append([len(nodes) - 3, len(nodes) - 6, len(nodes) - 2])
        faces.append([len(nodes) - 2, len(nodes) - 5, len(nodes) - 4])
        faces.append([len(nodes) - 2, len(nodes) - 5, len(nodes) - 1])
        faces.append([len(nodes) - 1, len(nodes) - 4, len(nodes) - 3])
        faces.append([len(nodes) - 1, len(nodes) - 4, len(nodes) - 6])

    if len(nodes) > 0:
        x_mesh = pv.make_tri_mesh(np.array(nodes), np.array(faces))
        plotter.add_mesh(x_mesh, color=x_color)

    nodes = []
    faces = []
    for vert in node_positions[c[:, 1] > 0]:
        points = np.zeros((6, 3))
        points[0] = vert + np.array([-l_size / 4, 0, 0])
        points[1] = vert + np.array([0, l_size / 4, l_size / 2])
        points[2] = vert + np.array([0, -l_size / 4, l_size / 2])
        points[3] = points[0] + np.array([l_size / 2, 0, 0])
        points[4] = points[1] + np.array([l_size / 2, 0, 0])
        points[5] = points[2] + np.array([l_size / 2, 0, 0])
        nodes += points.tolist()
        faces.append([len(nodes) - 3, len(nodes) - 2, len(nodes) - 1])
        faces.append([len(nodes) - 6, len(nodes) - 5, len(nodes) - 4])
        faces.append([len(nodes) - 3, len(nodes) - 6, len(nodes) - 5])
        faces.append([len(nodes) - 3, len(nodes) - 6, len(nodes) - 2])
        faces.append([len(nodes) - 2, len(nodes) - 5, len(nodes) - 4])
        faces.append([len(nodes) - 2, len(nodes) - 5, len(nodes) - 1])
        faces.append([len(nodes) - 1, len(nodes) - 4, len(nodes) - 3])
        faces.append([len(nodes) - 1, len(nodes) - 4, len(nodes) - 6])

    if len(nodes) > 0:
        y_mesh = pv.make_tri_mesh(np.array(nodes), np.array(faces))
        plotter.add_mesh(y_mesh, color=y_color)

    nodes = []
    faces = []
    for vert in node_positions[c[:, 2] > 0]:
        points = np.zeros((6, 3))
        points[0] = vert + np.array([-l_size / 4, 0, 0])
        points[1] = vert + np.array([0.0, l_size / 2, l_size / 4])
        points[2] = vert + np.array([0.0, l_size / 2, -l_size / 4])
        points[3] = points[0] + np.array([l_size / 2, 0, 0])
        points[4] = points[1] + np.array([l_size / 2, 0, 0])
        points[5] = points[2] + np.array([l_size / 2, 0, 0])
        nodes += points.tolist()
        faces.append([len(nodes) - 3, len(nodes) - 2, len(nodes) - 1])
        faces.append([len(nodes) - 6, len(nodes) - 5, len(nodes) - 4])
        faces.append([len(nodes) - 3, len(nodes) - 6, len(nodes) - 5])
        faces.append([len(nodes) - 3, len(nodes) - 6, len(nodes) - 2])
        faces.append([len(nodes) - 2, len(nodes) - 5, len(nodes) - 4])
        faces.append([len(nodes) - 2, len(nodes) - 5, len(nodes) - 1])
        faces.append([len(nodes) - 1, len(nodes) - 4, len(nodes) - 3])
        faces.append([len(nodes) - 1, len(nodes) - 4, len(nodes) - 6])

    if len(nodes) > 0:
        z_mesh = pv.make_tri_mesh(np.array(nodes), np.array(faces))
        plotter.add_mesh(z_mesh, color=z_color)

    for idx in np.where(np.abs(f).sum(axis=1) > 0)[0]:
        vert = node_positions[idx]
        f_d = f[idx]
        plotter.add_mesh(pv.Arrow(vert, f_d, scale=l_size * 10), color=f_color)

    plotter.show_grid()

    return plotter


def plot_problem(node_positions, elements, c, f, plotter=None, **kwargs):
    if node_positions.shape[1] == 2:
        return plot_problem_2D(node_positions, elements, c, f, ax=plotter, **kwargs)
    elif node_positions.shape[1] == 3:
        return plot_problem_3D(
            node_positions, elements, c, f, plotter=plotter, **kwargs
        )
    else:
        raise Exception("Only 2D and 3D meshes are supported")


def plot_mesh(node_positions, elements, plotter=None, **kwargs):
    if node_positions.shape[1] == 2:
        return plot_mesh_2D(node_positions, elements, ax=plotter, **kwargs)
    elif node_positions.shape[1] == 3:
        return plot_mesh_3D(node_positions, elements, plotter=plotter, **kwargs)
    else:
        raise Exception("Only 2D and 3D meshes are supported")


def plot_field(
    node_positions, element_centroids, elements, field, plotter=None, **kwargs
):
    if node_positions.shape[1] == 2:
        return plot_field_2D(node_positions, elements, field, plotter=plotter, **kwargs)
    elif node_positions.shape[1] == 3:
        return plot_field_3D(
            node_positions, element_centroids, field, plotter=plotter, **kwargs
        )
    else:
        raise Exception("Only 2D and 3D meshes are supported")
