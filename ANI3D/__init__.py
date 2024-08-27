'''
Visualization tools for 3D
'''

import pyvista as pv
import numpy as np
import mcubes
from .geometry import generate_structured_mesh, get_pv
import moviepy.editor as mp
import os
import uuid
from tqdm import trange
from scipy.spatial import KDTree
import copy

def animate_cross_voxelized(top, shape, bc=None, load=None, n_frames = 50, fps=25, f_name='top_anim.mp4', grid=False, face_color='white', edge_color='#87CEFA', highlight_color='royalblue', meat_color='indigo'):
    elements, nodes = generate_structured_mesh(shape,shape)

    f_name_t = uuid.uuid4().hex

    os.makedirs(f_name_t, exist_ok=True)

    max_x = nodes[elements[top[:,0]]].max(1)[:,0].max()
    min_x = nodes[elements[top[:,0]]].min(1)[:,0].min()

    if bc is not None:
        x_points = bc[bc[:,3]==1,0:3]*shape.max()
        y_points = bc[bc[:,4]==1,0:3]*shape.max()
        z_points = bc[bc[:,5]==1,0:3]*shape.max()

        x_mesh = pv.PolyData(x_points)
        y_mesh = pv.PolyData(y_points)
        z_mesh = pv.PolyData(z_points)
    
    if load is not None:
        load_points = load[:,0:3]*shape.max()


    range_of_clip = np.linspace(max_x-0.001,min_x,n_frames)
    mesh = get_pv(nodes, elements[top[:,0]])

    def get_frame(i):
        plotter = pv.Plotter(off_screen=True,lighting='three lights')
        plotter.clear()
        plotter.enable_anti_aliasing('msaa')
        c_mesh = mesh.clip(normal=(1,0,0),origin=(range_of_clip[i],0,0),invert=True)
        c_face = c_mesh.clip(normal=(1,0,0),origin=(range_of_clip[i]-0.001,0,0),invert=False)
        c_edges = c_face.extract_feature_edges(0,
                                               boundary_edges=True,
                                               feature_edges=False,
                                               manifold_edges=False,
                                               non_manifold_edges=False)

        plotter.add_mesh(mesh, show_edges=False, edge_color='royalblue', color='white', opacity=0.0)
        plotter.add_mesh(c_mesh, show_edges=True, edge_color=edge_color, color=face_color, opacity=1.0)
        plotter.add_mesh(c_edges, color=highlight_color, line_width=5)


        if bc is not None:
            plotter.add_mesh(x_mesh, color='red', point_size=20, render_points_as_spheres=True, opacity=0.7)
            plotter.add_mesh(y_mesh, color='green', point_size=20, render_points_as_spheres=True, opacity=0.7)
            plotter.add_mesh(z_mesh, color='blue', point_size=20, render_points_as_spheres=True, opacity=0.7)

        if load is not None:
            for i in range(load_points.shape[0]):
                plotter.add_mesh(pv.Arrow(load_points[i], load[i,3:6].astype(float), scale=6, shaft_radius=0.02, tip_radius=0.06, tip_length=0.2), color='indigo')
        
        if grid:
            plotter.show_grid()

        return plotter

    for i in trange(range_of_clip.shape[0]):
        p = get_frame(i)
        p.screenshot(f_name_t + f'/{i}.png')
        p.close()
    
    # make video
    clip = mp.ImageSequenceClip([f_name_t + f'/{i}.png' for i in range(range_of_clip.shape[0])], fps=fps)
    clip = mp.concatenate_videoclips([clip, clip.fx(mp.vfx.time_mirror)])
    clip.write_videofile(f_name)

    # remove temp files
    for i in range(range_of_clip.shape[0]):
        os.remove(f_name_t + f'/{i}.png')
    os.removedirs(f_name_t)

def animate_cross_smooth(top, shape, bc=None, load=None, n_frames = 50, fps=25, f_name='top_anim.mp4', grid=False, face_color='lightgrey', edge_color=None, highlight_color='black', meat_color='indigo'):
    f_name_t = uuid.uuid4().hex

    os.makedirs(f_name_t, exist_ok=True)

    vertices, triangles = mcubes.marching_cubes(np.pad(top.reshape(shape),[[1,1]]*3), 0.5)
    vertices = vertices - 0.5
    mesh = pv.make_tri_mesh(vertices, triangles)
    mesh = mesh.smooth(n_iter=1000, relaxation_factor=0.01)

    max_x = mesh.points[:,0].max()
    min_x = mesh.points[:,0].min()

    range_of_clip = np.linspace(max_x-0.001,min_x+0.001,n_frames)


    if bc is not None or load is not None:
        tree = KDTree(mesh.points)
    if bc is not None:
        x_points_ = bc[bc[:,3]==1,0:3]*shape.max()
        y_points_ = bc[bc[:,4]==1,0:3]*shape.max()
        z_points_ = bc[bc[:,5]==1,0:3]*shape.max()
    

    
        x_points = mesh.points[tree.query(x_points_)[1]]
        y_points = mesh.points[tree.query(y_points_)[1]]
        z_points = mesh.points[tree.query(z_points_)[1]]
        
        del_x = np.linalg.norm(x_points-x_points_, axis=1)
        del_y = np.linalg.norm(y_points-y_points_, axis=1)
        del_z = np.linalg.norm(z_points-z_points_, axis=1)

        x_points[del_x>2] = x_points_[del_x>2]
        y_points[del_y>2] = y_points_[del_y>2]
        z_points[del_z>2] = z_points_[del_z>2]

        x_mesh = pv.PolyData(x_points)
        y_mesh = pv.PolyData(y_points)
        z_mesh = pv.PolyData(z_points)
    
    if load is not None:
        load_points = load[:,0:3]*shape.max()
        load_points = mesh.points[tree.query(load_points)[1]]

    if edge_color is None:
        show_edges = False
    else:
        show_edges = True

    def get_frame(i):
        c_mesh = mesh.clip(normal=(1,0,0),origin=(range_of_clip[i],0,0),invert=True)
        c_face = c_mesh.clip(normal=(1,0,0),origin=(range_of_clip[i]-0.001,0,0),invert=False)
        c_edges = c_face.extract_feature_edges(0,
                                               boundary_edges=True,
                                               feature_edges=False,
                                               manifold_edges=False,
                                               non_manifold_edges=False)

        #make a plane surface of the cut location
        plane_mesh = pv.Plane(center=(range_of_clip[i],shape[1]/2,shape[2]/2), direction=(1,0,0), i_size=shape.max()*2, j_size=shape.max()*2).triangulate().subdivide(3)
        cut_surface = plane_mesh.clip_surface(mesh)

        
        plotter = pv.Plotter(off_screen=True,lighting='three lights')
        plotter.enable_anti_aliasing('msaa')
        # plotter.enable_eye_dome_lighting()
        plotter.add_mesh(mesh, show_edges=False, edge_color='royalblue', color='#000000', opacity=0.0)
        plotter.add_mesh(c_mesh, show_edges=show_edges, edge_color=edge_color, color=face_color, opacity=1.0, smooth_shading=True)
        if cut_surface.n_points > 0:
            plotter.add_mesh(cut_surface, color=meat_color)
        if c_edges.n_points > 0:
            plotter.add_mesh(c_edges, color=highlight_color, line_width=3)

        if bc is not None:
            plotter.add_mesh(x_mesh, color='red', point_size=20, render_points_as_spheres=True, opacity=0.7)
            plotter.add_mesh(y_mesh, color='green', point_size=20, render_points_as_spheres=True, opacity=0.7)
            plotter.add_mesh(z_mesh, color='blue', point_size=20, render_points_as_spheres=True, opacity=0.7)

        if load is not None:
            for i in range(load_points.shape[0]):
                plotter.add_mesh(pv.Arrow(load_points[i], load[i,3:6].astype(float), scale=6, shaft_radius=0.02, tip_radius=0.06, tip_length=0.2), color='indigo')
        
        if grid:
            plotter.show_grid()

        return plotter

    for i in trange(range_of_clip.shape[0]):
        plotter = get_frame(i)
        plotter.screenshot(f_name_t + f'/{i}.png')
    
    plotter.close()

    clip = mp.ImageSequenceClip([f_name_t + f'/{i}.png' for i in range(n_frames)], fps=fps)
    clip = mp.concatenate_videoclips([clip, clip.fx(mp.vfx.time_mirror)])

    clip.write_videofile(f_name)

    for i in range(n_frames):
        os.remove(f_name_t + f'/{i}.png')
    
    os.rmdir(f_name_t)

def animate_rotation_voxelized(top, shape, bc=None, load=None, n_frames = 50, fps=25, f_name='top_anim.mp4', grid=False, face_color='white', edge_color='#87CEFA'):
    elements, nodes = generate_structured_mesh(shape,shape)

    f_name_t = uuid.uuid4().hex

    os.makedirs(f_name_t, exist_ok=True)

    if bc is not None:
        x_points = bc[bc[:,3]==1,0:3]*shape.max()
        y_points = bc[bc[:,4]==1,0:3]*shape.max()
        z_points = bc[bc[:,5]==1,0:3]*shape.max()

        x_mesh = pv.PolyData(x_points)
        y_mesh = pv.PolyData(y_points)
        z_mesh = pv.PolyData(z_points)
    
    if load is not None:
        load_points = load[:,0:3]*shape.max()

    mesh = get_pv(nodes, elements[top[:,0]])

    plotter = pv.Plotter(off_screen=True,lighting='three lights')
    plotter.clear()
    plotter.enable_anti_aliasing('msaa')
    plotter.add_mesh(mesh, show_edges=False, edge_color='royalblue', color='white', opacity=1.0)
    if bc is not None:
        plotter.add_mesh(x_mesh, color='red', point_size=20, render_points_as_spheres=True, opacity=0.7)
        plotter.add_mesh(y_mesh, color='green', point_size=20, render_points_as_spheres=True, opacity=0.7)
        plotter.add_mesh(z_mesh, color='blue', point_size=20, render_points_as_spheres=True, opacity=0.7)

    if load is not None:
        for i in range(load_points.shape[0]):
            plotter.add_mesh(pv.Arrow(load_points[i], load[i,3:6].astype(float), scale=6, shaft_radius=0.02, tip_radius=0.06, tip_length=0.2), color='indigo')
    
    if grid:
        plotter.show_grid()
    plotter.screenshot(f_name_t + f'/temp.png')
    os.remove(f_name_t + f'/temp.png')
    c_p = copy.deepcopy(plotter.camera_position)
    plotter.close()

    def get_frame(i):
        plotter = pv.Plotter(off_screen=True,lighting='three lights')
        plotter.clear()
        plotter.enable_anti_aliasing('msaa')
        plotter.add_mesh(mesh, show_edges=True, edge_color=edge_color, color=face_color, opacity=1.0)
        plotter.camera_position = c_p
        plotter.camera.azimuth += i*360/n_frames

        if bc is not None:
            plotter.add_mesh(x_mesh, color='red', point_size=20, render_points_as_spheres=True, opacity=0.7)
            plotter.add_mesh(y_mesh, color='green', point_size=20, render_points_as_spheres=True, opacity=0.7)
            plotter.add_mesh(z_mesh, color='blue', point_size=20, render_points_as_spheres=True, opacity=0.7)

        if load is not None:
            for i in range(load_points.shape[0]):
                plotter.add_mesh(pv.Arrow(load_points[i], load[i,3:6].astype(float), scale=6, shaft_radius=0.02, tip_radius=0.06, tip_length=0.2), color='indigo')
        
        if grid:
            plotter.show_grid()

        return plotter

    for i in trange(n_frames):
        p = get_frame(i)
        p.screenshot(f_name_t + f'/{i}.png')
        p.close()
    
    # make video
    clip = mp.ImageSequenceClip([f_name_t + f'/{i}.png' for i in range(n_frames)], fps=fps)
    clip.write_videofile(f_name)

    # remove temp files
    for i in range(n_frames):
        os.remove(f_name_t + f'/{i}.png')
    os.removedirs(f_name_t)

def animate_rotation_smooth(top, shape, bc=None, load=None, n_frames = 50, fps=25, f_name='top_anim.mp4', grid=False, face_color='lightgrey', edge_color=None):
    f_name_t = uuid.uuid4().hex

    os.makedirs(f_name_t, exist_ok=True)

    vertices, triangles = mcubes.marching_cubes(np.pad(top.reshape(shape),[[1,1]]*3), 0.5)
    vertices = vertices - 0.5
    mesh = pv.make_tri_mesh(vertices, triangles)
    mesh = mesh.smooth(n_iter=1000, relaxation_factor=0.01)

    if bc is not None or load is not None:
        tree = KDTree(mesh.points)
    if bc is not None:
        x_points_ = bc[bc[:,3]==1,0:3]*shape.max()
        y_points_ = bc[bc[:,4]==1,0:3]*shape.max()
        z_points_ = bc[bc[:,5]==1,0:3]*shape.max()
    
        x_points = mesh.points[tree.query(x_points_)[1]]
        y_points = mesh.points[tree.query(y_points_)[1]]
        z_points = mesh.points[tree.query(z_points_)[1]]
        
        del_x = np.linalg.norm(x_points-x_points_, axis=1)
        del_y = np.linalg.norm(y_points-y_points_, axis=1)
        del_z = np.linalg.norm(z_points-z_points_, axis=1)

        x_points[del_x>2] = x_points_[del_x>2]
        y_points[del_y>2] = y_points_[del_y>2]
        z_points[del_z>2] = z_points_[del_z>2]

        x_mesh = pv.PolyData(x_points)
        y_mesh = pv.PolyData(y_points)
        z_mesh = pv.PolyData(z_points)
    
    if load is not None:
        load_points = load[:,0:3]*shape.max()
        load_points = mesh.points[tree.query(load_points)[1]]

    if edge_color is None:
        show_edges = False
    else:
        show_edges = True

    plotter = pv.Plotter(off_screen=True,lighting='three lights')
    plotter.enable_anti_aliasing('msaa')
    plotter.add_mesh(mesh, show_edges=False, edge_color='royalblue', color='white', opacity=1.0, smooth_shading=True)
    if bc is not None:
        plotter.add_mesh(x_mesh, color='red', point_size=20, render_points_as_spheres=True, opacity=0.7)
        plotter.add_mesh(y_mesh, color='green', point_size=20, render_points_as_spheres=True, opacity=0.7)
        plotter.add_mesh(z_mesh, color='blue', point_size=20, render_points_as_spheres=True, opacity=0.7)
    
    if load is not None:
        for i in range(load_points.shape[0]):
            plotter.add_mesh(pv.Arrow(load_points[i], load[i,3:6].astype(float), scale=6, shaft_radius=0.02, tip_radius=0.06, tip_length=0.2), color='indigo')
    
    if grid:
        plotter.show_grid()
    plotter.screenshot(f_name_t + f'/temp.png')
    os.remove(f_name_t + f'/temp.png')
    c_p = copy.deepcopy(plotter.camera_position)
    plotter.close()

    def get_frame(i):
        plotter = pv.Plotter(off_screen=True,lighting='three lights')
        plotter.enable_anti_aliasing('msaa')
        plotter.add_mesh(mesh, show_edges=show_edges, edge_color=edge_color, color=face_color, opacity=1.0, smooth_shading=True)
        plotter.camera_position = c_p
        plotter.camera.azimuth += i*360/n_frames

        if bc is not None:
            plotter.add_mesh(x_mesh, color='red', point_size=20, render_points_as_spheres=True, opacity=0.7)
            plotter.add_mesh(y_mesh, color='green', point_size=20, render_points_as_spheres=True, opacity=0.7)
            plotter.add_mesh(z_mesh, color='blue', point_size=20, render_points_as_spheres=True, opacity=0.7)

        if load is not None:
            for i in range(load_points.shape[0]):
                plotter.add_mesh(pv.Arrow(load_points[i], load[i,3:6].astype(float), scale=6, shaft_radius=0.02, tip_radius=0.06, tip_length=0.2), color='indigo')
        
        if grid:
            plotter.show_grid()

        return plotter

    for i in trange(n_frames):
        p = get_frame(i)
        p.screenshot(f_name_t + f'/{i}.png')
        p.close()
    
    # make video
    clip = mp.ImageSequenceClip([f_name_t + f'/{i}.png' for i in range(n_frames)], fps=fps)
    clip.write_videofile(f_name)

    # remove temp files
    for i in range(n_frames):
        os.remove(f_name_t + f'/{i}.png')
    os.removedirs(f_name_t)

def animate_top_series(tops, shape, bc=None, load=None, fps=25, f_name='top_anim.mp4', grid=False, face_color='lightgrey', edge_color=None, azimuth=None, elevation=None, roll=None):
    f_name_t = uuid.uuid4().hex

    os.makedirs(f_name_t, exist_ok=True)

    vertices, triangles = mcubes.marching_cubes(np.pad(tops[-1].reshape(shape),[[1,1]]*3), 0.5)
    vertices = vertices - 0.5
    mesh = pv.make_tri_mesh(vertices, triangles)
    mesh = mesh.smooth(n_iter=1000, relaxation_factor=0.01)

    if bc is not None or load is not None:
        tree = KDTree(mesh.points)
    if bc is not None:
        x_points_ = bc[bc[:,3]==1,0:3]*shape.max()
        y_points_ = bc[bc[:,4]==1,0:3]*shape.max()
        z_points_ = bc[bc[:,5]==1,0:3]*shape.max()
    
        x_points = mesh.points[tree.query(x_points_)[1]]
        y_points = mesh.points[tree.query(y_points_)[1]]
        z_points = mesh.points[tree.query(z_points_)[1]]
        
        del_x = np.linalg.norm(x_points-x_points_, axis=1)
        del_y = np.linalg.norm(y_points-y_points_, axis=1)
        del_z = np.linalg.norm(z_points-z_points_, axis=1)

        x_points[del_x>2] = x_points_[del_x>2]
        y_points[del_y>2] = y_points_[del_y>2]
        z_points[del_z>2] = z_points_[del_z>2]

        x_mesh = pv.PolyData(x_points)
        y_mesh = pv.PolyData(y_points)
        z_mesh = pv.PolyData(z_points)
    
    if load is not None:
        load_points = load[:,0:3]*shape.max()
        load_points = mesh.points[tree.query(load_points)[1]]

    if edge_color is None:
        show_edges = False
    else:
        show_edges = True
    
    def get_frame(i):
        vertices, triangles = mcubes.marching_cubes(np.pad(tops[i].reshape(shape),[[1,1]]*3), 0.5)
        vertices = vertices - 0.5
        mesh_ = pv.make_tri_mesh(vertices, triangles)
        if mesh_.n_points > 0:
            mesh_ = mesh_.smooth(n_iter=1000, relaxation_factor=0.01)

        plotter = pv.Plotter(off_screen=True,lighting='three lights')
        plotter.enable_anti_aliasing('msaa')
        plotter.add_mesh(mesh, show_edges=show_edges, edge_color=edge_color, color=face_color, opacity=0.0)
        if mesh_.n_points > 0:
            plotter.add_mesh(mesh_, show_edges=show_edges, edge_color=edge_color, color=face_color, opacity=1.0, smooth_shading=True)
        if bc is not None:
            plotter.add_mesh(x_mesh, color='red', point_size=20, render_points_as_spheres=True, opacity=0.7)
            plotter.add_mesh(y_mesh, color='green', point_size=20, render_points_as_spheres=True, opacity=0.7)
            plotter.add_mesh(z_mesh, color='blue', point_size=20, render_points_as_spheres=True, opacity=0.7)
        
        if load is not None:
            for i in range(load_points.shape[0]):
                plotter.add_mesh(pv.Arrow(load_points[i], load[i,3:6].astype(float), scale=6, shaft_radius=0.02, tip_radius=0.06, tip_length=0.2), color='indigo')
        
        if grid:
            plotter.show_grid()

        return plotter
    
    for i in trange(len(tops)):
        p = get_frame(i)
        p.screenshot(f_name_t + f'/{i}.png')
        cam = copy.deepcopy(p.camera_position)
        p.camera_position = cam
        if azimuth is not None:
            p.camera.azimuth = azimuth
        if elevation is not None:
            p.camera.elevation = elevation
        if roll is not None:
            p.camera.roll = roll
        p.screenshot(f_name_t + f'/{i}.png')
        p.close()

    # make video
    clip = mp.ImageSequenceClip([f_name_t + f'/{i}.png' for i in range(len(tops))], fps=fps)
    clip.write_videofile(f_name)

    # remove temp files
    for i in range(len(tops)):
        os.remove(f_name_t + f'/{i}.png')
    os.removedirs(f_name_t)

def animate_top_series_w_rotation(tops, shape, bc=None, load=None, fps=25, f_name='top_anim.mp4', grid=False, face_color='lightgrey', edge_color=None):
    f_name_t = uuid.uuid4().hex

    os.makedirs(f_name_t, exist_ok=True)

    vertices, triangles = mcubes.marching_cubes(np.pad(tops[-1].reshape(shape),[[1,1]]*3), 0.5)
    vertices = vertices - 0.5
    mesh = pv.make_tri_mesh(vertices, triangles)
    mesh = mesh.smooth(n_iter=1000, relaxation_factor=0.01)

    if bc is not None or load is not None:
        tree = KDTree(mesh.points)
    if bc is not None:
        x_points_ = bc[bc[:,3]==1,0:3]*shape.max()
        y_points_ = bc[bc[:,4]==1,0:3]*shape.max()
        z_points_ = bc[bc[:,5]==1,0:3]*shape.max()
    
        x_points = mesh.points[tree.query(x_points_)[1]]
        y_points = mesh.points[tree.query(y_points_)[1]]
        z_points = mesh.points[tree.query(z_points_)[1]]
        
        del_x = np.linalg.norm(x_points-x_points_, axis=1)
        del_y = np.linalg.norm(y_points-y_points_, axis=1)
        del_z = np.linalg.norm(z_points-z_points_, axis=1)

        x_points[del_x>2] = x_points_[del_x>2]
        y_points[del_y>2] = y_points_[del_y>2]
        z_points[del_z>2] = z_points_[del_z>2]

        x_mesh = pv.PolyData(x_points)
        y_mesh = pv.PolyData(y_points)
        z_mesh = pv.PolyData(z_points)
    
    if load is not None:
        load_points = load[:,0:3]*shape.max()
        load_points = mesh.points[tree.query(load_points)[1]]

    if edge_color is None:
        show_edges = False
    else:
        show_edges = True
    
    plotter = pv.Plotter(off_screen=True,lighting='three lights')
    plotter.enable_anti_aliasing('msaa')
    plotter.add_mesh(mesh, show_edges=False, edge_color='royalblue', color='white', opacity=1.0, smooth_shading=True)
    if bc is not None:
        plotter.add_mesh(x_mesh, color='red', point_size=20, render_points_as_spheres=True, opacity=0.7)
        plotter.add_mesh(y_mesh, color='green', point_size=20, render_points_as_spheres=True, opacity=0.7)
        plotter.add_mesh(z_mesh, color='blue', point_size=20, render_points_as_spheres=True, opacity=0.7)
    
    if load is not None:
        for i in range(load_points.shape[0]):
            plotter.add_mesh(pv.Arrow(load_points[i], load[i,3:6].astype(float), scale=6, shaft_radius=0.02, tip_radius=0.06, tip_length=0.2), color='indigo')
    
    if grid:
        plotter.show_grid()
    
    plotter.screenshot(f_name_t + f'/temp.png')
    os.remove(f_name_t + f'/temp.png')
    c_p = copy.deepcopy(plotter.camera_position)
    plotter.close()

    def get_frame(i):
        vertices, triangles = mcubes.marching_cubes(np.pad(tops[i].reshape(shape),[[1,1]]*3), 0.5)
        vertices = vertices - 0.5
        mesh_ = pv.make_tri_mesh(vertices, triangles)

        if mesh_.n_points > 0:
            mesh_ = mesh_.smooth(n_iter=1000, relaxation_factor=0.01)

        plotter = pv.Plotter(off_screen=True,lighting='three lights')
        plotter.enable_anti_aliasing('msaa')
        if mesh_.n_points > 0:
            plotter.add_mesh(mesh_, show_edges=show_edges, edge_color=edge_color, color=face_color, opacity=1.0, smooth_shading=True)
        plotter.camera_position = c_p
        plotter.camera.azimuth += i*360/len(tops)

        if bc is not None:
            plotter.add_mesh(x_mesh, color='red', point_size=20, render_points_as_spheres=True, opacity=0.7)
            plotter.add_mesh(y_mesh, color='green', point_size=20, render_points_as_spheres=True, opacity=0.7)
            plotter.add_mesh(z_mesh, color='blue', point_size=20, render_points_as_spheres=True, opacity=0.7)
        
        if load is not None:
            for i in range(load_points.shape[0]):
                plotter.add_mesh(pv.Arrow(load_points[i], load[i,3:6].astype(float), scale=6, shaft_radius=0.02, tip_radius=0.06, tip_length=0.2), color='indigo')
        
        if grid:
            plotter.show_grid()

        return plotter
    
    for i in trange(len(tops)):
        p = get_frame(i)
        p.screenshot(f_name_t + f'/{i}.png')
        p.close()
    
    # make video
    clip = mp.ImageSequenceClip([f_name_t + f'/{i}.png' for i in range(len(tops))], fps=fps)
    clip.write_videofile(f_name)

    # remove temp files
    for i in range(len(tops)):
        os.remove(f_name_t + f'/{i}.png')
    os.removedirs(f_name_t)

def smooth_top_plot(top, shape, bc=None, load=None, face_color='lightgrey', edge_color=None):
    vertices, triangles = mcubes.marching_cubes(np.pad(top.reshape(shape),[[1,1]]*3), 0.5)
    vertices = vertices - 0.5
    mesh = pv.make_tri_mesh(vertices, triangles)
    mesh = mesh.smooth(n_iter=1000, relaxation_factor=0.01)

    if bc is not None or load is not None:
        tree = KDTree(mesh.points)
    if bc is not None:
        x_points_ = bc[bc[:,3]==1,0:3]*shape.max()
        y_points_ = bc[bc[:,4]==1,0:3]*shape.max()
        z_points_ = bc[bc[:,5]==1,0:3]*shape.max()
    
        x_points = mesh.points[tree.query(x_points_)[1]]
        y_points = mesh.points[tree.query(y_points_)[1]]
        z_points = mesh.points[tree.query(z_points_)[1]]
        
        del_x = np.linalg.norm(x_points-x_points_, axis=1)
        del_y = np.linalg.norm(y_points-y_points_, axis=1)
        del_z = np.linalg.norm(z_points-z_points_, axis=1)

        x_points[del_x>2] = x_points_[del_x>2]
        y_points[del_y>2] = y_points_[del_y>2]
        z_points[del_z>2] = z_points_[del_z>2]

        x_mesh = pv.PolyData(x_points)
        y_mesh = pv.PolyData(y_points)
        z_mesh = pv.PolyData(z_points)
    
    if load is not None:
        load_points = load[:,0:3]*shape.max()
        load_points = mesh.points[tree.query(load_points)[1]]

    if edge_color is None:
        show_edges = False
    else:
        show_edges = True

    plotter = pv.Plotter(off_screen=True,lighting='three lights')
    plotter.enable_anti_aliasing('msaa')
    plotter.add_mesh(mesh, show_edges=show_edges, edge_color=edge_color, color=face_color, opacity=1.0, smooth_shading=True)
    if bc is not None:
        plotter.add_mesh(x_mesh, color='red', point_size=20, render_points_as_spheres=True, opacity=0.7)
        plotter.add_mesh(y_mesh, color='green', point_size=20, render_points_as_spheres=True, opacity=0.7)
        plotter.add_mesh(z_mesh, color='blue', point_size=20, render_points_as_spheres=True, opacity=0.7)
    
    if load is not None:
        for i in range(load_points.shape[0]):
            plotter.add_mesh(pv.Arrow(load_points[i], load[i,3:6].astype(float), scale=6, shaft_radius=0.02, tip_radius=0.06, tip_length=0.2), color='indigo')
    
    return plotter

def voxelized_top_plot(top, shape, bc=None, load=None, face_color='white', edge_color='#87CEFA'):
    elements, nodes = generate_structured_mesh(shape,shape)

    mesh = get_pv(nodes, elements[top[:,0]])

    if bc is not None:
        x_points = bc[bc[:,3]==1,0:3]*shape.max()
        y_points = bc[bc[:,4]==1,0:3]*shape.max()
        z_points = bc[bc[:,5]==1,0:3]*shape.max()

        x_mesh = pv.PolyData(x_points)
        y_mesh = pv.PolyData(y_points)
        z_mesh = pv.PolyData(z_points)
    
    if load is not None:
        load_points = load[:,0:3]*shape.max()

    plotter = pv.Plotter(off_screen=True,lighting='three lights')
    plotter.enable_anti_aliasing('msaa')
    plotter.add_mesh(mesh, show_edges=True, edge_color=edge_color, color=face_color, opacity=1.0)
    if bc is not None:
        plotter.add_mesh(x_mesh, color='red', point_size=20, render_points_as_spheres=True, opacity=0.7)
        plotter.add_mesh(y_mesh, color='green', point_size=20, render_points_as_spheres=True, opacity=0.7)
        plotter.add_mesh(z_mesh, color='blue', point_size=20, render_points_as_spheres=True, opacity=0.7)

    if load is not None:
        for i in range(load_points.shape[0]):
            plotter.add_mesh(pv.Arrow(load_points[i], load[i,3:6].astype(float), scale=6, shaft_radius=0.02, tip_radius=0.06, tip_length=0.2), color='indigo')
    
    return plotter