import sys
sys.path.append("build_ngp")
import argparse

import numpy as np
import pyngp as ngp
from tqdm import tqdm
from plyfile import PlyData, PlyElement


def export(testbed):
    mc = testbed.compute_marching_cubes_mesh()

    vertex = np.array(list(zip(*mc["V"].T)), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_color = np.array(list(zip(*((mc["C"] * 255).T))), dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    n = len(vertex)
    assert len(vertex_color) == n

    vertex_all = np.empty(n, vertex.dtype.descr + vertex_color.dtype.descr)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    for prop in vertex_color.dtype.names:
        vertex_all[prop] = vertex_color[prop]

    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=False)
    ply.write('nerf_pc.ply')


parser = argparse.ArgumentParser(prog = 'replay')
parser.add_argument('snapshot')
parser.add_argument("--gui", action="store_true")
parser.add_argument("--export", action="store_true")
args = parser.parse_args()

ngp = ngp.Testbed(ngp.TestbedMode.Nerf, 0)
ngp.load_snapshot(args.snapshot)
ngp.dynamic_res = True
ngp.dynamic_res_target_fps = 5
ngp.shall_train = False

if args.gui:
    ngp.init_window(1980, 800)
    for i in tqdm(range(1000)):
        ngp.frame()
        ngp.apply_camera_smoothing(200)

if args.export:
    export(ngp)
