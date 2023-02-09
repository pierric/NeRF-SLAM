import sys
sys.path.append("build_ngp")

import pyngp as ngp
from tqdm import tqdm

ngp = ngp.Testbed(ngp.TestbedMode.Nerf, 0)
ngp.load_snapshot("snapshot.msgpack")
ngp.dynamic_res = True
ngp.dynamic_res_target_fps = 5
ngp.shall_train = False

ngp.init_window(1980, 800)
for i in tqdm(range(1000)):
    ngp.frame()
    ngp.apply_camera_smoothing(200)
