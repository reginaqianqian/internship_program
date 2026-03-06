import os

# constants

CHANNEL = range(5)
"""
0-4: right thumb -> little
5-9: left thumb -> little
"""

HOST_IP = '192.168.10.240'
"""
host ip
"""

HOST_PORT = 50001
"""
host port
"""

MODEL_PATH = {
    'models/EF_lighter128_0905_290d5.onnx': [
        0, 1, 2, 3, 4,
    ],
    # 'models/model_epoch_10__HA3_390_366_finetune_all.onnx': [
    #     4,
    # ],
    # add new model
    # 'your_model.engine': [
    #     xxxxx, xxxxx,
    # ]
}
"""
path for NN models used for inference
"""

FPS = 180
"""
frames per second
"""

FAKE_POSE = True
"""
if fake pose is used for each finger for rviz visualization
"""

ALLOWED_PACK_LOSS = 0
"""
allow packet loss for each frame, see HsTouch.__init__()
"""

CALIB_ZERO_FRAMES = 50
"""
number of frames used to remove initial offset of force and deform value
"""

NUM_WORKER = 2
"""
number of threads use for inference, see HsTouch.__init__()
"""

BATCH_SIZE = min(len(CHANNEL), 5)
"""
number of frames inside a inference batch
"""

MESH_PATH = 'surface.npy'
"""
mesh data used for generating fingertip point cloud
"""

PT_CLOUD_WIDTH = 120
"""
point cloud width
"""

PT_CLOUD_HEIGHT = 160
"""
point cloud height
"""

DEFORM_THRESH = -100
"""
only points with deform larger than this value will be published
"""

DEFORM_CLIP = (0, 3)
"""
deform will be cliped into this range [mm]
"""

# path relative to absolute
_cwd = os.path.dirname(os.path.abspath(__file__))
MESH_PATH = f'{_cwd}/{MESH_PATH}'
MODEL_PATH = {f'{_cwd}/{path}': ports for path, ports in MODEL_PATH.items()}
