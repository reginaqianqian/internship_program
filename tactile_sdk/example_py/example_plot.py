from tactile_sdk.sharpa.tactile import Touch, TouchSetting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import time
import threading
from collections import deque
import numpy as np

CHANNEL = 0
FPS = 180
DURATION = 2
HOST_PORT = 50001
HOST_IP = '192.168.10.240'

class SafeQueue:
    def __init__(self, max_len=1):
        self.q = deque(maxlen=max_len)
        self.cond = threading.Condition()

    def enqueue(self, item):
        with self.cond:
            self.q.append(item)
            self.cond.notify()

    def dequeue(self, timeout=None):
        with self.cond:
            is_obtained = self.cond.wait_for(lambda: self.q, timeout)
        return self.q.popleft() if is_obtained else None

q = SafeQueue(max_len=5)

def callback(frame):
    if 'F6' not in frame['content']: return
    if frame['channel'] != CHANNEL: return
    ts = frame['ts']
    f6 = frame['content']['F6'].squeeze()
    q.enqueue((ts, f6))

setting = TouchSetting(
    model_path={"config/models/DEV2pin_400_de7d860c.onnx": [CHANNEL]},
    infer_from_device=False,
    fps = FPS
)
touch = Touch(HOST_IP, HOST_PORT, [CHANNEL], board_ip= ["192.168.10.20"], callback=callback, setting=setting)
t_start = time.time_ns() * 1e-9
if touch.start()!=0:
    raise RuntimeError('HsTouch start failed')
touch.calib_zero()
# plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_ylim(-0.01, 0.01)
ax.set_xlim(0, DURATION)
ax.set_xlabel('Time')
ax.set_title('Real-time Data Stream')

fx, fy, fz, ts = [], [], [], []

line_x, = ax.plot([], [], 'r-', label='fx')
line_y, = ax.plot([], [], 'g-', label='fy')
line_z, = ax.plot([], [], 'b-', label='fz')


current_key = None 

def on_key_press(event):
    global current_key
    current_key = event.key

fig.canvas.mpl_connect('key_press_event', on_key_press)

def init():
    line_x.set_data([], [])
    return line_x, line_y, line_z

def animate(i):
    global current_key
    global touch
    if current_key == 't':
        touch.calib_zero()
        current_key = None

    data = q.dequeue(timeout=1./FPS)
    if data is None: return line_x, line_y, line_z
    frame_ts, f6 = data

    t = frame_ts - t_start
    ts.append(t)
    fx.append(f6[0])
    fy.append(f6[1])
    fz.append(f6[2])

    y_min = min(min(fx), min(fy), min(fz), -0.01)
    y_max = max(max(fx), max(fy), max(fz), 0.01)
    ax.set_ylim(y_min, y_max)
    
    while ts[-1] - ts[0] > DURATION:
        fx.pop(0)
        fy.pop(0)
        fz.pop(0)
        ts.pop(0)
    
    # update the plot
    line_x.set_data(ts, fx)
    line_y.set_data(ts, fy)
    line_z.set_data(ts, fz)
    if t > DURATION:
        ax.set_xlim(t-DURATION, t)
    ax.figure.canvas.draw()
    
    return line_x, line_y, line_z

ani = FuncAnimation(fig, animate, init_func=init,
                   interval=1000./FPS, blit=True, cache_frame_data=False)
plt.show()
touch.stop()
