import os
import sys
import threading
import time
import pytest
import subprocess

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
sys.path.append(os.path.join(current_path, '../py'))
from app.zmq_wrapper import ZmqWrapper
import zmq
import app.sharpa_pb2 as pb

process = subprocess.Popen([
    "tactile-pub-zmq",
    "--cfg",
    "config/zmq_cfg.py",
])

def panic(msg):
    process.kill()
    pytest.fail(msg)

def test_tactile_height_width():
    # 只创建订阅者，等待外部发布者发送数据
    sub = ZmqWrapper(zmq.SUB, "tcp://127.0.0.1:6666")
    event = threading.Event()
    result = {}

    def parse_tactile(data):
        tactile_msg = pb.Tactile()
        tactile_msg.ParseFromString(data)
        print('force', tactile_msg.force6d.force.x, tactile_msg.force6d.force.y, tactile_msg.force6d.force.z)
        print('deform height*width', tactile_msg.deform.height, tactile_msg.deform.width)
        result['height'] = tactile_msg.deform.height
        result['width'] = tactile_msg.deform.width
        event.set()

    sub.start_receiver_thread(parse_tactile)
    
    # 等待外部发布者发送数据
    print("Waiting for tactile data from external publisher...")

    # 使用循环等待，每0.1秒检查一次，总共等待10秒
    timeout = 30  # 30秒超时
    start_time = time.time()
    while not event.is_set():
        if time.time() - start_time > timeout:
            panic(f"No tactile data received within {timeout} seconds")
        time.sleep(0.1)
    
    # 使用pytest断言验证数据
    if not result['width'] == 320:
        panic(f"Expected width=320, got {result['width']}")

    sub.close()

    process.kill()
