from tactile_sdk.sharpa.tactile import Touch, TouchSetting

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import WrenchStamped, TransformStamped
from tf2_ros import StaticTransformBroadcaster
from cv_bridge import CvBridge

import numpy as np

import sys
import time
import argparse
import struct
import importlib.util
from types import SimpleNamespace

class TactilePublisher(Node):
    def __init__(self, channel, buffer_size=2, fake_pose=False):
        super().__init__('tactile')
        bz = buffer_size
        self.pub_raw = {ch: self.create_publisher(Image, f'tactile/ch_{ch}/raw', bz) for ch in channel}
        self.pub_f6 = {ch: self.create_publisher(WrenchStamped, f'tactile/ch_{ch}/force6d', bz) for ch in channel}
        self.pub_deform = {ch: self.create_publisher(Image, f'tactile/ch_{ch}/deform', bz) for ch in channel}
        self.pub_pt_cloud = {ch: self.create_publisher(PointCloud2, f'tactile/ch_{ch}/pt_cloud', bz) for ch in channel}
        self.bridge = CvBridge()
        self.pt_cloud_fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        # record start time
        self.hs_touch_t0 = time.time_ns() * 1e-9
        self.ros_t0 = self.get_clock().now()

        if fake_pose:
            tf_broadcaster = StaticTransformBroadcaster(self)
            pos_x = 0.
            for ch in channel:
                msg = TransformStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'map'
                msg.child_frame_id = f'tactile/p_{ch}'
                msg.transform.translation.x = pos_x
                tf_broadcaster.sendTransform(msg)
                pos_x += .05

    def pub(self, ch, ts, raw=None, f6=None, deform=None, pt_cloud=None):
        elapsed_time = ts - self.hs_touch_t0
        current_time = self.ros_t0 + rclpy.time.Duration(seconds=elapsed_time)
        header = Header()
        header.stamp = current_time.to_msg()
        header.frame_id = f'tactile/p_{ch}'
        if raw is not None:
            msg = self.bridge.cv2_to_imgmsg(raw, encoding='mono8', header=header)
            self.pub_raw[ch].publish(msg)
        if f6 is not None:
            msg = WrenchStamped(header=header)
            f6 = f6.flatten().astype(float)
            msg.wrench.force.x = f6[0]
            msg.wrench.force.y = f6[1]
            msg.wrench.force.z = f6[2]
            msg.wrench.torque.x = f6[3]
            msg.wrench.torque.y = f6[4]
            msg.wrench.torque.z = f6[5]
            self.pub_f6[ch].publish(msg)
        if deform is not None:
            msg = self.bridge.cv2_to_imgmsg(deform, encoding='64FC1', header=header)
            self.pub_deform[ch].publish(msg)
        if pt_cloud is not None:
            msg = PointCloud2(
                height=1,
                width=pt_cloud.shape[0],
                fields=self.pt_cloud_fields,
                is_bigendian=False,
                point_step=16,
                row_step=16 * pt_cloud.shape[0],
                data=b"".join(
                    [struct.pack("ffff", *pt) for pt in pt_cloud]
                ),
                is_dense=True,
                header=header,
            )
            self.pub_pt_cloud[ch].publish(msg)

def main():
    parser = argparse.ArgumentParser(description='tactile publisher')
    parser.add_argument('--cfg', type=str, help='load config', default='')
    args = parser.parse_args()

    cfg = dict(
        HOST_IP='0.0.0.0',
        HOST_PORT=50001,
        CHANNEL=range(10),
        MODEL_PATH={},
        ALLOWED_PACK_LOSS=0,
        NUM_WORKER=2,
        BATCH_SIZE=1,
        CALIB_ZERO_FRAMES=0,
        FAKE_POSE=False,
        MESH_PATH='',
        PT_CLOUD_WIDTH=160,
        PT_CLOUD_HEIGHT=120,
        DEFORM_THRESH=0.05,
        DEFORM_CLIP=(0, 3),
    )

    if args.cfg:
        spec = importlib.util.spec_from_file_location('config', args.cfg)
        cfg_ext = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_ext)
        for key in cfg.keys():
            if hasattr(cfg_ext, key):
                cfg[key] = getattr(cfg_ext, key)
    cfg = SimpleNamespace(**cfg)
    print(cfg)

    rclpy.init(args=sys.argv)
    node = TactilePublisher(cfg.CHANNEL, fake_pose=cfg.FAKE_POSE)

    # use initial frames to zero out force and deform offset
    calib_buffer = {ch: ([], []) for ch in cfg.CHANNEL}
    calib_result = {ch: None for ch in cfg.CHANNEL}

    def callback(frame):
        ch = frame['channel']
        ts = frame['ts']
        img, f6, deform = None, None, None
        if frame['content'].get("RAW") is not None: img = frame['content']['RAW'].squeeze()
        if frame['content'].get("DEFORM") is not None: deform = frame['content']['DEFORM'].squeeze()
        if frame['content'].get('F6') is not None: f6 = frame['content']['F6'].squeeze()
        if len(calib_buffer[ch][0]) == cfg.CALIB_ZERO_FRAMES and calib_result[ch] is None:
            calib_result[ch] = (np.array(calib_buffer[ch][0]).mean(axis=0),
                                   np.array(calib_buffer[ch][1]).mean(axis=0))
        if len(calib_buffer[ch][0]) < cfg.CALIB_ZERO_FRAMES:
            calib_buffer[ch][0].append(f6.ravel())
            calib_buffer[ch][1].append(deform)
            return
        if cfg.CALIB_ZERO_FRAMES > 0:
            f6 -= calib_result[ch][0]
            deform = deform - calib_result[ch][1]
        deform = np.clip(deform, cfg.DEFORM_CLIP[0], cfg.DEFORM_CLIP[1])
        node.pub(ch, ts, img, f6, deform)

    setting = TouchSetting(
        model_path=cfg.MODEL_PATH,
        allowed_pack_loss=cfg.ALLOWED_PACK_LOSS,
        num_worker=cfg.NUM_WORKER,
        batch_size=cfg.BATCH_SIZE,
    )
    touch = Touch(
        cfg.HOST_IP,
        cfg.HOST_PORT,
        cfg.CHANNEL,
        callback=callback,
        setting=setting,
    )
    if not touch.start():
        raise RuntimeError('HsTouch start failed')
    rclpy.spin(node)
    touch.stop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
