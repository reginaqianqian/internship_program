from tactile_sdk.sharpa.tactile import Touch, TouchSetting
import cv2
import numpy as np
import matplotlib.pyplot as plt

import time
import json

CHANNEL = range(5)
HOST_PORT = 50001
HOST_IP = '192.168.10.240'

latency = []

for ch in CHANNEL:
    cv2.namedWindow(f'{ch}', cv2.WINDOW_NORMAL)
    cv2.namedWindow(f'{ch}_deform', cv2.WINDOW_NORMAL)

setting = TouchSetting(
    model_path={"config/models/DEV2pin_400_de7d860c.onnx": CHANNEL},
    infer_from_device=False,
    fps=180,
)
touch = Touch(HOST_IP, HOST_PORT, CHANNEL, board_ip= ["192.168.10.20"], setting=setting)
if touch.start()!=0:
    raise RuntimeError('HsTouch start failed')

key = 0
while key != 27:
    key = cv2.waitKey(1)
    if key == ord('t'):
        touch.calib_zero()
    for ch in CHANNEL:
        ret = touch.fetch(ch, timeout=0)
        if ret is None: continue  # None means timeout
        if ret["content"]["RAW"] is None:
            print("RAW is None") 
            continue
        ts= ret["ts"]
        img=ret["content"]["RAW"]
        latency.append(time.time_ns() * 1e-9 - ts)
        cv2.imshow(f'{ch}', img.squeeze())
        if (ret["content"].get("DEFORM") is not None) and (ret["content"].get("F6") is not None):
            f=ret["content"]["F6"].flatten()
            deform = ret["content"]["DEFORM"].squeeze()
            deform_img = cv2.cvtColor(deform, cv2.COLOR_GRAY2BGR)
            z = 20
            for var in f:
                cv2.putText(deform_img, f'{var:.3f}',
                        (5, z), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                z += 20
            if (ret["content"].get("CONTACT_POINT") is not None):
                p = ret['content']['CONTACT_POINT'].squeeze()
                p = p.reshape(-1, p.shape[-1])
                for item in p:
                    x, y = int(item[0]), int(item[1])
                    cv2.circle(deform_img,(x, y), 2, (0,0,255.0))
                    cv2.putText(deform_img, f'{item[2]:.2f}',
                            (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255.0), 2)
            cv2.imshow(f'{ch}_deform', deform_img)

t_end = time.time_ns() * 1e-9

touch.stop()
cv2.destroyAllWindows()

summary = touch.summary()
summary = json.loads(summary)
print(json.dumps(summary, indent=2))

# throughput statictics
user_got_frame = 0
total_time = 0
for item in summary['fingers']:
    if item['recv_info'] is not None and item['recv_info']['start_ts'] > 0:
        user_got_frame += item['recv_info']['frame_got']
        total_time += t_end - item['recv_info']['start_ts']
print(f'sdk got frame: {user_got_frame} in {total_time} s, frame rate: {user_got_frame / total_time}')
print(f'user got frame: {len(latency)} in {total_time} s, frame rate: {len(latency) / total_time}')

# latency statictics
latency = np.array(latency)
plt.hist(latency, bins=50, color='blue', edgecolor='white', alpha=0.7)
plt.title(f'mean: {latency.mean()} stddev: {latency.std()}')
plt.xlabel('latency')
plt.ylabel('frequency')
plt.show()
