from tactile_sdk.sharpa.tactile import Touch, TouchSetting
import cv2
import numpy as np
import pyvista as pv

# pyvista setup
plotter = pv.Plotter(title='tactile display')
mesh = pv.read('config/static/thumb_ha4.obj')

line_mesh = [pv.Line((0, 0, 0), (0, 0, 0)) for _ in range(2)]
actor = plotter.add_mesh(mesh, texture=pv.numpy_to_texture(np.zeros((240, 240))))
plotter.add_mesh(line_mesh[0], color='green', line_width=8)
plotter.add_mesh(line_mesh[1], color='red', line_width=8)
plotter.show(auto_close=False, interactive_update=True)

# tactile setup
CHANNEL = 4
HOST_PORT = 50001
HOST_IP = '192.168.10.240'

cv2.namedWindow(f'{CHANNEL}', cv2.WINDOW_NORMAL)
cv2.namedWindow(f'{CHANNEL}_deform', cv2.WINDOW_NORMAL)

setting = TouchSetting(
    model_path={'config/models/EF_light_0905_500d5_DB_DLV2.onnx': [CHANNEL]},
    infer_from_device=False,
)
touch = Touch(HOST_IP, HOST_PORT, [CHANNEL], board_ip= ["192.168.10.20"], setting=setting)
if touch.start()!=0:
    raise RuntimeError('HsTouch start failed')

# main loop
key = 0
while key != 27:
    key = cv2.waitKey(1)
    if key == ord('t'): touch.calib_zero()
    ret = touch.fetch(CHANNEL, timeout=0)
    if ret is None: continue  # None means timeout
    if ret['content']['RAW'] is None:
        print('RAW not exist')
        continue
    cv2.imshow(f'{CHANNEL}', ret["content"]["RAW"].squeeze())
    
    if any(ret['content'][content_name] is None for content_name in ['F6', 'DEFORM']): continue
    f=ret['content']['F6'].flatten()
    deform = ret["content"]["DEFORM"].squeeze()
    contact = ret['content']['CONTACT_POINT']
    deform_texture = cv2.applyColorMap(255 - deform, cv2.COLORMAP_JET)
    deform_img = cv2.cvtColor(deform, cv2.COLOR_GRAY2BGR)
    z = 20
    for var in f:
        cv2.putText(deform_img, f'{var:.3f}',
                (5, z), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        z += 20

    is_force_drawn = False
    if contact is not None:
        intensity = np.empty((contact.shape[0],))
        for p_idx, p in enumerate(contact):
            col, row = int(p[0]), int(p[1])
            size = 5
            t, b, l, r = row - size, row + size + 1, col - size, col + size + 1
            t, b, l, r = max(0, t), min(b, deform_img.shape[0]), max(0, l), min(r, deform.shape[1])
            intensity[p_idx] = deform_img[t:b, l:r].sum()
            cv2.circle(deform_img, (col, row), 5, (0, 0, 255), 1)
        idx_main_contact = np.argmax(intensity)

        if intensity[idx_main_contact] > 0:
            p = contact[idx_main_contact]
            col, row = int(p[0]), int(p[1])
            contact_3d = touch.deform_map_uv(CHANNEL, row, col)
            if contact_3d is not None:  # None means invalid area
                contact_norm = contact_3d[3:6]
                f_contact = f[:3] * intensity[p_idx] / intensity.sum()
                f_norm = (f_contact @ contact_norm) * contact_norm
                f_shear = f_contact - f_norm

                # draw normal and shear
                force_viz_factor = 5.
                line_mesh[0].points[0] = contact_3d[:3]
                line_mesh[0].points[1] = contact_3d[:3] - force_viz_factor * f_norm
                line_mesh[1].points[0] = contact_3d[:3]
                line_mesh[1].points[1] = contact_3d[:3] - force_viz_factor * f_shear
                is_force_drawn = True
    if not is_force_drawn:
        # clear line_mesh
        line_mesh[0].points[0] = line_mesh[0].points[1] = (0, 0, 0)
        line_mesh[1].points[0] = line_mesh[1].points[1] = (0, 0, 0)

    cv2.imshow(f'{CHANNEL}_deform', deform_img)

    actor.texture = pv.numpy_to_texture(deform_texture)
    plotter.update()
    plotter.render()

touch.stop()
cv2.destroyAllWindows()
plotter.close()

