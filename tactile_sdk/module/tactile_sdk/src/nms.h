#pragma once
namespace sharpa {
namespace tactile {

void nms_init(int col,
              int row,
              int img_height = 240,
              int img_width = 240,
              float select_thres = 8.0);

int nms_execute(float* ptr_contact_point, float* ptr_ret);
}  // namespace tactile
}  // namespace sharpa
