#include "nms.h"
    namespace sharpa {
    namespace tactile {
    static float select_thres_;
    static int col_;
    static int row_;
    static int img_width_;
    static int img_height_;
    static const int neighbour_grid_size_ = 5;
    static const int neighbour_grid_[neighbour_grid_size_][2] = {
        {0, 1}, {1, 1}, {0, 2}, {1, 2}, {2, 1},
    };

    static bool _is_valid(int x, int y) {
        return (x >= 0 && x < col_ && y >= 0 && y < row_);
    };

    static void _select(int shift,
                        int ret_size,
                        float* ptr_contact_point,
                        float* ptr_ret_) {
        *(ptr_ret_ + ret_size * 3) = *(ptr_contact_point + shift) * img_width_;
        *(ptr_ret_ + ret_size * 3 + 1) = *(ptr_contact_point + shift + 1) * img_height_;
        *(ptr_ret_ + ret_size * 3 + 2) = *(ptr_contact_point + shift + 2);
    };

    static bool _is_peak(int shift, float* ptr_contact_point) {
        // shift for confidence
        shift = shift + 2;
        float curr_conf = *(ptr_contact_point + shift);

        if (curr_conf < select_thres_)
            return false;

        int x = (shift / 3) % row_;
        int y = (shift / 3) / row_;
        for (int idx = 0; idx < neighbour_grid_size_; idx++) {
            int dx = neighbour_grid_[idx][0];
            int dy = neighbour_grid_[idx][1];
            if (_is_valid(x - dx, y - dy) &&
                (*(ptr_contact_point + shift - 3 * dx - 3 * row_ * dy) > curr_conf))
                return false;
            if (_is_valid(x + dy, y - dx) &&
                (*(ptr_contact_point + shift + 3 * dy - 3 * row_ * dx) > curr_conf))
                return false;
            if (_is_valid(x + dx, y + dy) &&
                (*(ptr_contact_point + shift + 3 * dx + 3 * row_ * dy) > curr_conf))
                return false;
            if (_is_valid(x - dy, y + dx) &&
                (*(ptr_contact_point + shift - 3 * dy + 3 * row_ * dx) > curr_conf))
                return false;
        }
        return true;
    };

    int nms_execute(float* ptr_contact_point, float* ptr_ret) {
        int ret_size = 0;
        int curr_shift = 0;
        while (curr_shift < col_ * row_ * 3) {
            if (_is_peak(curr_shift, ptr_contact_point)) {
                _select(curr_shift, ret_size, ptr_contact_point, ptr_ret);
                ++ret_size;
            }
            curr_shift += 3;
        }
        return ret_size;
    };

    void nms_init(int col, int row, int img_height, int img_width, float select_thres) {
        col_ = col;
        row_ = row;
        img_height_ = img_height;
        img_width_ = img_width;
        select_thres_ = select_thres;
    };

    }  // namespace tactile
}  // namespace sharpa
