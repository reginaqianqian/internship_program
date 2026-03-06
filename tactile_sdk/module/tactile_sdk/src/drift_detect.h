#ifndef SHARPA_TACTILE_DRIFT_DETECT_H_
#define SHARPA_TACTILE_DRIFT_DETECT_H_

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "def.h"    
#include "tensor.h"  

namespace sharpa {
namespace tactile {

struct ChannelData {
    ChannelData()
        : is_touch(false), is_calib(false), ts(0.0), prev_ts(0.0), last_calib_ts(0.0) {}

    cv::Mat first_frame;
    cv::Mat prev_frame;
    cv::Mat second_diff;
    cv::Mat cur_frame;

    std::atomic<bool> is_touch;
    std::atomic<bool> is_calib;

    double ts;
    double prev_ts;
    double last_calib_ts;

    std::deque<bool> calib_window;

    std::shared_ptr<ChannelData> clone_shared() const {
        std::shared_ptr<ChannelData> cloned = std::make_shared<ChannelData>();

        cloned->is_touch = is_touch.load();
        cloned->is_calib = is_calib.load();

        cloned->ts = ts;
        cloned->prev_ts = prev_ts;
        cloned->last_calib_ts = last_calib_ts;

        if (!first_frame.empty()) {
            cloned->first_frame = first_frame.clone(); 
        }
        if (!prev_frame.empty()) {
            cloned->prev_frame = prev_frame.clone();
        }
        if (!second_diff.empty()) {
            cloned->second_diff = second_diff.clone();
        }
        if (!cur_frame.empty()) {
            cloned->cur_frame = cur_frame.clone();
        }

        cloned->calib_window = calib_window;  
        return cloned;
    }
};

class drift_detect {
public:
    explicit drift_detect(double frame_rate,
                        //   std::mutex& calib_mutex,
                        //   std::map<int, std::shared_ptr<int>>& calib_num,
                          std::function<void(int)> func_calib,
                          int imgheight = 240,
                          int imgwidth = 320);
    ~drift_detect();

    bool auto_calib(const std::shared_ptr<Frame> frame);
    bool start();
    bool stop();

private:
    static const int kChannelNum = 10;
    static const int kThreshold1 = 6000;  // 1000 HA3 2000 HA4
    static const int kThreshold2 = 10;    // 10 HA3 2 HA4
    static const int kCalibWindowSize = 30;
    static constexpr double kCalibInterval = 2.0;      // 2 seconds
    static constexpr double kFpsRatioThreshold = 0.6;  // 2 seconds

    const int kImgHeight;
    const int kImgWidth;

    std::vector<std::shared_ptr<ChannelData>> ch_data;
    std::unordered_map<int, std::mutex> ch_mtx;

    std::function<void(int)> func_calib_;

    std::condition_variable cv;
    double fps;
    double min_interval;
    std::atomic<bool> running{false};
    std::thread worker;
    std::mutex q_mtx;
    std::queue<int> global_q;

    void worker_function();
    int process_img(const cv::Mat& img, int height, int width, int channel, int id);
    // std::map<int, std::shared_ptr<int>>& calib_num_;  // 0 normal, 1 to update ref_image, 2 to update offset
    // std::mutex& calib_mutex_;
};

}  // namespace tactile
}  // namespace sharpa

#endif  // SHARPA_TACTILE_DRIFT_DETECT_H_
