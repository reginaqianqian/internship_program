//  drift_detect.cpp
#include "drift_detect.h"
#include <iostream>
#include <functional>
#include <opencv2/opencv.hpp>
#include "util.h"
#include <fmt/core.h>

namespace sharpa {
namespace tactile {

int drift_detect::process_img(const cv::Mat &img, int height, int width, int thread_id, int id) {
    // calculate mean
    cv::Mat reshaped = img.reshape(1, 1);
    cv::Mat sorted;
    cv::sort(reshaped, sorted, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);

    size_t n = sorted.cols;
    uint8_t median = (n % 2 == 0)
                         ? (sorted.at<uint8_t>(n / 2 - 1) + sorted.at<uint8_t>(n / 2)) / 2
                         : sorted.at<uint8_t>(n / 2);

    // calculate MAD (Median Absolute Deviation)
    cv::Mat diff;
    cv::absdiff(img, cv::Scalar(median), diff);

    cv::Mat diff_reshaped = diff.reshape(1, 1);
    cv::Mat sorted_diff;
    cv::sort(diff_reshaped, sorted_diff, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);

    int mad = (n % 2 == 0)
                  ? (sorted_diff.at<uint8_t>(n / 2 - 1) + sorted_diff.at<uint8_t>(n / 2)) / 2
                  : sorted_diff.at<uint8_t>(n / 2);

    // estimate stddev sigma = 1.4826 * MAD
    double sigma = 1.4826 * mad;
    double threshold_val = 3.0 * sigma;
    if(threshold_val>45) threshold_val=45;

    // binany
    cv::Mat binary;
    cv::threshold(img, binary, threshold_val, 255, cv::THRESH_BINARY);
    binary.convertTo(binary, CV_8U);

    // open operation (remove noise)
    if (id == 1) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
    }
    // get connected regions
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);

    // init output image
    cv::Mat filtered_img = cv::Mat::zeros(height, width, CV_8U);
    const int min_area = 3;
    for (int i = 1; i < num_labels; i++) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= min_area)
            filtered_img.setTo(255, labels == i);
    }

    return cv::countNonZero(filtered_img);
}

drift_detect::drift_detect(double frame_rate,
                        //    std::mutex& calib_mutex,
                        //    std::map<int, std::shared_ptr<int>>& calib_num,
                           std::function<void(int)> func_calib,
                           int imgheight,
                           int imgwidth)
    : fps(frame_rate),
      min_interval(1.0 / frame_rate),
    //   calib_mutex_(calib_mutex),
    //   calib_num_(calib_num),
      func_calib_(func_calib),
      kImgHeight(imgheight),
      kImgWidth(imgwidth) {
    if (fps <= 0)
        throw std::invalid_argument("FPS must be positive.");
    ch_data.clear();
    ch_data.reserve(kChannelNum);
    for (int i = 0; i < kChannelNum; ++i) {
        ch_data.emplace_back(std::make_unique<ChannelData>());
        ch_mtx.try_emplace(i);
    }
    for (auto &c : ch_data) {
        c->prev_frame = cv::Mat::zeros(kImgHeight, kImgWidth, CV_8U);
        c->second_diff = cv::Mat::zeros(kImgHeight, kImgWidth, CV_8U);
    }
}

bool drift_detect::start(){
    if(running.load()) return false;
    
    worker = std::thread(&drift_detect::worker_function, this);
    running = true;
    return true;
}

bool drift_detect::stop(){
    //if(!running.load()) return false;
    running = false;
    cv.notify_all();
    if(worker.joinable()) worker.join();
    return true;
}

drift_detect::~drift_detect()
{
    running = false;
    cv.notify_all();
    if (worker.joinable())
        worker.join();
}

bool drift_detect::auto_calib(const std::shared_ptr<Frame> frame)
{
    sharpa::tactile::assert_throw(frame->channel >= 0 && frame->channel < kChannelNum, "invalid channel");
    int ch = frame->channel;
    std::shared_ptr<ChannelData> cd;
    {
        std::lock_guard<std::mutex> lk(ch_mtx[ch]);
        cd=ch_data[ch]->clone_shared();
    }
    {
        if ((frame->ts - cd->ts) < min_interval)
            return false;

        if (cd->ts == 0) {
            cd->ts = frame->ts;
            cd->last_calib_ts = frame->ts;
        } else {
            cd->prev_ts = cd->ts;
            cd->ts = frame->ts;
            cd->prev_frame = std::move(cd->cur_frame.clone());
        }

        auto raw = frame->content.find("RAW");
        if (raw == frame->content.end())
            return false;
        cd->cur_frame = cv::Mat(kImgHeight, kImgWidth, CV_8U, raw->second->data()).clone();
    }
    {
        std::lock_guard<std::mutex> lk(ch_mtx[ch]);
        ch_data[ch]=cd;
    }
    {
        std::lock_guard<std::mutex> lk(q_mtx);
        global_q.push(ch);
        cv.notify_one();
    }
    return true;
}

void drift_detect::worker_function() {
    while (running.load()) {
        int ch = -1;

        { // get todo channel
            std::unique_lock<std::mutex> lk(q_mtx);
            cv.wait(lk, [this]()
                    { return !global_q.empty() || !running.load(); });

            if(!running.load()) break;
            ch = global_q.front();
            global_q.pop();
        }
        
        std::shared_ptr<ChannelData> cd;
        {
            std::lock_guard<std::mutex> _(ch_mtx[ch]);  
            cd=ch_data[ch]->clone_shared();
        }

        {
            if (cd->first_frame.empty())
                cd->first_frame = std::move(cd->cur_frame.clone());

            cv::Mat diff1, diff2, diff3;
            cv::absdiff(cd->cur_frame, cd->first_frame, diff1);
            cv::absdiff(cd->cur_frame, cd->prev_frame, diff2);
            // ratio of time interval compared with fps
            auto k = min_interval / (cd->ts - cd->prev_ts);
            if (k < kFpsRatioThreshold)
                diff2.convertTo(diff2, CV_8U, kFpsRatioThreshold);
            else
                diff2.convertTo(diff2, CV_8U, k);
            cv::absdiff(diff2, cd->second_diff, diff3);

            cd->second_diff = std::move(diff2.clone());
            cd->prev_frame = std::move(cd->cur_frame.clone());

            int res1 = process_img(diff1, kImgHeight, kImgWidth, ch, 1);
            int res2 = process_img(diff3, kImgHeight, kImgWidth, ch, 2);
            // abort if k is too small (frame rate too low)
            bool is_touch = !(res1 < kThreshold1 && res2 < kThreshold2 && k > kFpsRatioThreshold);
            cd->is_touch.store(is_touch);
            cd->calib_window.push_back(is_touch);
            bool is_calib = true;
            if (cd->calib_window.size() <= kCalibWindowSize)
                is_calib = false;
            else {
                while (cd->calib_window.size() > kCalibWindowSize) {
                    cd->calib_window.pop_front();
                }
                for (bool touching : cd->calib_window) {
                    if (touching) {
                        is_calib = false;
                        break;
                    }
                }
            }

            cd->is_calib.store(is_calib);

            if (k < kFpsRatioThreshold) {
                LOG_DEBUG(
                    "[calib][CH{}] frame rate warning, abort, interval: {}ms", ch,
                    (cd->ts - cd->prev_ts) * 1e3);
            }

            if (is_calib) {
                if (cd->ts - cd->last_calib_ts > kCalibInterval) {
                    func_calib_(ch);
                    cd->first_frame = std::move(cd->cur_frame.clone());
                }
            }
        }
    
        {
            std::lock_guard<std::mutex> _(ch_mtx[ch]);
            ch_data[ch]=cd;
        }
    }
}

} // namespace tactile
} // namespace sharpa
