#include <touch.h>

#include <opencv2/opencv.hpp>
#include <fmt/format.h>

#include <mutex>

using FramePtr = sharpa::tactile::Frame::Ptr;
using DataPtr = sharpa::tactile::DataBlock::Ptr;

std::map<int, std::tuple<DataPtr, DataPtr, DataPtr, DataPtr>> frame;
std::mutex lock;

void set_frame(int channel, DataPtr pImg, DataPtr pF6, DataPtr pDeform, DataPtr pContactPoint) {
    std::unique_lock<std::mutex> _{lock};
    frame[channel] = {pImg, pF6, pDeform, pContactPoint};
}

std::tuple<DataPtr, DataPtr, DataPtr, DataPtr> get_frame(int channel) {
    std::unique_lock<std::mutex> _{lock};
    auto frame_iter = frame.find(channel);
    if(frame_iter == frame.end()) return {nullptr, nullptr, nullptr, nullptr};
    else return frame_iter->second;
}

int main() {
    std::vector<int> channels = {0, 1, 2, 3, 4};
    for(auto ch : channels) {
        cv::namedWindow(std::to_string(ch), cv::WINDOW_NORMAL);
        cv::namedWindow("deform_" + std::to_string(ch), cv::WINDOW_NORMAL);
    }

    const auto kHostIp{"192.168.10.240"};
    const auto kHostPort{50001};

    sharpa::tactile::TouchSetting setting{
        .model_path = {{"config/models/DEV2pin_400_de7d860c.onnx",
                        channels}},
        .fps=180,
        .infer_from_device = false,
    };
    std::vector<std::string> board_ip{"192.168.10.20"};
    sharpa::tactile::Touch touch(kHostIp, kHostPort, channels, board_ip, setting);

    touch.set_callback([](FramePtr f) {
        auto img_iter = f->content.find("RAW");
        auto f6_iter = f->content.find("F6");
        auto deform_iter = f->content.find("DEFORM");
        auto contact_point_iter = f->content.find("CONTACT_POINT");

        set_frame(f->channel
            , img_iter == f->content.end() ? nullptr : img_iter->second
            , f6_iter == f->content.end() ? nullptr : f6_iter->second
            , deform_iter == f->content.end() ? nullptr : deform_iter->second
            , contact_point_iter == f->content.end() ? nullptr : contact_point_iter->second
        );
    });

    if (touch.start() != 0)
        throw std::runtime_error("HsTouch start failed");

    char key{0};
    while(key != 27) {
        key = cv::waitKey(1);
        if (key == 't') touch.calib_zero();
        for(auto ch : channels) {
            auto [img, f6, deform, contact_point_db] = get_frame(ch);
            if(!img) continue;
            cv::Mat cv_img(img->shape()[1], img->shape()[2], CV_8UC1, img->data());
            cv::imshow(std::to_string(ch), cv_img);

            /* display deform */
            if(!deform || !f6) continue;
            cv::Mat cv_deform(deform->shape()[1], deform->shape()[2], CV_8UC1, deform->data());
            cv::Mat cv_deform_display;
            cv::cvtColor(cv_deform, cv_deform_display, cv::COLOR_GRAY2BGR);

            /* display f6 */
            auto f6_tensor = f6->as<float>();
            int z{20};
            for(const auto &idx : f6_tensor.shape().all_indices()) {
                cv::putText(cv_deform_display, 
                        fmt::format("{:.3f}", f6_tensor.at(idx)),
                        {5, z}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 0}, 2);
                z += 20;
            }

            /* display contact point */
            if(contact_point_db) {
                auto contact_point = contact_point_db->as<float>();
                for (size_t idx = 0; idx < contact_point.shape().size() / 3; ++idx) {
                    cv::Point pos{int(contact_point.at({idx, 0})), int(contact_point.at({idx, 1}))};
                    cv::circle(cv_deform_display, pos, 2, {0, 0, 255}); 
                    cv::putText(cv_deform_display, 
                                fmt::format("{:.2f}", contact_point.at({idx, 2})),
                                pos + cv::Point(10, 10),
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 0, 255}, 2);
                }
            }
            cv::imshow("deform_" + std::to_string(ch), cv_deform_display);
        }        
    }
    touch.stop();
}