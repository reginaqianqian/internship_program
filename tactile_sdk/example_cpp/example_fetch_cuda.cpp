#include <touch.h>
#include <opencv2/opencv.hpp>
#include <fmt/format.h>

#include <optional>

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

    if (touch.start() != 0)
        throw std::runtime_error("HsTouch start failed");

    char key{0};
    while(key != 27) {
        key = cv::waitKey(1);
        if (key == 't') touch.calib_zero();
        for(auto ch : channels) {
            auto frame = touch.fetch(ch, 0.1);
            if(!frame) continue;
            auto img = frame->content["RAW"];
            cv::Mat cv_img(img->shape()[1], img->shape()[2], CV_8UC1, img->data());
            cv::imshow(std::to_string(ch), cv_img);

            auto deform_iter = frame->content.find("DEFORM");
            auto f6_iter = frame->content.find("F6");
            auto contact_point_iter = frame->content.find("CONTACT_POINT");

            /* display deform */
            if(deform_iter == frame->content.end() || f6_iter == frame->content.end() || contact_point_iter == frame->content.end()) {
                std::cout << "inferred data not exist\n";
                continue;
            }

            auto deform = deform_iter->second;
            cv::Mat cv_deform(deform->shape()[1], deform->shape()[2], CV_8UC1, deform->data());
            cv::Mat cv_deform_display;
            cv::cvtColor(cv_deform, cv_deform_display, cv::COLOR_GRAY2BGR);

            /* display f6 */
            auto f6 = f6_iter->second->as<float>();
            int z{20};
            for(const auto &idx : f6.shape().all_indices()) {
                cv::putText(cv_deform_display, 
                        fmt::format("{:.3f}", f6.at(idx)),
                        {5, z}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 0}, 2);
                z += 20;
            }

            /* display contact point */
            if(contact_point_iter != frame->content.end()) {
                auto contact_point = contact_point_iter->second->as<float>();
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

