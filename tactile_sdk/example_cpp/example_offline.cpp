#include <fmt/format.h>
#include <touch.h>
#include <opencv2/opencv.hpp>

#include <unistd.h>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>
#include "infer_offline.h"

bool readPicToFloat(const std::string& filePath, cv::Mat& image) {
    // read picture as grayscale image
    cv::Mat bmpImage = cv::imread(filePath, cv::IMREAD_GRAYSCALE);

    if (bmpImage.empty()) {
        std::cerr << "wrong path, please check if it exists: " << filePath << std::endl;
        return false;
    }

    if (bmpImage.channels() != 1) {
        std::cerr << "accept single channel image, convert to gray" << std::endl;
        cv::cvtColor(bmpImage, bmpImage, cv::COLOR_BGR2GRAY);
    }

    image = bmpImage;

    return true;
}

int main() {
    cv::namedWindow("initial", cv::WINDOW_NORMAL);
    cv::namedWindow("realtime", cv::WINDOW_NORMAL);
    cv::namedWindow("deform", cv::WINDOW_NORMAL);

    sharpa::tactile::InferOffline inferOffline("config/models/DEV2pin_400_de7d860c.onnx");

    std::string initial_path = "./data/initial.bmp";
    std::string realtime_path = "./data/realtime.bmp";
    cv::Mat initial_img, realtime_img;
    readPicToFloat(initial_path, initial_img);
    readPicToFloat(realtime_path, realtime_img);
    cv::imshow("realtime", realtime_img);
    cv::imshow("initial", initial_img);

    auto frame = inferOffline.infer(initial_img.data, realtime_img.data, 240 * 320);

    auto deform_iter = frame->content.find("DEFORM");
    auto f6_iter = frame->content.find("F6");
    auto contact_point_iter = frame->content.find("CONTACT_POINT");

    /* display deform */
    if (deform_iter == frame->content.end() || f6_iter == frame->content.end()) {
        std::cout << "inferred data not exist\n";
        return -1;
    }

    auto deform = deform_iter->second;
    cv::Mat cv_deform(deform->shape()[1], deform->shape()[2], CV_8UC1, deform->data());
    cv::Mat cv_deform_display;
    cv::cvtColor(cv_deform, cv_deform_display, cv::COLOR_GRAY2BGR);

    /* display f6 */
    auto f6 = f6_iter->second->as<float>();
    int z{20};
    for (const auto& idx : f6.shape().all_indices()) {
        cv::putText(cv_deform_display, fmt::format("{:.3f}", f6.at(idx)), {5, z},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 0, 0}, 2);
        z += 20;
    }

    auto contact_point = contact_point_iter->second->as<float>();
    for (int idx = 0; idx < contact_point.shape().size() / 3; idx++) {
        cv::circle(cv_deform_display,
                   {static_cast<int>(contact_point.at({idx, 0})),
                    static_cast<int>(contact_point.at({idx, 1}))},
                   2, {0, 0, 255});
        cv::putText(cv_deform_display, fmt::format("{:.2f}", contact_point.at({idx, 2})),
                    {static_cast<int>(contact_point.at({idx, 0})) + 10,
                     static_cast<int>(contact_point.at({idx, 1})) + 10},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 0, 255}, 2);
    }

    cv::imshow("deform", cv_deform_display);
    cv::waitKey(0);
}
