#include <touch.h>
#include <unordered_map>

#include "board_cfg.h"
#include "build_info.h"
#include "receiver.h"
#include "version_map.h"

#include <nlohmann/json.hpp>

#include <unistd.h>
#include <exception>
#include <fstream>
#include <iostream>
#include <regex>

namespace sharpa {
namespace tactile {

struct Touch::Impl {
    std::string host_ip;
    int host_port;
    TouchSetting s;
    std::vector<int> channels{0, 1, 2, 3, 4};
    std::vector<std::string> board_ip{};
    std::shared_ptr<InferEngine> infer_engine{nullptr};
    std::unique_ptr<Receiver> recv{nullptr};
    std::unique_ptr<SafeQueue<RawFrame>> queue_in{nullptr};
    std::shared_ptr<ThreadPool> pool{nullptr};
    std::function<void(Frame::Ptr)> callback{};
    nlohmann::json summary{{"version", kVersion},
                           {"sn", nullptr},
                           {"fingers", nlohmann::json::array()},
                           {"udp_info", nlohmann::json::object()}};

    DataBlock::Ptr general_point, general_normal, thumb_point, thumb_normal;

    const int kPortParamServer_{59998};
    const int kPortUpdateServer_{60000};

    enum Type { Int, Float, String, Bool, Bytes };

    const std::map<std::string, Type> kParamType_ = {
        {"version", String},   {"ip", String},
        {"dest_ip", String},   {"stream_port", Int},
        {"exposure", Int},     {"gain", Int},
        {"fps", Int},          {"require_jpg", Bool},
        {"require_raw", Bool}, {"require_deform", Bool},
        {"require_f6", Bool},  {"require_infer", Bool},
        {"f6_offset", Bytes},  {"deform_offset", Bytes},
        {"sn", String},        {"ftsid", String},
        {"flash_exp", String}, {"ini_img", Bytes},
        {"disable_finger_led", String},
    };

    const std::vector<std::string> kActionType_ = {
        "refresh",
        "reboot",
        "update_ini_img",
    };

    std::any msgpack2any(Type t, msgpack11::MsgPack obj) {
        try {
            switch (t) {
                case Impl::Type::Int: {
                    assert_throw(obj.is_int32(), "type mismatch");
                    return obj.int_value();
                }
                case Impl::Type::Bool: {
                    assert_throw(obj.is_bool(), "type mismatch");
                    return obj.bool_value();
                }
                case Impl::Type::Float: {
                    assert_throw(obj.is_float64(), "type mismatch");
                    return obj.float64_value();
                }
                case Impl::Type::String: {
                    assert_throw(obj.is_string(), "type mismatch");
                    return obj.string_value();
                }
                case Impl::Type::Bytes: {
                    assert_throw(obj.is_binary(), "type mismatch");
                    return obj.binary_items();
                }
                default:
                    assert_throw(false, "invalid type");
            }
        } catch (std::exception& e) {
            LOG_ERROR_RAW(e.what());
            assert_throw(false, e.what());
        }
        throw std::runtime_error("panic");
    }

    msgpack11::MsgPack any2msgpack(Type t, std::any obj) {
        try {
            switch (t) {
                case Impl::Type::Int:
                    return std::any_cast<int>(obj);
                case Impl::Type::Bool:
                    return std::any_cast<bool>(obj);
                case Impl::Type::Float:
                    return std::any_cast<float>(obj);
                case Impl::Type::String:
                    return std::any_cast<std::string>(obj);
                case Impl::Type::Bytes:
                    return std::any_cast<std::vector<uint8_t>>(obj);
                default:
                    assert_throw(false, "invalid type");
            }
        } catch (std::bad_any_cast& e) {
            LOG_ERROR_RAW(e.what());
            assert_throw(false, e.what());
        }
        throw std::runtime_error("panic");
    }

    bool any_equal(Type t, std::any x, std::any y) {
        try {
            switch (t) {
                case Impl::Type::Int:
                    return std::any_cast<int>(x) == std::any_cast<int>(y);
                case Impl::Type::Bool:
                    return std::any_cast<bool>(x) == std::any_cast<bool>(y);
                case Impl::Type::Float:
                    return std::any_cast<float>(x) == std::any_cast<float>(y);
                case Impl::Type::String:
                    return std::any_cast<std::string>(x) == std::any_cast<std::string>(y);
                case Impl::Type::Bytes: {
                    auto x_vec = std::any_cast<std::vector<uint8_t>>(x);
                    auto y_vec = std::any_cast<std::vector<uint8_t>>(y);
                    if (x_vec.size() != y_vec.size())
                        return false;
                    for (int i = 0; i < x_vec.size(); ++i) {
                        if (x_vec[i] != y_vec[i])
                            return false;
                    }
                    return true;
                }
                default:
                    assert_throw(false, "invalid type");
            }
        } catch (std::bad_any_cast& e) {
            LOG_ERROR_RAW(e.what());
            assert_throw(false, e.what());
        }
        throw std::runtime_error("panic");
    }

    std::tuple<int, msgpack11::MsgPack> board_cfg_(const std::string& ip,
                                                   const msgpack11::MsgPack& req,
                                                   double timeout) {
        auto req_ret = msgpack_request(ip, kPortParamServer_, req, timeout);
        auto err_no = std::get<0>(req_ret);
        auto ret_opt = std::get<1>(req_ret);
        if (err_no != 0)
            return {err_no, {}};
        do {
            if (!ret_opt.has_value()) {
                if (err_no != 0)
                    break;
                err_no = static_cast<int>(TACTILE_ERROR_CODE::TCP_RECV_ERROR);
            }
            auto ret = *ret_opt;
            LOG_DEBUG_RAW(parse_msgpack(ret));

            if (ret["ret_code"].int_value() != 0) {
                LOG_WARN("ret code {}. msg: {}", ret["ret_code"].int_value(),
                         ret["msg"].string_value());
                if (err_no != 0)
                    break;
                err_no = static_cast<int>(TACTILE_ERROR_CODE::CMD_EXEC_ERROR);
                break;
            }
            if (ret["content"].is_null())
                return {static_cast<int>(TACTILE_ERROR_CODE::NO_ERROR), true};
            return {static_cast<int>(TACTILE_ERROR_CODE::NO_ERROR), ret["content"]};
        } while (false);
        LOG_WARN("device-cfg failed, device IP: {}", ip);
        return {err_no, {}};
    }

    std::tuple<int, std::map<int, std::any>> board_cfg(const std::string& ip,
                                                       int channel,
                                                       const std::string& key,
                                                       std::any value,
                                                       double timeout) {
        int err_no = 0;
        if (auto it = kParamType_.find(key); it != kParamType_.end()) {
            if (!value.has_value()) {
                auto cfg_res = board_cfg_(ip,
                                          msgpack11::MsgPack::object{
                                              {"type", "get"},
                                              {"channel", channel},
                                              {"key", key},
                                          },
                                          timeout);
                err_no = std::get<0>(cfg_res);
                auto cfg_ret = std::get<1>(cfg_res).array_items();

                if (err_no != 0)
                    return {err_no, {}};
                if (cfg_ret.empty())
                    return {static_cast<int>(TACTILE_ERROR_CODE::UNKNOWN_ERROR), {}};
                std::map<int, std::any> ret;
                for (auto& item : cfg_ret) {
                    ret.emplace(item["channel"].int_value(),
                                msgpack2any(it->second, item["value"]));
                }
                return {static_cast<int>(TACTILE_ERROR_CODE::NO_ERROR), ret};
            } else {
                auto cfg_res = board_cfg_(ip,
                                          msgpack11::MsgPack::object{
                                              {"type", "set"},
                                              {"channel", channel},
                                              {"key", key},
                                              {"value", any2msgpack(it->second, value)},
                                          },
                                          timeout);
                err_no = std::get<0>(cfg_res);
                auto cfg_ret = std::get<1>(cfg_res);
                if (err_no != 0)
                    return {err_no, {}};
                if (cfg_ret.is_null())
                    return {static_cast<int>(TACTILE_ERROR_CODE::UNKNOWN_ERROR), {}};
                else
                    return {static_cast<int>(TACTILE_ERROR_CODE::NO_ERROR), {{-1, true}}};
            }
        } else if (auto it = std::find(kActionType_.begin(), kActionType_.end(), key);
                   it != kActionType_.end()) {
            auto cfg_res =
                board_cfg_(ip,
                           msgpack11::MsgPack::object{
                               {"type", "action"},
                               {"key", key},
                               {"args", msgpack11::MsgPack::object{{"channel", channel}}},
                           },
                           timeout);
            err_no = std::get<0>(cfg_res);
            auto cfg_ret = std::get<1>(cfg_res);
            if (err_no != 0)
                return {err_no, {}};
            if (cfg_ret.is_null())
                return {static_cast<int>(TACTILE_ERROR_CODE::UNKNOWN_ERROR), {}};
            else
                return {0, {{-1, true}}};
        } else {
            LOG_ERROR("key {} not exist", key);
            return {static_cast<int>(TACTILE_ERROR_CODE::BOARD_CFG_KEY_NOT_EXIST), {}};
        }
        // throw std::runtime_error(fmt::format("key {} not exist", key));
    }

    std::tuple<int, std::map<int, std::any>> check_param(const std::string& ip,
                                                         const std::string& key,
                                                         std::any value,
                                                         bool& need_set) {
        auto it = kParamType_.find(key);
        assert_throw(it != kParamType_.end(), fmt::format("key {} not exist", key));
        auto timeout{1.0};
        auto ret_cfg = board_cfg(ip, -1, key, {}, timeout);
        auto err_no = std::get<0>(ret_cfg);
        auto ret = std::get<1>(ret_cfg);

        bool this_need_set = ret.empty() || (err_no != 0);
        if (!ret.empty()) {
            for (const auto& [channel, value_got] : ret) {
                if (!any_equal(it->second, value, value_got)) {
                    this_need_set = true;
                    break;
                }
            }
        }
        if (this_need_set) {
            LOG_INFO("setting {} key {}", ip, key);
            board_cfg(ip, -1, key, value, timeout);
            auto ret_cfg = board_cfg(ip, -1, key, {}, timeout);
            err_no = std::get<0>(ret_cfg);
            ret = std::get<1>(ret_cfg);
            if (err_no != 0) {
                LOG_ERROR("set {} failed", key);
                return {err_no, {}};
            }
            if (ret.empty()) {
                LOG_ERROR("set {} failed", key);
                return {static_cast<int>(TACTILE_ERROR_CODE::UNKNOWN_ERROR), {}};
                // return std::make_tuple<int, std::map<int, std::any>>(
                //     static_cast<int>(TACTILE_ERROR_CODE::UNKNOWN_ERROR), {});
            }
            // assert_throw(!ret.empty(), "set failed");
        }
        need_set |= this_need_set;
        return {0, ret};
    }

    std::vector<int> update_firmwares(std::vector<std::string> board_ip_candidates) {
        /* version related */
        std::vector<int> res = std::vector<int>(board_ip_candidates.size(), 0);
        std::string fm_version_exp = version_map(kVersion);
        std::vector<std::string> to_wait;
        for (int i = 0; i < board_ip_candidates.size(); i++) {
            auto ip = board_ip_candidates[i];
            auto ret_cfg = board_cfg(ip, -1, "version", {}, 0.4);
            auto err_no = std::get<0>(ret_cfg);
            auto ret = std::get<1>(ret_cfg);
            if (err_no != 0) {
                res[i] = err_no;
                continue;
            }
            if (ret.empty()) {
                res[i] = static_cast<int>(TACTILE_ERROR_CODE::UNKNOWN_ERROR);
                continue;
            }
            // board_ip.push_back(ip);

            auto fm_version = std::any_cast<std::string>(ret.begin()->second);
            if (fm_version == fm_version_exp)
                continue;
            LOG_INFO("firmware version {} excepted version: {}", fm_version,
                     fm_version_exp);
            assert_throw(s.update_firmware == true,
                         "Terminated because the firmware version does NOT MATCH.");

            size_t last_dot = ip.find_last_of('.');
            assert_throw(last_dot != std::string::npos, "bad ip");
            std::string last_octet = ip.substr(last_dot + 1);
            auto firmware_path = fmt::format("{}/etc/firmware-{}-{}.tar", kInstallDir,
                                             fm_version_exp, last_octet);
            err_no = board_update(ip, firmware_path);
            if (err_no != 0) {
                res[i] = err_no;
            }
            // no matter board_update success or not, it is safe to reboot board
            to_wait.push_back(ip);
        }

        /* wait for readiness */
        if (!to_wait.empty()) {
            // sleep(5); /* THIS IS VERY BAD! */
            while (!is_ready(to_wait)) {
                LOG_INFO("wait for hardware to reboot");
                sleep(2);
            }
            LOG_INFO("boot finished");
        }
        return res;
    }

    int check_board_params() {
        int err_no = 0;
        for (const auto& ip : board_ip) {
            auto is_set{false};
            /* TO_DO get SN and ftsid */
            auto fps_ret = check_param(ip, "fps", s.fps, is_set);
            err_no = std::get<0>(fps_ret);
            auto fps_d = std::get<1>(fps_ret);
            if (err_no != 0)
                return err_no;
            auto require_raw_ret =
                check_param(ip, "require_raw", false, is_set); /* never use raw */
            err_no = std::get<0>(require_raw_ret);
            auto require_raw_d = std::get<1>(require_raw_ret);
            if (err_no != 0)
                return err_no;

            auto require_infer_ret =
                check_param(ip, "require_infer", s.infer_from_device, is_set);
            err_no = std::get<0>(require_infer_ret);
            auto require_infer_d = std::get<1>(require_infer_ret);
            if (err_no != 0)
                return err_no;

            auto require_jpeg_ret =
                check_param(ip, "require_jpg", s.require_jpeg, is_set);
            err_no = std::get<0>(require_jpeg_ret);
            auto require_jpeg_d = std::get<1>(require_jpeg_ret);
            if (err_no != 0)
                return err_no;

            for (const auto& [ch, _] : fps_d) {
                summary["fingers"].push_back(nlohmann::json{
                    {"channel", std::any_cast<int>(ch)},
                    {"fstid", nullptr},
                    {"fps", std::any_cast<int>(fps_d[ch])},
                    {"require_infer", std::any_cast<bool>(require_infer_d[ch])},
                    {"f6_offset", nullptr},
                    {"deform_offset", nullptr},
                    {"recv_info",
                     nlohmann::json{
                         {"packet_got", 0},
                         {"packet_loss", 0},
                         {"frame_got", 0},
                         {"frame_loss", 0},
                         {"start_ts", -1.},
                     }},
                });
            }
            auto timeout{1.0};
            if (is_set) {
                LOG_INFO("refresh hardware");
                auto ret = board_cfg(ip, -1, "refresh", {}, timeout);
                err_no = std::get<0>(ret);
                if (err_no != 0)
                    return err_no;
            }
        }
        summary["udp_info"] =
            nlohmann::json{{"port", host_port}, {"total_pack", 0}, {"invalid_pack", 0}};
        return 0;
    }

    // void summary_init() {
    //     for (const auto& ip : board_ip) {
    //         auto is_set{false};
    //         /* TO_DO get SN and ftsid */
    //         auto fps_d = check_param(ip, "fps", s.fps, is_set);
    //         for (const auto& [ch, _] : fps_d) {
    //             summary["fingers"].push_back(nlohmann::json{
    //                 {"channel", std::any_cast<int>(ch)},
    //                 {"fstid", nullptr},
    //                 {"fps", std::any_cast<int>(fps_d[ch])},
    //                 {"require_infer", std::any_cast<bool>(require_infer_d[ch])},
    //                 {"f6_offset", nullptr},
    //                 {"deform_offset", nullptr},
    //                 {"recv_info",
    //                  nlohmann::json{
    //                      {"packet_got", 0},
    //                      {"packet_loss", 0},
    //                      {"frame_got", 0},
    //                      {"frame_loss", 0},
    //                      {"start_ts", -1.},
    //                  }},
    //             });
    //         }
    //     }
    //     summary["udp_info"] =
    //         nlohmann::json{{"port", host_port}, {"total_pack", 0}, {"invalid_pack",
    //         0}};
    // }

    void deform_map_init() {
        std::string config_dir{fmt::format("{}/etc/config", kInstallDir)};
        if (!s.config_dir.empty()) {
            config_dir = s.config_dir;
        }
        auto general_point_path =
            fmt::format("{}/static/general_ha4_map_point.txt", config_dir);
        auto general_normal_path =
            fmt::format("{}/static/general_ha4_map_normal.txt", config_dir);
        auto thumb_point_path =
            fmt::format("{}/static/thumb_ha4_map_point.txt", config_dir);
        auto thumb_normal_path =
            fmt::format("{}/static/thumb_ha4_map_normal.txt", config_dir);

        size_t deform_height = infer_engine->deform_height();
        size_t deform_width = infer_engine->deform_width();
        size_t block_size = deform_height * deform_width * 3;  // 3 value for every pixel
        general_point = f32_from_txt(general_point_path, block_size);
        if (general_point)
            general_point->reshape({{deform_height, deform_width, 3}});
        general_normal = f32_from_txt(general_normal_path, block_size);
        if (general_normal)
            general_normal->reshape({{deform_height, deform_width, 3}});
        thumb_point = f32_from_txt(thumb_point_path, block_size);
        if (thumb_point)
            thumb_point->reshape({{deform_height, deform_width, 3}});
        thumb_normal = f32_from_txt(thumb_normal_path, block_size);
        if (thumb_normal)
            thumb_normal->reshape({{deform_height, deform_width, 3}});
    }

    int board_update(const std::string& ip, const std::string& firmware_path) {
        int err_no = 0;
        uintmax_t file_size = 0;
        std::ifstream in(firmware_path, std::ifstream::ate | std::ifstream::binary);
        if (in.is_open()) {
            file_size = static_cast<uintmax_t>(in.tellg());
            in.close();
        }
        std::regex version_regex(R"(-(.+)\.)");
        std::smatch matches;
        std::string version;
        const double timeout = 1.0;
        do {
            if (std::regex_search(firmware_path, matches, version_regex) &&
                matches.size() > 1) {
                version = matches[1].str();
            } else
                break;

            std::string upload_msg =
                "upload##" + version + "##" + std::to_string(file_size) + "##";
            upload_msg.resize(128, ' '); /* pad to 128 characters */

            auto tcp_ret =
                tcp_request(ip, kPortUpdateServer_,
                            std::vector<uint8_t>(upload_msg.begin(), upload_msg.end()),
                            timeout, firmware_path);
            err_no = std::get<0>(tcp_ret);
            if (err_no != 0)
                break;
            auto res_upload = std::get<1>(tcp_ret);
            if (res_upload.empty()) {
                err_no = static_cast<int>(TACTILE_ERROR_CODE::TCP_RECV_ERROR);
                break;
            }

            std::string update_msg = "update##" + version + "##";
            update_msg.resize(128, ' '); /* pad to 128 characters */
            auto tcp_ret1 = tcp_request(
                ip, kPortUpdateServer_,
                std::vector<uint8_t>(update_msg.begin(), update_msg.end()), timeout, "");
            err_no = std::get<0>(tcp_ret1);
            if (err_no != 0)
                break;
            auto res_update = std::get<1>(tcp_ret1);
            if (res_update.empty()) {
                err_no = static_cast<int>(TACTILE_ERROR_CODE::TCP_RECV_ERROR);
                break;
            }

            auto board_cfg_ret = board_cfg(ip, -1, "reboot", {}, timeout);
            err_no = std::get<0>(board_cfg_ret);
            if (err_no != 0)
                break;
            auto res_reboot = std::get<1>(board_cfg_ret);
            if (res_reboot.empty()) {
                err_no = static_cast<int>(TACTILE_ERROR_CODE::TCP_RECV_ERROR);
                break;
            }

            std::string combined_response;
            combined_response.insert(combined_response.end(), res_upload.begin(),
                                     res_upload.end());
            combined_response.insert(combined_response.end(), res_update.begin(),
                                     res_update.end());
            LOG_DEBUG_RAW(combined_response);
            return static_cast<int>(TACTILE_ERROR_CODE::NO_ERROR);
        } while (false);
        LOG_ERROR("failed to update board");
        return err_no;
    }

    bool is_ready(const std::vector<std::string>& board_ip_) {
        double timeout{1.0};
        std::vector<std::string> the_ips = board_ip_.empty() ? board_ip : board_ip_;
        for (const auto& ip : the_ips) {
            auto cfg_ret = board_cfg(ip, -1, "version", {}, timeout);
            auto err_no = std::get<0>(cfg_ret);
            auto ret = std::get<1>(cfg_ret);
            if (ret.empty() || err_no != 0) {
                LOG_WARN("board:{} is not ready", ip);
                return false;
            }
        }
        return true;
    }

    void reset_thread_pool() {
        pool = std::make_shared<ThreadPool>(s.num_worker);
        infer_engine->set_thread_pool(pool);
        recv->set_thread_pool(pool);
    }

    void get_set_init_image() {
        std::string key = "ini_img";
        for (const auto& ip : board_ip) {
            auto timeout{1.0};
            auto cfg_ret = board_cfg(ip, -1, key, {}, timeout);
            auto err_no = std::get<0>(cfg_ret);
            auto ret = std::get<1>(cfg_ret);
            if (ret.empty() || (err_no != 0)) {
                LOG_WARN("get ini_img failed for device {}", ip);
                continue;
            }
            for (auto [ch, _] : ret) {
                auto img_vec = std::any_cast<std::vector<uint8_t>>(ret[ch]);
                auto raw =
                    std::make_shared<DataBlock>(Shape{{1, 240, 320}}, sizeof(uint8_t));
                assert_throw(raw->nbytes() == img_vec.size(), "size mismatch");
                std::memcpy(raw->data(), img_vec.data(), img_vec.size());
                auto it = std::find(channels.begin(), channels.end(), ch);
                if (it != channels.end()) {
                    infer_engine->set_ref_image(ch, raw);
                }
            }
        }
    }
};

Touch::Touch(std::string host_ip,
             int host_port,
             std::optional<std::vector<int>> channels,
             std::vector<std::string> board_ip,
             std::optional<TouchSetting> setting,
             std::shared_ptr<LoggerBase> logger) {
    impl_ = std::make_shared<Impl>();
    if (setting.has_value())
        impl_->s = setting.value();
    else
        impl_->s = TouchSetting{};

    // If infer_from_device is not true and model_path already set and decode_jpeg not
    // ture, infer will not work cause decoded jpeg required for infer
    assert_throw(
        impl_->s.model_path.empty() || impl_->s.infer_from_device || impl_->s.decode_jpeg,
        "Decoded jpg REQUIRED for infer while decode jpeg option is set to false");
    impl_->host_ip = host_ip;
    impl_->host_port = host_port;
    if (channels.has_value())
        impl_->channels = std::move(*channels);
    // it is necessary to set board_ip from now on
    impl_->board_ip = std::move(board_ip);

    if (logger) {
        LoggerSingleton::getInstance().setLogger(logger);
    } else {
        LoggerSingleton::getInstance().setLogger(std::make_shared<LoggerDefault>());
    }

    // impl_->pool = std::make_shared<ThreadPool>(impl_->s.num_worker);

    // ignore updatge firmware update error
    impl_->update_firmwares(impl_->board_ip);

    impl_->infer_engine = std::make_shared<InferEngine>(
        impl_->s.model_path, impl_->channels, impl_->s.batch_size, impl_->s.buffer_size,
        &(impl_->callback), nullptr, impl_->s.infer_from_device);

    impl_->recv = std::make_unique<Receiver>(
        impl_->channels, impl_->host_ip, impl_->host_port, impl_->s.allowed_pack_loss,
        nullptr, impl_->infer_engine, impl_->s.decode_jpeg);

    impl_->deform_map_init();
}

void Touch::set_callback(std::function<void(Frame::Ptr)> callback) {
    impl_->callback = callback;
}

int Touch::start() {
    int err_no = 0;
    assert_throw(impl_ != nullptr, "initialize failed");

    if (impl_->board_ip.empty())
        return static_cast<int>(TACTILE_ERROR_CODE::BOARD_IP_NOT_CONFIGURED);

    err_no = impl_->check_board_params();
    if (err_no != 0)
        return err_no;

    if (!impl_->s.infer_from_device)
        impl_->get_set_init_image();
    impl_->reset_thread_pool();
    assert_throw(impl_->infer_engine != nullptr, "infer engine not initialized");
    assert_throw(impl_->recv != nullptr, "receiver not initialized");
    if (!impl_->infer_engine->start()) {
        // if return directly, reciver will never be started.
        LOG_INFO("tactile infer engine already run");
    }
    int ret = impl_->recv->start();
    return ret;
}

bool Touch::stop() {
    bool is_recv_stoped = impl_->recv->stop();
    bool is_infer_stopped = impl_->infer_engine->stop();
    impl_->pool->stop();
    return is_recv_stoped && is_infer_stopped;
}

bool Touch::is_sdk_running() {
    if (impl_ && impl_->recv && impl_->infer_engine) {
        return impl_->recv->is_running() && impl_->infer_engine->is_running();
    }
    return false;
}

Frame::Ptr Touch::fetch(int channel, double timeout) {
    return impl_->infer_engine->dequeue(channel, timeout);
}

bool Touch::is_ready(const std::vector<std::string>& board_ip) {
    if (impl_) {
        return impl_->is_ready(board_ip);
    }
    return false;
}

std::string Touch::summary() {
    auto ret = impl_->recv->summary();
    for (auto& [channel, stat] : ret.channel_statistic) {
        for (auto& f : impl_->summary["fingers"]) {
            if (channel != f["channel"])
                continue;
            f["recv_info"]["packet_got"] = stat.packet_got;
            f["recv_info"]["packet_loss"] = stat.packet_loss;
            f["recv_info"]["frame_got"] = stat.frame_got;
            f["recv_info"]["frame_loss"] = stat.frame_loss;
            f["recv_info"]["start_ts"] = stat.start_ts;
        }
    }
    impl_->summary["udp_info"]["total_pack"] = ret.total_pack;
    impl_->summary["udp_info"]["invalid_pack"] = ret.invalid_pack;
    return impl_->summary.dump();
}

bool Touch::calib_zero(int num_frames, int max_retry) {
    assert_throw(num_frames > 0 && max_retry > 0, "num_frames and max_retry should >0");
    for (auto ip : impl_->board_ip) {
        auto timeout{1.0};
        auto ret1 = board_cfg(ip, -1, "update_ini_img", {}, timeout);
        if (ret1.empty()) {
            LOG_DEBUG("update_ini_img action failed");
        }
    }
    if (impl_->s.infer_from_device) {
        return true;
    }
    return impl_->infer_engine->calib_zero(num_frames, max_retry);
}

std::map<int, std::any> Touch::board_cfg(const std::string& ip,
                                         int channel,
                                         const std::string& key,
                                         std::any value,
                                         double timeout) {
    auto cfg_ret = impl_->board_cfg(ip, channel, key, value, timeout);
    return std::get<1>(cfg_ret);
}

bool Touch::board_update(const std::string& ip, const std::string& firmware_path) {
    if (impl_) {
        auto update_ret = impl_->board_update(ip, firmware_path);
        return update_ret == 0;
    }
    return false;
}

std::optional<std::array<float, 6>> Touch::deform_map_uv(int channel,
                                                         size_t row,
                                                         size_t col) {
    DataBlock::Ptr p, n;
    if (channel == 4 || channel == 9) {
        assert_throw(impl_->thumb_point != nullptr && impl_->thumb_normal != nullptr,
                     "thumb point or thumb normal map not initialized");
        p = impl_->thumb_point;
        n = impl_->thumb_normal;
    } else if (0 <= channel || channel <= 9) {
        assert_throw(impl_->general_point != nullptr && impl_->general_normal != nullptr,
                     "general finger point or general finger normal map not initialized");
        p = impl_->general_point;
        n = impl_->general_normal;
    } else {
        LOG_WARN("channel {} invalid", channel);
        return std::nullopt;
    }
    auto tp = p->as<float>();
    auto tn = n->as<float>();

    if (tp.at({row, col, 0}) == 0.f)
        return std::nullopt;
    return {{tp.at({row, col, 0}), tp.at({row, col, 1}), tp.at({row, col, 2}),
             tn.at({row, col, 0}), tn.at({row, col, 1}), tn.at({row, col, 2})}};
}

float Touch::deform_map_value(uint8_t value_ui8) {
    if (value_ui8 < 100)
        return value_ui8 * 5e-3f;
    else
        return (value_ui8 - 100) * 3e-2f + 0.5f;
}

}  // namespace tactile
}  // namespace sharpa
