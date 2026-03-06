
#include <map>
#include <vector>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include <touch.h>
#include <iostream>
#include "infer_offline.h"

namespace py = pybind11;
using namespace py::literals;

namespace sharpa {
namespace tactile {

class PyLoggerBase : public LoggerBase {
public:
    using LoggerBase::LoggerBase;
    void debug(const std::string &msg) override {
        PYBIND11_OVERRIDE_PURE(void, LoggerBase, debug, msg);
    }
    void info(const std::string &msg) override {
        PYBIND11_OVERRIDE_PURE(void, LoggerBase, info, msg);
    }
    void warn(const std::string &msg) override {
        PYBIND11_OVERRIDE_PURE(void, LoggerBase, warn, msg);
    }
    void error(const std::string &msg) override {
        PYBIND11_OVERRIDE_PURE(void, LoggerBase, error, msg);
    }
};

// PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

template <typename T>
py::array_t<T> tensor2pyarray(DataBlock::Ptr db) {
    if(db->size() == 0) return py::array_t<T>();

    auto t = db->as<T>();
    return py::array_t<T>(
        t.shape().data()
        , t.stride()
        , t.data()
        , py::cast(db)
    );
}

std::map<std::string, py::array> py_frame(Frame::Ptr f) {
    py::array_t<uint8_t> raw, deform;         // raw pics
    py::array_t<uint8_t> raw_jpg, deform_jpg; //jpeg pics
    py::array_t<float> f6, contact_point; 
    if(f->content.find("RAW") != f->content.end()) {
        raw = tensor2pyarray<uint8_t>(f->content["RAW"]);
    }
    if(f->content.find("F6") != f->content.end()) {
        f6 = tensor2pyarray<float>(f->content["F6"]);
    }
    if(f->content.find("DEFORM") != f->content.end()) {
        deform = tensor2pyarray<uint8_t>(f->content["DEFORM"]);
    }
    if(f->content.find("RAW_JPG") != f->content.end()) {
        raw_jpg = tensor2pyarray<uint8_t>(f->content["RAW_JPG"]);
    }
    if(f->content.find("DEFORM_JPG") != f->content.end()) {
        deform_jpg = tensor2pyarray<uint8_t>(f->content["DEFORM_JPG"]);
    }
    if(f->content.find("CONTACT_POINT") != f->content.end()) {
        contact_point = tensor2pyarray<float>(f->content["CONTACT_POINT"]);
    }

    return {{"RAW", raw},
            {"F6", f6},
            {"DEFORM", deform},
            {"RAW_JPG", raw_jpg},
            {"DEFORM_JPG", deform_jpg},
            {"CONTACT_POINT", contact_point}};
}

py::object any2py(const std::any& a) {
    if(!a.has_value()) return py::none{};
    else if(a.type() == typeid(int)) return py::cast(std::any_cast<int>(a));
    else if(a.type() == typeid(double)) return py::cast(std::any_cast<double>(a));
    else if(a.type() == typeid(std::string)) return py::cast(std::any_cast<std::string>(a));
    else if(a.type() == typeid(bool)) return py::cast(std::any_cast<bool>(a));
    else if(a.type() == typeid(std::vector<uint8_t>)) {
        const auto &v = std::any_cast<const std::vector<uint8_t>&>(a);
        return py::bytes(
            reinterpret_cast<const char*>(v.data()),
            v.size()
        );
    } else throw std::runtime_error("bad type");
}

std::any py2any(py::object obj) {
    if(py::isinstance<py::int_>(obj)) return py::cast<int>(obj);
    else if(py::isinstance<py::float_>(obj)) return py::cast<double>(obj);
    else if(py::isinstance<py::str>(obj)) return py::cast<std::string>(obj);
    else if(py::isinstance<py::none>(obj)) return {};
    else if(py::isinstance<py::bytes>(obj) || py::isinstance<py::bytearray>(obj)) {
      py::buffer_info info(py::reinterpret_borrow<py::buffer>(obj).request());
      auto ptr = static_cast<uint8_t*>(info.ptr);
      return std::vector<uint8_t>(ptr, ptr + info.size);
    } else throw std::runtime_error("bad type");
}

bool array_valid(const py::array &arr) {
    if (!arr) return false;
    if (arr.size() == 0) return false;
    auto shape = arr.shape();
    for (py::ssize_t i = 0; i < arr.ndim(); ++i) {
        if (shape[i] == 0) return false;
    }
    return true;
}

PYBIND11_MODULE(tactile_sdk_py, m) {
    m.doc() = "python wrapper tactile_sdk";
    m.attr("__name__") = "sharpa.tactile.tactile_sdk_py";

    py::class_<DataBlock, std::shared_ptr<DataBlock>>(m, "DataBlock")
    ;

    py::class_<LoggerBase, PyLoggerBase, std::shared_ptr<LoggerBase>>(m, "LoggerBase")
    .def(py::init<>())
    .def("debug", &LoggerBase::debug)
    .def("info", &LoggerBase::info)
    .def("warn", &LoggerBase::warn)
    .def("error", &LoggerBase::error)
    ;
    py::class_<InferOffline>(m, "InferOffline")
        .def(py::init([](std::string model_path) {
            auto ret = new InferOffline(model_path);
            return ret;
        }))
        .def(
            "infer",
            [](InferOffline& self, py::array_t<uint8_t> initial_image,
               py::array_t<uint8_t> realtime_image, int size) -> py::object {
                auto frame = self.infer(initial_image.mutable_data(), realtime_image.mutable_data(), size);
                if (!frame)
                    return py::none();
                auto f = py_frame(frame);
                py::object inferred;

                // RAW_JPG and DEFORM_JPG is jpeg, DEFORM and RAW is raw
                inferred = py::dict(
                    "F6"_a = array_valid(f["F6"]) ? py::object(f["F6"])
                                                  : py::object(py::none()),
                    "DEFORM"_a = array_valid(f["DEFORM"]) ? py::object(f["DEFORM"])
                                                          : py::object(py::none()),
                    "CONTACT_POINT"_a = array_valid(f["CONTACT_POINT"])
                                            ? py::object(f["CONTACT_POINT"])
                                            : py::object(py::none()));

                return py::dict("ts"_a = frame->ts, "channel"_a = frame->channel,
                                "frame_id"_a = frame->frame_id, "content"_a = inferred);
            },
            "initial_image"_a, "realtime_image"_a, "size"_a);

    py::class_<TouchSetting>(m, "TouchSetting")
    .def(py::init<std::map<std::string, std::vector<int>>
        ,int, int, int, int, int, bool, bool, bool, bool>()
        , "model_path"_a = std::map<std::string, std::vector<int>>{}
        , "allowed_pack_loss"_a = 0
        , "buffer_size"_a = 2
        , "batch_size"_a = 1
        , "num_worker"_a = 3
        , "fps"_a = 30
        , "infer_from_device"_a = true
        , "require_jpeg"_a = true
        , "decode_jpeg"_a = true
        , "update_firmware"_a = false
    )
    .def_readwrite("model_path", &TouchSetting::model_path)
    .def_readwrite("allowed_pack_loss", &TouchSetting::allowed_pack_loss)
    .def_readwrite("buffer_size", &TouchSetting::buffer_size)
    .def_readwrite("batch_size", &TouchSetting::batch_size)
    .def_readwrite("num_worker", &TouchSetting::num_worker)
    .def_readwrite("fps", &TouchSetting::fps)
    .def_readwrite("infer_from_device", &TouchSetting::infer_from_device)
    .def_readwrite("require_jpeg", &TouchSetting::require_jpeg)
    .def_readwrite("decode_jpeg", &TouchSetting::decode_jpeg)
    .def_readwrite("update_firmware", &TouchSetting::update_firmware)
    .def("__repr__", [](const TouchSetting &ts) {
        return "<TouchSetting: "
               "allowed_pack_loss=" + std::to_string(ts.allowed_pack_loss) + 
               ", buffer_size=" + std::to_string(ts.buffer_size) +
               ", batch_size=" + std::to_string(ts.batch_size) + 
               ", fps=" + std::to_string(ts.fps) + ">";
    })
    ;

    py::class_<Touch>(m, "Touch")
    .def(py::init
    ([](
        std::string host_ip
        , int host_port
        , std::optional<std::vector<int>> channels
        , std::vector<std::string> board_ip
        , std::optional<std::function<void(py::object)>> cb
        , std::optional<TouchSetting> setting
        , std::shared_ptr<LoggerBase> logger
    ){
        auto ret = new Touch(host_ip, host_port, channels, board_ip, setting, logger);
        if(cb.has_value()) {
            ret->set_callback([cb](Frame::Ptr frame){
                py::gil_scoped_acquire acquire;
                try {
                    auto f = py_frame(frame);
                    py::object inferred;
                    //if(!array_valid(f["F6"]) && !array_valid(f["DEFORM"]) && array_valid(f["RAW"])) inferred = py::none{};
                    inferred = py::dict(
                        "F6"_a = array_valid(f["F6"])? py::object(f["F6"]) : py::object(py::none()),
                        "DEFORM"_a = array_valid(f["DEFORM"])? py::object(f["DEFORM"]) : py::object(py::none()),
                        "RAW"_a = array_valid(f["RAW"])? py::object(f["RAW"]) : py::object(py::none()),
                        "DEFORM_JPG"_a= array_valid(f["DEFORM_JPG"])? py::object(f["DEFORM_JPG"]) : py::object(py::none()),
                        "RAW_JPG"_a = array_valid(f["RAW_JPG"])? py::object(f["RAW_JPG"]) : py::object(py::none()),
                        "CONTACT_POINT"_a = array_valid(f["CONTACT_POINT"])? py::object(f["CONTACT_POINT"]) : py::object(py::none())
                    );

                    (*cb)(
                        py::dict(
                            "ts"_a=frame->ts,
                            "channel"_a=frame->channel,
                            "frame_id"_a=frame->frame_id,
                            "content"_a=inferred
                        )
                    );
                } catch(const std::exception &e) {
                    std::cout << "py error: " << e.what() << "\n";
                }
            });
        }
        return ret;
    }), "host_ip"_a, "host_port"_a
    , "channels"_a = std::nullopt
    , "board_ip"_a = nullptr
    , "callback"_a = std::nullopt
    , "setting"_a = std::nullopt
    , "logger"_a = nullptr)
    /* methods */
    // .def("is_ready", &Touch::is_ready, "board_ip"_a)
    .def("start", &Touch::start)
    .def("stop", [](Touch &self) {
        py::gil_scoped_release release;
        return self.stop();
    })
    .def("is_sdk_running",&Touch::is_sdk_running)
    .def("fetch", [](Touch &self, int channel, double timeout) -> py::object {
        auto frame = self.fetch(channel, timeout);
        if(!frame) return py::none();
        auto f = py_frame(frame);
        py::object inferred;

        //RAW_JPG and DEFORM_JPG is jpeg, DEFORM and RAW is raw 
        inferred = py::dict(
            "F6"_a = array_valid(f["F6"])? py::object(f["F6"]) : py::object(py::none()),
            "DEFORM"_a = array_valid(f["DEFORM"])? py::object(f["DEFORM"]) : py::object(py::none()),
            "RAW"_a = array_valid(f["RAW"])? py::object(f["RAW"]) : py::object(py::none()),
            "DEFORM_JPG"_a= array_valid(f["DEFORM_JPG"])? py::object(f["DEFORM_JPG"]) : py::object(py::none()),
            "RAW_JPG"_a = array_valid(f["RAW_JPG"])? py::object(f["RAW_JPG"]) : py::object(py::none()),
            "CONTACT_POINT"_a = array_valid(f["CONTACT_POINT"])? py::object(f["CONTACT_POINT"]) : py::object(py::none())
        );

        return py::dict(
            "ts"_a=frame->ts,
            "channel"_a=frame->channel,
            "frame_id"_a=frame->frame_id,
            "content"_a=inferred
        );
    }, "channel"_a, "timeout"_a = 1.0)
    .def("summary", &Touch::summary)
    .def("calib_zero", [](Touch &self, int num_frames, int max_retry) {
        /* calib can fail due to locking of callback function */
        /* therefore, releasing lock here is essential */
        py::gil_scoped_release release;
        return self.calib_zero(num_frames, max_retry);
    }, "num_frames"_a=20, "max_retry"_a=10)
    .def("board_cfg", [](Touch& self, 
                       const std::string& ip,
                       int channel,
                       const std::string& key,
                       py::object value,
                       double timeout) {
        auto ret_cpp = self.board_cfg(ip, channel, key, py2any(value), timeout);
        // py::dict ret;
        std::map<int, py::object> ret;
        for(const auto &[channel, value] : ret_cpp) {
            ret[channel] = any2py(value);
        }
        return ret;
    }, "ip"_a, "channel"_a, "key"_a, "value"_a=py::none(), "timeout"_a=1.0)
    .def("board_update", &Touch::board_update, 
         py::arg("ip"), py::arg("firmware_path"))
    .def("deform_map_uv", [](Touch& self, 
                       int channel,
                       size_t row,
                       size_t col) ->py::object {
        auto ret_cpp = self.deform_map_uv(channel, row, col);

        if(ret_cpp.has_value()){
            py::array_t<float> arr(6);
            auto arr_data = arr.mutable_data();
            std::memcpy(arr_data, ret_cpp.value().data(), 6 * sizeof(float));
            return arr;
        }
        return py::none();
    },"channel"_a, "row"_a, "col"_a)
    .def("deform_map_value", &Touch::deform_map_value, 
         py::arg("value_ui8"));
}

}
}
