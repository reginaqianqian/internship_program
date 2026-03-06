#include "infer_offline.h"
#include "engine_base.h"
#include "infer.h"
#include "nms.h"
#include "util.h"

#include <fmt/core.h>
#include <Eigen/Dense>
#include <algorithm>
#include <cstring>

#if (INFER_ENGINE == 2)
#include "coreml.h"
#elif (INFER_ENGINE == 1)
#include "trt.h"
#else
#include "dummy_engine.h"
#endif

namespace sharpa {
namespace tactile {

struct InferOffline::Impl {
    std::string model_path_;
    std::shared_ptr<EngineBase> engine_;
    const size_t kImgHeight_{240}, kImgWidth_{320}, kImgSize_{kImgHeight_ * kImgWidth_};
    const size_t kDeformHeight_{240}, kDeformWidth_{240};
    const size_t kF6Size_{6};
    const size_t kGridHeight_{15}, kGridWidth_{15},
        kContactPointSize_{kGridHeight_ * kGridWidth_ * 3};
    float* nms_ret_;

    Impl(std::string model_path) : model_path_(model_path) {
        LoggerSingleton::getInstance().setLogger(std::make_shared<LoggerDefault>());
#if (INFER_ENGINE == 2)
        engine_ = std::make_shared<CoremlInference>(model_path_, 1);
#elif (INFER_ENGINE == 1)
        engine_ = std::make_shared<TensorRTInference>(model_path_, 1);
#else
        engine_ = std::make_shared<DummyEngine>();
#endif

        nms_init(static_cast<int>(kGridWidth_), static_cast<int>(kGridHeight_),
                 static_cast<int>(kDeformHeight_), static_cast<int>(kDeformWidth_));
        nms_ret_ = new float[3 * kGridHeight_ * kGridWidth_];
    }

    ~Impl() { delete[] nms_ret_; }

    Frame::Ptr infer(uint8_t* initial_image, uint8_t* realtime_image, int size) {
        assert_throw(size == kImgSize_,
                     "size of input should be " + std::to_string(kImgSize_));
        Frame::Ptr frame = std::make_shared<Frame>();
        float* to_infer = new float[2 * kImgSize_];
        float *ref_ptr(to_infer), *rt_ptr(to_infer + kImgSize_);
        uint8_to_float(initial_image, ref_ptr, size);
        uint8_to_float(realtime_image, rt_ptr, size);

        // one pair of realtime and ref image once
        auto ret = engine_->infer(to_infer, kImgSize_);
        delete[] to_infer;
        if (ret.empty())
            return nullptr; /* engine error, return with no infered content */

        assert_throw(ret.size() == 3, "panic"); /* f6 + deform + contact point + status*/

        auto ptr_f6 = ret[0].data();
        auto ptr_deform = ret[1].data();
        auto ptr_contact_point = ret[2].data();

        // compared to f6 processing when online infering,  offline inferece generated f6
        // dose not substract the f6_offset because f6_offset is generated according to
        // the just-passed frames which are Temporally continuous and for offline
        // inference we cannot assume that the inputs are temporally continuous so we do
        // not substract the f6_offset
        auto f6 = std::make_shared<DataBlock>(Shape{{1, 1, kF6Size_}}, sizeof(float));
        memcpy(f6->data(), ptr_f6, kF6Size_ * sizeof(float));
        frame->content.emplace("F6", f6);

        auto deform = std::make_shared<DataBlock>(
            Shape{{1, kDeformHeight_, kDeformWidth_}}, sizeof(float));
        memcpy(deform->data(), ptr_deform, deform->nbytes());
        frame->content.emplace("DEFORM", deform);
        frame->content["DEFORM"] = db_f32_to_ui8(frame->content["DEFORM"]);

        auto contact_point = std::make_shared<DataBlock>(
            Shape{{kGridHeight_, kGridWidth_, 3}}, sizeof(float));
        memcpy(contact_point->data(), ptr_contact_point, contact_point->nbytes());
        frame->content.emplace("CONTACT_POINT", contact_point);
        int ret_size =
            nms_execute((float*)(frame->content["CONTACT_POINT"]->data()), nms_ret_);
        contact_point = std::make_shared<DataBlock>(
            ret_size > 0 ? Shape{{static_cast<size_t>(ret_size), 3}} : Shape{},
            sizeof(float));
        memcpy(contact_point->data(), nms_ret_, contact_point->nbytes());
        frame->content["CONTACT_POINT"] = contact_point;

        frame->ts = 0.0f;
        frame->channel = -1;
        frame->frame_id = 666;

        return frame;
    }
};

InferOffline::InferOffline(std::string model_path)
    : impl_(std::make_shared<Impl>(model_path)) {}
InferOffline::~InferOffline() = default;
Frame::Ptr InferOffline::infer(uint8_t* initial_image,
                               uint8_t* realtime_image,
                               int size) {
    return impl_->infer(initial_image, realtime_image, size);
}

}  // namespace tactile
}  // namespace sharpa
