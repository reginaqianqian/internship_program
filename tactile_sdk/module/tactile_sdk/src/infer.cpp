#include "infer.h"
#ifdef USE_OPENCV
#include "drift_detect.h"
#endif
#include "nms.h"

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

struct CalibStat {
    CalibStat(std::vector<int>& channels,
              size_t kF6Size,
              size_t kDeformHeight,
              size_t kDeformWidth,
              size_t kImgHeight,
              size_t kImgWidth)
        : channels_(std::move(channels)),
          kF6Size_(kF6Size),
          kDeformHeight_(kDeformHeight),
          kDeformWidth_(kDeformWidth),
          kImgHeight_(kImgHeight),
          kImgWidth_(kImgWidth) {
        for (auto ch : channels_) {
            calib_num_[ch] = 0;
            offset_f6_[ch] =
                std::make_shared<DataBlock>(Shape{{1, 1, kF6Size_}}, sizeof(float));
            offset_deform_[ch] = std::make_shared<DataBlock>(
                Shape{{1, kDeformHeight_, kDeformWidth_}}, sizeof(float));
            offset_f6_[ch]->set_zero();
            offset_deform_[ch]->set_zero();
            ref_img_[ch] = nullptr;
            calib_num_mutex_[ch];
            offset_mutex_[ch];
        }
        calib_iter_ = 0;
    }

    void to_calib(int ch) {
        std::lock_guard<std::mutex> _(calib_num_mutex_[ch]);
        if (calib_num_[ch] != 0) {
            LOG_WARN("Channel {} calibration failed due to conflict", ch);
            return;
        }
        calib_num_[ch] = 1;
    }

    void get_ref_image(int ch, float* ref_ptr) {
        memcpy(ref_ptr, ref_img_[ch]->data(), ref_img_[ch]->nbytes());
    }

    void check_ref_image(int ch, DataBlock::Ptr raw) {
        // set ref_img if ref_img is null
        if (!ref_img_[ch]) {
            set_ref_image(ch, raw);
            return;
        }
        // set ref_img if calib
        {
            std::lock_guard<std::mutex> _(calib_num_mutex_[ch]);
            if (calib_num_[ch] != 1) {
                return;
            }
            calib_num_[ch] = 2;
        }
        set_ref_image(ch, raw);
    }

    void set_ref_image(int ch, DataBlock::Ptr raw) {
        std::lock_guard<std::mutex> _(calib_num_mutex_[ch]);
        if (!ref_img_[ch]) {
            ref_img_[ch] = std::make_shared<DataBlock>(
                Shape{{1, kImgHeight_, kImgWidth_}}, sizeof(float));
        }
        uint8_to_float((uint8_t*)raw->data(), (float*)ref_img_[ch]->data(), raw->size());
    }

    void check_offset(Frame::Ptr f) {
        int ch = f->channel;
        {
            std::lock_guard<std::mutex> _(calib_num_mutex_[ch]);
            if (calib_num_[ch] != 2) {
                return;
            }
            calib_num_[ch] = 0;
        }
        {
            std::lock_guard<std::mutex> _(offset_mutex_[ch]);
            offset_f6_[ch]->set_zero();
            offset_f6_[ch] = add_db_float(offset_f6_[ch], f->content["F6"]);
            offset_deform_[ch]->set_zero();
            offset_deform_[ch] = add_db_float(offset_deform_[ch], f->content["DEFORM"]);
        }
    }

    void apply_offset(Frame::Ptr f) {
        std::lock_guard<std::mutex> _(offset_mutex_[f->channel]);
        if (f->content.find("F6") != f->content.end()) {
            f->content["F6"] = sub_db_float(f->content["F6"], offset_f6_[f->channel]);
        }
        if (f->content.find("DEFORM") != f->content.end()) {
            f->content["DEFORM"] = db_f32_to_ui8(
                sub_db_float(f->content["DEFORM"], offset_deform_[f->channel]));
        }
    }

    void check_calib_finished() {
        {
            std::lock_guard<std::mutex> _(calib_iter_mutex_);
            if (calib_iter_ == 0) {
                return;
            } else if (calib_iter_ == 1) {
                for (auto ch : channels_) {
                    std::lock_guard<std::mutex> _(calib_num_mutex_[ch]);
                    if (calib_num_[ch] != 0) {
                        LOG_WARN("Channel {} calibration failed", ch);
                        calib_num_[ch] = 0;
                    }
                }
                calib_success_ = false;
                calib_cv_.notify_all();
                return;
            } else {
                --calib_iter_;
            }
        }
        for (auto ch : channels_) {
            std::lock_guard<std::mutex> _(calib_num_mutex_[ch]);
            if (calib_num_[ch] != 0) {
                return;
            }
        }
        {
            std::lock_guard<std::mutex> _(calib_iter_mutex_);
            calib_iter_ = 1;
            calib_success_ = true;
            calib_cv_.notify_all();
        }
    }

    bool calib_zero(int calib_iter) {
        // calib_iter must > 0
        if (calib_iter <= 0) {
            LOG_WARN("Invalid calib_iter");
            return false;
        }
        // must wait until last calib_zero finished
        {
            std::lock_guard<std::mutex> _(calib_iter_mutex_);
            if (calib_iter_ != 0) {
                LOG_WARN("Calibration failed due to conflict");
                return false;
            }
            calib_iter_ = calib_iter;
        }
        for (auto ch : channels_) {
            to_calib(ch);
        }
        {
            std::unique_lock<std::mutex> ul(calib_iter_mutex_);
            calib_cv_.wait(ul, [this]() { return calib_iter_ == 1; });
            calib_iter_ = 0;
        }
        for (auto ch : channels_) {
            std::lock_guard<std::mutex> _(calib_num_mutex_[ch]);
            calib_num_[ch] = 0;
        }
        return calib_success_;
    }

private:
    std::vector<int> channels_;
    std::map<int, int> calib_num_;  // 0 normal, 1 to update ref_image, 2 to update offset
    std::map<int, std::shared_ptr<DataBlock>> offset_f6_;
    std::map<int, std::shared_ptr<DataBlock>> offset_deform_;
    std::map<int, std::shared_ptr<DataBlock>> ref_img_;
    std::unordered_map<int, std::mutex> offset_mutex_;
    std::unordered_map<int, std::mutex> calib_num_mutex_;
    std::mutex calib_iter_mutex_;
    size_t kF6Size_;
    size_t kDeformHeight_;
    size_t kDeformWidth_;
    size_t kImgHeight_;
    size_t kImgWidth_;
    bool calib_success_;
    int calib_iter_;  // 0 normal, 1 exceed limit & calib failed, >1 trying to set ref_img
                      // for all channel

    std::condition_variable calib_cv_;
};

struct InferEngine::Impl {
    std::map<std::string, std::vector<int>> model_path_;
    std::vector<int> channels_;
    int batch_size_;
    int buffer_size_;
    SafeQueue<Frame::Ptr> queue_in_;
    std::map<int, SafeQueue<Frame::Ptr>> queue_out_;
    const std::function<void(Frame::Ptr)>* callback_{nullptr};
    bool infer_from_device_;
    std::map<std::string, std::shared_ptr<EngineBase>> engine_;
    std::thread th_;
    bool is_running_{false};
    bool is_quit_{false};
    mutable std::mutex lock_;
    mutable std::shared_mutex lock_qout_;
    const size_t kImgHeight_{240}, kImgWidth_{320}, kImgSize_{kImgHeight_ * kImgWidth_};
    const size_t kDeformHeight_{240}, kDeformWidth_{240};
    const size_t kF6Size_{6};
    const size_t kGridHeight_{15}, kGridWidth_{15},
        kContactPointSize_{kGridHeight_ * kGridWidth_ * 3};

    std::shared_ptr<CalibStat> calib_stat_;
    float* nms_ret_;

#ifdef USE_OPENCV
    std::unique_ptr<drift_detect> drift_{nullptr};
#endif
    std::shared_ptr<ThreadPool> pool_;

    Impl(std::map<std::string, std::vector<int>> model_path,
         std::vector<int> channels,
         int batch_size,
         int buffer_size,
         const std::function<void(Frame::Ptr)>* callback,
         std::shared_ptr<ThreadPool> pool,
         bool infer_from_device)
        : model_path_(std::move(model_path)),
          channels_(std::move(channels)),
          batch_size_(batch_size),
          buffer_size_(buffer_size),
          queue_in_(SafeQueue<Frame::Ptr>(buffer_size * batch_size)),
          callback_(callback),
          pool_(pool),
          infer_from_device_(infer_from_device) {
        for (const auto& [path, _] : model_path_) {
#if (INFER_ENGINE == 2)
            auto engine = std::make_shared<CoremlInference>(path, batch_size);
#elif (INFER_ENGINE == 1)
            auto engine = std::make_shared<TensorRTInference>(path, batch_size);
#else
            auto engine = std::make_shared<DummyEngine>();
#endif
            engine_.emplace(path, engine);
        }
        nms_init(static_cast<int>(kGridWidth_), static_cast<int>(kGridHeight_),
                 static_cast<int>(kDeformHeight_), static_cast<int>(kDeformWidth_));
        nms_ret_ = new float[3 * kGridHeight_ * kGridWidth_];
        calib_stat_ = std::make_shared<CalibStat>(channels_, kF6Size_, kDeformHeight_,
                                                  kDeformWidth_, kImgHeight_, kImgWidth_);

#ifdef USE_OPENCV
        drift_ = std::make_unique<drift_detect>(
            15.0, [this](int ch) { this->calib_stat_->to_calib(ch); }, kImgHeight_,
            kImgWidth_);
#endif
    }

    ~Impl() { delete[] nms_ret_; }

    void set_thread_pool(std::shared_ptr<ThreadPool> pool) { pool_ = pool; }

    bool start() {
        if (is_running_)
            return false;
        th_ = std::thread([this] { infer_(); });

#ifdef USE_OPENCV
        if (drift_)
            drift_->start();
#endif
        is_running_ = true;
        return true;
    }

    bool stop() {
        if (!is_running_)
            return false;
        {
            std::unique_lock<std::mutex> _(lock_);
            is_quit_ = true;
        }
        th_.join();
#ifdef USE_OPENCV
        if (drift_)
            drift_->stop();
#endif
        is_quit_ = false;
        is_running_ = false;
        return true;
    }

    bool is_running() { return is_running_; }

    void enqueue(Frame::Ptr content) {
        bool is_new_channel{false};
        {
            std::shared_lock<std::shared_mutex> _(lock_qout_);
            is_new_channel = (queue_out_.find(content->channel) == queue_out_.end());
        }

        if (is_new_channel) {
            std::unique_lock<std::shared_mutex> _(lock_qout_);
            queue_out_.emplace(content->channel, unsigned(buffer_size_));
        }
        queue_in_.enqueue(content);
    }

    Frame::Ptr dequeue(int ch, double timeout) {
        std::optional<Frame::Ptr> ret;
        {
            std::shared_lock _(lock_qout_);
            if (queue_out_.find(ch) == queue_out_.end())
                return nullptr;
            ret = queue_out_.at(ch).dequeue(timeout);
        }
        if (!ret.has_value())
            return nullptr;
        return *ret;
    }

    void infer_() {
        while (!is_quit()) {
            // if(queue_in_.size() < batch_size_) continue;
            /* init jobs */
            std::map<std::string, std::vector<Frame::Ptr>> jobs;
            for (const auto& [path, _] : model_path_)
                jobs.emplace(path, std::vector<Frame::Ptr>{});
            /* organize frames-to-infer into jobs */
            int num_frames{0};
            while (num_frames < batch_size_ && !is_quit()) {
                auto frame_opt = queue_in_.dequeue(0.1); /* set timeout to allow quit */
                if (!frame_opt.has_value())
                    continue;
                auto frame = *frame_opt;
                if (!frame->is_to_infer()) {
                    add_output_job_(frame);
                    continue;
                }
                /* based on channel, add content into jobs */
                bool job_added{false};
                for (const auto& [path, channels] : model_path_) {
                    if (std::any_of(channels.begin(), channels.end(),
                                    [frame](int x) { return x == frame->channel; })) {
                        jobs[path].push_back(frame);
                        ++num_frames;
                        job_added = true;
                        break;
                    }
                }
                if (!job_added)
                    add_output_job_(frame);
            }

            /* batch infer */
            for (auto& [path, frames] : jobs) {
                if (frames.empty())
                    continue;
                auto engine = engine_[path];
                float* to_infer = new float[frames.size() * 2 * kImgSize_];
                float *ref_ptr(to_infer), *rt_ptr(to_infer + frames.size() * kImgSize_);
                for (const auto& frame : frames) {
                    auto block = frame->content["RAW"];
                    assert_throw(block->size() == kImgSize_, "panic");
                    calib_stat_->check_ref_image(frame->channel, block);
                    calib_stat_->get_ref_image(frame->channel, ref_ptr);
                    ref_ptr += block->size();
                    uint8_to_float((uint8_t*)block->data(), rt_ptr, block->size());
                    rt_ptr += block->size();
                }

                auto ret = engine->infer(to_infer, frames.size() * kImgSize_);
                delete[] to_infer;
                if (ret.empty())
                    continue; /* engine error, return with no infered content */

                assert_throw(ret.size() == 3, "panic"); /* f6 + deform + contact point*/
                auto ptr_f6 = ret[0].data();
                auto ptr_deform = ret[1].data();
                auto ptr_contact_point = ret[2].data();
                for (auto& frame : frames) {
                    auto f6 = std::make_shared<DataBlock>(Shape{{1, 1, kF6Size_}},
                                                          sizeof(float));
                    memcpy(f6->data(), ptr_f6, kF6Size_ * sizeof(float));
                    frame->content.emplace("F6", f6);

                    auto deform = std::make_shared<DataBlock>(
                        Shape{{1, kDeformHeight_, kDeformWidth_}}, sizeof(float));
                    memcpy(deform->data(), ptr_deform, deform->nbytes());
                    frame->content.emplace("DEFORM", deform);

                    auto contact_point = std::make_shared<DataBlock>(
                        Shape{{kGridHeight_, kGridWidth_, 3}}, sizeof(float));
                    memcpy(contact_point->data(), ptr_contact_point,
                           contact_point->nbytes());
                    frame->content.emplace("CONTACT_POINT", contact_point);

                    calib_stat_->check_offset(frame);
                    calib_stat_->check_calib_finished();

                    ptr_f6 += kF6Size_;
                    ptr_deform += deform->size();
                    ptr_contact_point += kContactPointSize_;
                }
            }
            /* run callback and output */
            for (auto& [_, job] : jobs) {
                for (auto& frame : job) {
#ifdef USE_OPENCV
                    if (drift_)
                        drift_->auto_calib(frame->copy());
#endif
                    add_output_job_(frame);
                }
            }
        }
    }

    void add_output_job_(Frame::Ptr f) {
        pool_->enqueue([this, f] {
            {
                if (!infer_from_device_)
                    calib_stat_->apply_offset(f);
                if (f->content.find("CONTACT_POINT") != f->content.end()) {
                    int ret_size = nms_execute(
                        (float*)(f->content["CONTACT_POINT"]->data()), nms_ret_);
                    auto contact_point = std::make_shared<DataBlock>(
                        ret_size > 0 ? Shape{{static_cast<size_t>(ret_size), 3}}
                                     : Shape{},
                        sizeof(float));
                    memcpy(contact_point->data(), nms_ret_, contact_point->nbytes());
                    f->content["CONTACT_POINT"] = contact_point;
                }
            }
            if (callback_ && (*callback_))
                (*callback_)(f);
            {
                std::shared_lock _(lock_qout_);
                assert_throw(queue_out_.find(f->channel) != queue_out_.end(),
                             "bad channel");
            }
            queue_out_.at(f->channel).enqueue(f);
        });
    }

    bool is_quit() const {
        std::unique_lock<std::mutex> _(lock_);
        return is_quit_;
    }
};

InferEngine::InferEngine(std::map<std::string, std::vector<int>> model_path,
                         std::vector<int> channels,
                         int batch_size,
                         int buffer_size,
                         const std::function<void(Frame::Ptr)>* callback,
                         std::shared_ptr<ThreadPool> pool,
                         bool infer_from_device) {
    impl_ = std::make_shared<Impl>(model_path, channels, batch_size, buffer_size,
                                   callback, pool, infer_from_device);
}

void InferEngine::set_thread_pool(std::shared_ptr<ThreadPool> pool) {
    impl_->set_thread_pool(pool);
}

bool InferEngine::start() {
    return impl_->start();
}
bool InferEngine::is_running() {
    return impl_->is_running();
}
bool InferEngine::stop() {
    return impl_->stop();
}
void InferEngine::enqueue(Frame::Ptr content) {
    impl_->enqueue(content);
}
Frame::Ptr InferEngine::dequeue(int ch, double timeout) {
    return impl_->dequeue(ch, timeout);
}
void InferEngine::set_ref_image(int ch, DataBlock::Ptr raw) {
    return impl_->calib_stat_->set_ref_image(ch, raw);
}
bool InferEngine::calib_zero(int num_frames, int max_retry) {
    return impl_->calib_stat_->calib_zero(num_frames * max_retry);
}
size_t InferEngine::deform_height() const {
    return impl_->kDeformHeight_;
};
size_t InferEngine::deform_width() const {
    return impl_->kDeformWidth_;
};

}  // namespace tactile
}  // namespace sharpa