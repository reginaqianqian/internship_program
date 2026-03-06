#pragma once

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#include "content_protocol.h"
#include "engine_base.h"
#include "util.h"

namespace sharpa {
namespace tactile {

class InferEngine {
public:
    InferEngine(std::map<std::string, std::vector<int>> model_path,
                std::vector<int> channels,
                int batch_size,
                int buffer_size,
                const std::function<void(Frame::Ptr)>* callback,
                std::shared_ptr<ThreadPool> pool,
                bool infer_from_device);

    void set_thread_pool(std::shared_ptr<ThreadPool> pool);
    bool start();
    bool stop();
    bool is_running();
    void enqueue(Frame::Ptr content);
    Frame::Ptr dequeue(int ch, double timeout);
    void set_ref_image(int ch, DataBlock::Ptr raw);
    bool calib_zero(int num_frames, int max_retry);
    size_t deform_height() const;
    size_t deform_width() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace tactile
}  // namespace sharpa
