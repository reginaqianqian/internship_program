
#include "util.h"

#include <Eigen/Dense>
#include <chrono>
#include <stdexcept>
#include "def.h"

namespace sharpa {
namespace tactile {

ThreadPool::ThreadPool(size_t threads) {
    /* the constructor just launches some amount of workers */
    for(size_t i = 0;i<threads;++i) workers.emplace_back( [this] { for(;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            this->condition.wait(
                lock, [this] { return this->stop_flag || !this->tasks.empty(); });
            if (this->stop_flag && this->tasks.empty())
                return;
            task = std::move(this->tasks.front());
            this->tasks.pop();
        }
        task();
        // cond_wait.notify_all(); 
    }});
}

ThreadPool::~ThreadPool() {
    stop();
}

void ThreadPool::stop() {
    if (stop_flag)
        return;
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop_flag = true;
        condition.notify_all();
    }
    for (std::thread& worker : workers)
        worker.join();
}

double ts_unix() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

void assert_throw(bool cond, const std::string &error_msg) {
    if(!cond) throw std::runtime_error(error_msg);
}

LoggerSingleton& LoggerSingleton::getInstance() {
    static LoggerSingleton instance;
    return instance;
}

// 设置外部 logger
void LoggerSingleton::setLogger(std::shared_ptr<LoggerBase> logger) {
    // assert_throw(!logger_,
    //              "Logger already set! setLogger() should be called once at init
    //              time.");

    // for multi-thread scenario, it is safe to use mutex.
    std::lock_guard<std::mutex> lock(mtx);
    if (logger && !logger_) {
        logger_ = logger;
    }
}

// 获取当前 logger（优先使用外部设置的，否则返回默认）
const std::shared_ptr<LoggerBase>& LoggerSingleton::getLogger() {
    // if (!logger_) {
    //     // 线程安全：静态局部变量
    //     static auto default_logger = std::make_shared<LoggerDefault>();
    //     logger_ = default_logger;
    // }
    return logger_;
}

void uint8_to_float(const uint8_t* src, float* dst, size_t len) {
    Eigen::Map<const Eigen::Array<uint8_t, Eigen::Dynamic, 1>> srcVec(src, len);
    Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>> dstVec(dst, len);
    dstVec = srcVec.cast<float>();
    // dstVec = (dstVec / 255.f);
}
}
}
