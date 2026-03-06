
#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fmt/format.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <unistd.h>
#include "def.h"

namespace sharpa {
namespace tactile {

template <typename T>
class SafeQueue {
public:
    explicit SafeQueue(size_t buffer_size) : max_size(buffer_size) {
        if (buffer_size == 0) {
            throw std::invalid_argument("Buffer size must be greater than zero");
        }
    }

    ~SafeQueue() {
        if (!is_stop_)
            stop();
    }

    void enqueue(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        /* remove oldest element if queue is full */
        if (queue_.size() >= max_size) {
            queue_.pop();
        }
        queue_.push(std::move(item));
        lock.unlock();
        cond_.notify_one();
    }

    std::optional<T> dequeue(double timeout = -1.) {
        std::unique_lock<std::mutex> lock(mutex_);
        /* infinite wait */
        if (timeout < 0)
            cond_.wait(lock, [this] { return !queue_.empty() || is_stop_; });
        /* non-blocking */
        else if (timeout == 0) {
            if (queue_.empty())
                return std::nullopt;
        } else {
            if (!cond_.wait_for(lock, std::chrono::nanoseconds(uint64_t(timeout * 1e9)),
                                [this] { return !queue_.empty() || is_stop_; }))
                return std::nullopt;
        }
        auto item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    size_t size() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size();
    }

    void stop() {
        std::unique_lock<std::mutex> lock(mutex_);
        is_stop_ = true;
        cond_.notify_all();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cond_;
    std::queue<T> queue_;
    size_t max_size;
    bool is_stop_{false};
};

class ThreadPool {
public:
    explicit ThreadPool(size_t threads);
    ~ThreadPool();
    void stop();

    template <class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            /* don't allow enqueueing after stopping the pool */
            if (stop_flag)
                throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([this, task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

private:
    // need to keep track of threads, so we can join them
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;  // the task queue
    std::mutex queue_mutex;                   // synchronization
    std::condition_variable condition;
    // std::condition_variable cond_wait;
    std::atomic<bool> stop_flag{false};
};

class LoggerDefault : public LoggerBase {
public:
    LoggerDefault() {
        spdlog::drop("sharpa");
#ifndef NDEBUG
        spdlog::set_level(spdlog::level::debug);
#endif
        logger_ = spdlog::stdout_color_mt("sharpa");
        logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [tactile] [%^%l%$] %v");
    }
    void debug(const std::string& msg) override {
        logger_->debug(msg);
    }
    void info(const std::string& msg) override {
        logger_->info(msg);
    }
    void warn(const std::string& msg) override {
        logger_->warn(msg);
    }
    void error(const std::string& msg) override {
        logger_->error(msg);
    }

private:
    std::shared_ptr<spdlog::logger> logger_;
};

class LoggerSingleton {
public:
    // 获取单例实例（引用）
    static LoggerSingleton& getInstance();

    // 设置外部 logger
    void setLogger(std::shared_ptr<LoggerBase> logger);

    // 获取当前 logger（优先使用外部设置的，否则返回默认）
    const std::shared_ptr<LoggerBase>& getLogger();

private:
    LoggerSingleton() = default;   // 私有构造
    ~LoggerSingleton() = default;  // 析构
    LoggerSingleton(const LoggerSingleton&) = delete;
    LoggerSingleton& operator=(const LoggerSingleton&) = delete;

    std::shared_ptr<LoggerBase> logger_{};  // 持有的 logger
    std::mutex mtx;
};

// 定义日志宏

#define LOG_DEBUG(...)                                            \
    do {                                                          \
        auto logger = LoggerSingleton::getInstance().getLogger(); \
        logger->debug(fmt::format(__VA_ARGS__));                  \
    } while (0)

// 原始字符串输出
#define LOG_DEBUG_RAW(msg)                                        \
    do {                                                          \
        auto logger = LoggerSingleton::getInstance().getLogger(); \
        logger->debug(msg);                                       \
    } while (0)

#define LOG_INFO(...)                                             \
    do {                                                          \
        auto logger = LoggerSingleton::getInstance().getLogger(); \
        logger->info(fmt::format(__VA_ARGS__));                   \
    } while (0)

// 原始字符串输出
#define LOG_INFO_RAW(msg)                                         \
    do {                                                          \
        auto logger = LoggerSingleton::getInstance().getLogger(); \
        logger->info(msg);                                        \
    } while (0)

#define LOG_WARN(...)                                             \
    do {                                                          \
        auto logger = LoggerSingleton::getInstance().getLogger(); \
        logger->warn(fmt::format(__VA_ARGS__));                   \
    } while (0)

// 原始字符串输出
#define LOG_WARN_RAW(msg)                                         \
    do {                                                          \
        auto logger = LoggerSingleton::getInstance().getLogger(); \
        logger->warn(msg);                                        \
    } while (0)

#define LOG_ERROR(...)                                            \
    do {                                                          \
        auto logger = LoggerSingleton::getInstance().getLogger(); \
        logger->error(fmt::format(__VA_ARGS__));                  \
    } while (0)

// 原始字符串输出
#define LOG_ERROR_RAW(msg)                                        \
    do {                                                          \
        auto logger = LoggerSingleton::getInstance().getLogger(); \
        logger->error(msg);                                       \
    } while (0)

double ts_unix();

void assert_throw(bool cond, const std::string& error_msg);

void uint8_to_float(const uint8_t* src, float* dst, size_t len);

}  // namespace tactile
}  // namespace sharpa
