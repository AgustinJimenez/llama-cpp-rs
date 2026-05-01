// Safe C++ wrappers that catch exceptions from llama.cpp functions.
// Prevents C++ exceptions from propagating through the Rust FFI boundary.

#include "llama.h"
#include <cstring>
#include <stdexcept>
#include <thread>
#include <atomic>
#include <chrono>

static thread_local char g_last_error[1024] = {0};

extern "C" {

// Safe llama_decode: returns -99 on C++ exception
int32_t llama_decode_safe(struct llama_context * ctx, struct llama_batch batch) {
    g_last_error[0] = '\0';
    try {
        return llama_decode(ctx, batch);
    } catch (const std::exception & e) {
        strncpy(g_last_error, e.what(), sizeof(g_last_error) - 1);
        g_last_error[sizeof(g_last_error) - 1] = '\0';
        return -99;
    } catch (...) {
        strncpy(g_last_error, "unknown C++ exception in llama_decode", sizeof(g_last_error) - 1);
        return -99;
    }
}

// Safe llama_sampler_sample with timeout.
// Runs sample() on a detached thread. If it doesn't return within
// timeout_ms, returns LLAMA_TOKEN_NULL and sets error message.
// The hung thread is abandoned (detached) — it will eventually be
// cleaned up when the process exits or the CUDA operation completes.
llama_token llama_sampler_sample_safe(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx) {
    g_last_error[0] = '\0';

    std::atomic<bool> done{false};
    std::atomic<llama_token> result{-1};
    std::atomic<bool> had_exception{false};

    // Run sample on a separate thread so we can timeout
    std::thread worker([&]() {
        try {
            llama_token token = llama_sampler_sample(smpl, ctx, idx);
            result.store(token, std::memory_order_release);
        } catch (const std::exception & e) {
            strncpy(g_last_error, e.what(), sizeof(g_last_error) - 1);
            g_last_error[sizeof(g_last_error) - 1] = '\0';
            had_exception.store(true, std::memory_order_release);
        } catch (...) {
            strncpy(g_last_error, "unknown C++ exception in llama_sampler_sample", sizeof(g_last_error) - 1);
            had_exception.store(true, std::memory_order_release);
        }
        done.store(true, std::memory_order_release);
    });

    // Wait up to 8 seconds for sample to complete
    const int timeout_ms = 8000;
    const int poll_ms = 10;
    int elapsed = 0;
    while (!done.load(std::memory_order_acquire) && elapsed < timeout_ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
        elapsed += poll_ms;
    }

    if (done.load(std::memory_order_acquire)) {
        worker.join();
        if (had_exception.load(std::memory_order_acquire)) {
            return -1;
        }
        return result.load(std::memory_order_acquire);
    } else {
        // Timeout — sample() is stuck in CUDA synchronize.
        // Detach the thread (it will be cleaned up on process exit).
        worker.detach();
        strncpy(g_last_error, "sample() timed out (CUDA sync deadlock)", sizeof(g_last_error) - 1);
        fprintf(stderr, "[SAMPLE_SAFE] Timeout after %dms — CUDA sync deadlock detected\n", timeout_ms);
        return -1;
    }
}

const char * llama_decode_safe_get_error(void) {
    return g_last_error;
}

}
