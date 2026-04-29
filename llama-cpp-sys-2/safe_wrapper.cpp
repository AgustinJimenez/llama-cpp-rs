// Safe C++ wrapper that catches exceptions from llama.cpp functions.
// Prevents C++ exceptions from propagating through the Rust FFI boundary.

#include "llama.h"
#include <cstring>
#include <stdexcept>

static thread_local char g_last_error[1024] = {0};

extern "C" {

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

const char * llama_decode_safe_get_error(void) {
    return g_last_error;
}

}
