// Safe C++ wrappers that catch exceptions from llama.cpp functions.
// Prevents C++ exceptions from propagating through the Rust FFI boundary.

#include "llama.h"
#include <cstring>
#include <stdexcept>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

static thread_local char g_last_error[1024] = {0};

extern "C" {

// Safe llama_init_from_model that catches Windows SEH exceptions (e.g. ACCESS VIOLATION
// from new hybrid architecture KV cache creation in b9174).
// Returns nullptr on any exception; sets out_exception_code to the SEH code (or -1 for C++ exceptions).
// NOTE: __try/__except and C++ objects with destructors can't mix in the same function,
//       so this is a thin C-style wrapper that calls a helper.
#if defined(_WIN32)
static __declspec(noinline) struct llama_context * llama_init_from_model_inner(
    struct llama_model * model,
    struct llama_context_params params)
{
    return llama_init_from_model(model, params);
}

// Filter function to capture exception info before deciding to handle
static thread_local DWORD    g_seh_code = 0;
static thread_local void *   g_seh_addr = nullptr;

static DWORD seh_filter(EXCEPTION_POINTERS * ep, int32_t * out_code) {
    g_seh_code = ep->ExceptionRecord->ExceptionCode;
    g_seh_addr = ep->ExceptionRecord->ExceptionAddress;
    if (out_code) *out_code = (int32_t)g_seh_code;
    return EXCEPTION_EXECUTE_HANDLER;
}

struct llama_context * llama_init_from_model_safe(
    struct llama_model * model,
    struct llama_context_params params,
    int32_t * out_exception_code)
{
    if (out_exception_code) *out_exception_code = 0;
    g_last_error[0] = '\0';
    struct llama_context * ctx = nullptr;
    __try {
        ctx = llama_init_from_model_inner(model, params);
    }
    __except(seh_filter(GetExceptionInformation(), out_exception_code)) {
        snprintf(g_last_error, sizeof(g_last_error),
                 "SEH exception 0x%08lX at 0x%p in llama_init_from_model",
                 (unsigned long)g_seh_code, g_seh_addr);
        return nullptr;
    }
    return ctx;
}
#else
// Non-Windows: no SEH, just delegate to C++ try/catch
struct llama_context * llama_init_from_model_safe(
    struct llama_model * model,
    struct llama_context_params params,
    int32_t * out_exception_code)
{
    if (out_exception_code) *out_exception_code = 0;
    g_last_error[0] = '\0';
    try {
        return llama_init_from_model(model, params);
    } catch (const std::exception & e) {
        strncpy(g_last_error, e.what(), sizeof(g_last_error) - 1);
        g_last_error[sizeof(g_last_error) - 1] = '\0';
        if (out_exception_code) *out_exception_code = -1;
        return nullptr;
    } catch (...) {
        strncpy(g_last_error, "unknown exception in llama_init_from_model", sizeof(g_last_error) - 1);
        if (out_exception_code) *out_exception_code = -1;
        return nullptr;
    }
}
#endif






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

// Safe llama_sampler_sample: returns -1 on C++ exception
llama_token llama_sampler_sample_safe(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx) {
    g_last_error[0] = '\0';
    try {
        return llama_sampler_sample(smpl, ctx, idx);
    } catch (const std::exception & e) {
        strncpy(g_last_error, e.what(), sizeof(g_last_error) - 1);
        g_last_error[sizeof(g_last_error) - 1] = '\0';
        return -1;
    } catch (...) {
        strncpy(g_last_error, "unknown C++ exception in llama_sampler_sample", sizeof(g_last_error) - 1);
        return -1;
    }
}

const char * llama_decode_safe_get_error(void) {
    return g_last_error;
}

}
