// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "async_infer.hpp"

#include "errors.hpp"
#include "infer_request.hpp"

void asyncInfer(const Napi::CallbackInfo& info) {
    if (info.Length() != 3)
        reportError(info.Env(), "asyncInfer method takes three arguments.");

    auto ir = Napi::ObjectWrap<InferRequestWrap>::Unwrap(info[0].ToObject());
    if (info[1].IsArray()) {
        ir->infer(info[1].As<Napi::Array>());
    } else if (info[1].IsObject()) {
        ir->infer(info[1].As<Napi::Object>());
    } else {
        reportError(info.Env(), "asyncInfer method takes as a second argument an array or an object.");
    }

    Napi::Function cb = info[2].As<Napi::Function>();
    cb.Call(info.Env().Global(), {info.Env().Null(), ir->get_output_tensors(info)});
}

#include <thread>
using namespace Napi;
std::thread nativeThread;
ThreadSafeFunction tsfn;

Napi::Value asyncInferTSFN(const CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 3) {
        throw TypeError::New(env, "Expected two arguments");
    } else if (!info[2].IsFunction()) {
        throw TypeError::New(env, "Expected third arg to be function");
    } else if (!info[1].IsArray() || !info[1].IsObject()) {  // check info[1]
        reportError(info.Env(), "asyncInfer method takes as a second argument an array or an object.");
    }

    // InferRequestWrap * ir
    auto ir = Napi::ObjectWrap<InferRequestWrap>::Unwrap(info[0].ToObject());
    Napi::Value inputs = info[1].As<Napi::Value>();

    // Create a ThreadSafeFunction
    tsfn = ThreadSafeFunction::New(env,
                                   info[2].As<Function>(),  // JavaScript function called asynchronously
                                   "asyncInfer",            // Name
                                   0,                       // Unlimited queue
                                   1                        // Only one thread will use this initially
    );

    auto callback = [](Napi::Env env, Function jsCallback, InferRequestWrap* value) {
        // Transform native data into JS data, passing it to the provided
        // `jsCallback` -- the TSFN's JavaScript function.

        Napi::Object res = value->get_output_tensors(env);
        jsCallback.Call({env.Null(), res});
        // delete value;
    };

    if (inputs.IsArray()) {
        ir->infer(inputs.As<Napi::Array>());
    } else if (inputs.IsObject()) {
        ir->infer(inputs.As<Napi::Object>());

    } else {
        reportError(env, "asyncInfer method takes as a second argument an array or an object.");
    }

    // here a copy of InferRequest ir, and create a pointer to it
    // that will be deleted after TSFN is finished
    auto res = ir->get_output_tensors(env).As<Napi::Value>();
    auto res_ptr = &res;

    // Perform a blocking call
    napi_status status = tsfn.BlockingCall(ir, callback);

    // Release the thread-safe function
    tsfn.Release();

    return Boolean::New(env, true);
}
