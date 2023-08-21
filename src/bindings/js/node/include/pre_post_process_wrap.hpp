// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief This is a header file for the NAPI POC PrePostProcessorWrap
 * @file src/PrePostProcessorWrap.hpp
 */
#pragma once

#include <napi.h>

#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/openvino.hpp>

#include "element_type.hpp"
#include "errors.hpp"
#include "helper.hpp"
#include "model_wrap.hpp"

class PrePostProcessorWrap : public Napi::ObjectWrap<PrePostProcessorWrap> {
public:
    /**
     * @brief Constructs PrePostProcessorWrap class from the Napi::CallbackInfo.
     * @param info //TO DO
     */
    PrePostProcessorWrap(const Napi::CallbackInfo& info);
    /**
     * @brief Defines a Javascript PrePostProcessor class with constructor, static and instance properties and methods.
     * @param env The environment in which to construct a JavaScript class.
     * @return Napi::Function representing the constructor function for the Javascript PrePostProcessor class.
     */
    static Napi::Function GetClassConstructor(Napi::Env env);
    /** @brief This method is called during initialization of OpenVino native add-on.
     * It exports JavaScript PrePostProcessor class.
     */
    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    Napi::Value set_input_tensor_shape(const Napi::CallbackInfo& info);
    /**
     * @brief Apply resize algorithm
     * @param info[0] Napi::String algorithm name: "RESIZE_CUBIC", "RESIZE_NEAREST" or "RESIZE_LINEAR" (default)
     */
    Napi::Value preprocess_resize_input(const Napi::CallbackInfo& info);
    Napi::Value set_input_tensor_layout(const Napi::CallbackInfo& info);
    Napi::Value set_input_model_layout(const Napi::CallbackInfo& info);
    /**
     * @brief Sets type of element for specified input
     * @param info[0] index of the input
     * @param info[1] element_type
     */
    Napi::Value set_input_element_type(const Napi::CallbackInfo& info);
    Napi::Value build(const Napi::CallbackInfo& info);

private:
    std::unique_ptr<ov::preprocess::PrePostProcessor> _ppp;
};
