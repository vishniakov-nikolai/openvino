// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/session.h"

#include <limits.h>

#include <iostream>

#include "../include/shape_lite.h"
#include "openvino/openvino.hpp"

std::shared_ptr<ov::Model> loadModel(std::string xml_path, std::string bin_path) {
    ov::Core core;

    try {
        return core.read_model(xml_path, bin_path);
    } catch (const std::exception& e) {
        std::cout << "== Error in load_model: " << e.what() << std::endl;
        throw e;
    }
}

ov::CompiledModel compileModel(std::shared_ptr<ov::Model> model) {
    ov::Core core;
    std::cout << "== Model name: " << model->get_friendly_name() << std::endl;

    ov::CompiledModel compiled_model;
    const std::string backend = "TEMPLATE";
    try {
        compiled_model = core.compile_model(model, backend);
    } catch (const std::exception& e) {
        std::cout << "== Error in compile_model: " << e.what() << std::endl;
        throw e;
    }

    return compiled_model;
}

ov::Tensor performInference(ov::CompiledModel cm, ov::Tensor t) {
    ov::InferRequest infer_request = cm.create_infer_request();
    infer_request.set_input_tensor(t);
    infer_request.infer();

    return infer_request.get_output_tensor();
}

Session::Session(std::string xml_path, std::string bin_path, ShapeLite* shape, std::string layout, std::string input_type) {
    auto model = loadModel(xml_path, bin_path);
    try {
        this->model = compileModel(model);
    } catch (const std::exception& e) {
        std::cout << "== Error in Session constructor: " << e.what() << std::endl;
        throw e;
    }
}

TensorLite Session::infer(TensorLite* tensor_lite) {
    std::cout << "== Run inference" << std::endl;
    ov::Tensor output_tensor;

    try {
        output_tensor = performInference(this->model, *tensor_lite->get_tensor());
    } catch (const std::exception& e) {
        std::cout << "== Error in run: " << e.what() << std::endl;
        throw e;
    }

    return TensorLite(output_tensor);
}
