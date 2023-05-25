// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/session.h"

#include <limits.h>

#include <iostream>

#include "../include/shape_lite.h"
#include "openvino/openvino.hpp"

using supported_type_t = std::unordered_map<std::string, ov::element::Type>;
ov::element::Type getType2(std::string value, const supported_type_t& supported_precisions) {
    const auto precision = supported_precisions.find(value);
    if (precision == supported_precisions.end()) {
        throw std::logic_error("\"" + value + "\"" + " is not a valid precision");
    }

    return precision->second;
}
ov::element::Type get_type2(const std::string& value) {
    static const supported_type_t supported_types = {
        {"i8", ov::element::i8},
        {"u8", ov::element::u8},
        // {"u8c", ov::element::u8},

        {"i16", ov::element::i16},
        {"u16", ov::element::u16},

        {"i32", ov::element::i32},
        {"u32", ov::element::u32},

        {"f32", ov::element::f32},
        {"f64", ov::element::f64},

        {"i64", ov::element::i64},
        {"u64", ov::element::u64},
    };

    return getType2(value, supported_types);
}

std::shared_ptr<ov::Model> loadModel(std::string xml_path, std::string bin_path) {
    ov::Core core;

    try {
        return core.read_model(xml_path, bin_path);
    } catch (const std::exception& e) {
        std::cout << "== Error in load_model: " << e.what() << std::endl;
        throw e;
    }
}

void prePostProcessModel(std::shared_ptr<ov::Model> model, ShapeLite* shape, std::string layout, std::string input_type) {
    if (shape == nullptr && layout == "" && input_type == "") return;

    if ((shape == nullptr || layout == "") && !(shape == nullptr && layout == ""))
        throw ov::Exception("Shape and layout should be defined together");

    ov::preprocess::PrePostProcessor ppp(model);

    if (shape != nullptr) {
        ov::Layout tensor_layout = ov::Layout(layout);

        ov::Shape original_shape = shape->get_original();

        std::cout << "== Shape: " << original_shape.to_string() << " Layout: " << layout << std::endl;

        ppp.input().tensor().set_shape(original_shape).set_layout(tensor_layout);
    }

    if (input_type != "") {
        std::cout << "== Input type: " << input_type << std::endl;

        ov::element::Type type = get_type2(input_type);

        ppp.input().tensor().set_element_type(type);
    }

    ppp.build();
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
        prePostProcessModel(model, shape, layout, input_type);
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
