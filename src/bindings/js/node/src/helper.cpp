// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "helper.hpp"

#include <iostream>

#include "tensor.hpp"

const std::vector<std::string>& get_supported_types() {
    static const std::vector<std::string> supported_element_types = {"i8",
                                                                     "u8",
                                                                     "i16",
                                                                     "u16",
                                                                     "i32",
                                                                     "u32",
                                                                     "f32",
                                                                     "f64",
                                                                     "i64",
                                                                     "u64"};
    return supported_element_types;
}

napi_types napiType(Napi::Value val) {
    if (val.IsTypedArray())
        return val.As<Napi::TypedArray>().TypedArrayType();
    else if (val.IsArray())
        return js_array;
    else
        return val.Type();
}

bool acceptableType(Napi::Value val, const std::vector<napi_types>& acceptable) {
    return std::any_of(acceptable.begin(), acceptable.end(), [val](napi_types t) {
        return napiType(val) == t;
    });
}

template <>
int32_t js_to_cpp<int32_t>(const Napi::CallbackInfo& info,
                           const size_t idx,
                           const std::vector<napi_types>& acceptable_types) {
    const auto elem = info[idx];
    if (!acceptableType(elem, acceptable_types))
        throw std::invalid_argument(std::string("Cannot convert argument" + std::to_string(idx)));
    if (!elem.IsNumber()) {
        throw std::invalid_argument(std::string("Passed argument must be a number."));
    }
    return elem.ToNumber().Int32Value();
}

template <>
std::string js_to_cpp<std::string>(const Napi::CallbackInfo& info,
                                   const size_t idx,
                                   const std::vector<napi_types>& acceptable_types) {
    const auto elem = info[idx];
    if (!acceptableType(elem, acceptable_types))
        throw std::invalid_argument(std::string("Cannot convert argument") + std::to_string(idx));
    if (!elem.IsString()) {
        throw std::invalid_argument(std::string("Passed argument must be a string."));
    }
    return elem.ToString();
}

template <>
std::vector<size_t> js_to_cpp<std::vector<size_t>>(const Napi::CallbackInfo& info,
                                                   const size_t idx,
                                                   const std::vector<napi_types>& acceptable_types) {
    const auto elem = info[idx];
    if (!acceptableType(elem, acceptable_types))
        throw std::invalid_argument(std::string("Cannot convert argument") + std::to_string(idx));
    if (!elem.IsArray() && !elem.IsTypedArray()) {
        throw std::invalid_argument(std::string("Passed argument must be of type Array or TypedArray."));
    } else if (elem.IsArray()) {
        auto array = elem.As<Napi::Array>();
        size_t arrayLength = array.Length();

        std::vector<size_t> nativeArray;

        for (size_t i = 0; i < arrayLength; ++i) {
            Napi::Value arrayItem = array[i];
            if (!arrayItem.IsNumber()) {
                throw std::invalid_argument(std::string("Passed array must contain only numbers."));
            }
            Napi::Number num = arrayItem.As<Napi::Number>();
            nativeArray.push_back(static_cast<size_t>(num.Int32Value()));
        }
        return nativeArray;

    } else {  //( elem.IsTypedArray()){
        Napi::TypedArray buf;
        napi_typedarray_type type = elem.As<Napi::TypedArray>().TypedArrayType();
        if ((type != napi_int32_array) && (type != napi_uint32_array)) {
            throw std::invalid_argument(std::string("Passed argument must be a Int32Array."));
        } else if ((type == napi_uint32_array))
            buf = elem.As<Napi::Uint32Array>();
        else {
            buf = elem.As<Napi::Int32Array>();
        } 
        auto data_ptr = static_cast<int*>(buf.ArrayBuffer().Data());
        std::vector<size_t> vector(data_ptr, data_ptr + buf.ElementLength());
        return vector;
    }
}

template <>
std::unordered_set<std::string> js_to_cpp<std::unordered_set<std::string>>(
    const Napi::CallbackInfo& info,
    const size_t idx,
    const std::vector<napi_types>& acceptable_types) {
    const auto elem = info[idx];
    if (!elem.IsArray()) {
        throw std::invalid_argument(std::string("Passed argument must be of type Array."));
    } else {
        auto array = elem.As<Napi::Array>();
        size_t arrayLength = array.Length();

        std::unordered_set<std::string> nativeArray;

        for (size_t i = 0; i < arrayLength; ++i) {
            Napi::Value arrayItem = array[i];
            if (!arrayItem.IsString()) {
                throw std::invalid_argument(std::string("Passed array must contain only strings."));
            }
            Napi::String str = arrayItem.As<Napi::String>();
            nativeArray.insert(str.Utf8Value());
        }
        return nativeArray;
    }
}

template <>
ov::element::Type_t js_to_cpp<ov::element::Type_t>(const Napi::CallbackInfo& info,
                                                   const size_t idx,
                                                   const std::vector<napi_types>& acceptable_types) {
    const auto elem = info[idx];
    if (!acceptableType(elem, acceptable_types))
        throw std::invalid_argument(std::string("Cannot convert Napi::Value to ov::element::Type_t"));
    const std::string type = elem.ToString();
    const auto& types = get_supported_types();
    if (std::find(types.begin(), types.end(), type) == types.end())
        throw std::invalid_argument(std::string("Cannot create ov::element::Type"));

    return static_cast<ov::element::Type_t>(ov::element::Type(type));
}

template <>
ov::Layout js_to_cpp<ov::Layout>(const Napi::CallbackInfo& info,
                                 const size_t idx,
                                 const std::vector<napi_types>& acceptable_types) {
    auto layout = js_to_cpp<std::string>(info, idx, acceptable_types);
    return ov::Layout(layout);
}

template <>
ov::Shape js_to_cpp<ov::Shape>(const Napi::CallbackInfo& info,
                               const size_t idx,
                               const std::vector<napi_types>& acceptable_types) {
    auto shape = js_to_cpp<std::vector<size_t>>(info, idx, acceptable_types);
    return ov::Shape(shape);
}

template <>
Napi::String cpp_to_js<ov::element::Type_t, Napi::String>(const Napi::CallbackInfo& info,
                                                          const ov::element::Type_t type) {
    return Napi::String::New(info.Env(), ov::element::Type(type).to_string());
}

ov::Tensor get_request_tensor(ov::InferRequest infer_request, std::string key) {
    return infer_request.get_tensor(key);
}

ov::Tensor get_request_tensor(ov::InferRequest infer_request, size_t idx) {
    return infer_request.get_input_tensor(idx);
}

ov::Tensor cast_to_tensor(Napi::Object obj) {
    // Check of object type
    auto tensor_wrap = Napi::ObjectWrap<TensorWrap>::Unwrap(obj);
    return tensor_wrap->get_tensor();
}

ov::Tensor cast_to_tensor(Napi::TypedArray typed_array, const ov::Shape& shape, const ov::element::Type_t& type) {
    /* The difference between TypedArray::ArrayBuffer::Data() and e.g. Float32Array::Data() is byteOffset
    because the TypedArray may have a non-zero `ByteOffset()` into the `ArrayBuffer`. */
    if (typed_array.ByteOffset() != 0) {
        throw std::invalid_argument("TypedArray.byteOffset has to be equal to zero.");
    }
    auto array_buffer = typed_array.ArrayBuffer();
    auto tensor = ov::Tensor(type, shape, array_buffer.Data());
    if (tensor.get_byte_size() != array_buffer.ByteLength()) {
        throw std::invalid_argument("Memory allocated using shape and element::type mismatch passed data's size");
    }
    return tensor;
}
