// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <map>
#include <regex>
#include <unordered_set>
#include <string>
#include <functional>

#include "openvino/util/file_util.hpp"

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_constants.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"

#include "cache/cache.hpp"
#include "utils/node.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

static std::vector<std::regex> FROTEND_REGEXP = {
#ifdef ENABLE_OV_ONNX_FRONTEND
    std::regex(R"(.*\.onnx)"),
#endif
#ifdef ENABLE_OV_PADDLE_FRONTEND
    std::regex(R"(.*\.pdmodel)"),
    std::regex(R"(.*__model__)"),
#endif
#ifdef ENABLE_OV_TF_FRONTEND
    std::regex(R"(.*\.pb)"),
#endif
#ifdef ENABLE_OV_IR_FRONTEND
    std::regex(R"(.*\.xml)"),
#endif
#ifdef ENABLE_OV_TF_LITE_FRONTEND
    std::regex(R"(.*\.tflite)"),
#endif
#ifdef ENABLE_OV_PYTORCH_FRONTEND
    std::regex(R"(.*\.pt)"),
#endif
};

enum ModelCacheStatus {
    SUCCEED = 0,
    NOT_FULLY_CACHED = 1,
    NOT_READ = 2
};

static std::map<ModelCacheStatus, std::string> model_cache_status_to_str = {
    { ModelCacheStatus::SUCCEED, "successful_models" },
    { ModelCacheStatus::NOT_FULLY_CACHED, "not_fully_cached_models" },
    { ModelCacheStatus::NOT_READ, "not_read_models" },
};

std::pair<std::vector<std::string>, std::pair<ModelCacheStatus, std::vector<std::string>>>
find_models(const std::vector<std::string> &dirs, const std::string& regexp = ".*");

// model_cache_status: model_list
std::map<ModelCacheStatus, std::vector<std::string>> cache_models(
    std::vector<std::shared_ptr<ICache>>& caches,
    const std::vector<std::string>& models,
    bool extract_body);

void save_model_status_to_file(const std::map<ModelCacheStatus, std::vector<std::string>>& caching_status,
                               const std::string& output_dir);

inline ExtractedPattern
generate_model(const std::set<std::shared_ptr<ov::Node>>& nodes,
               std::unordered_set<std::string>& checked_ops,
               const std::string& extractor_name) {
    // map to recover graph using cloned nodes and original connections
    // { original_node_name, cloned_node }
    std::unordered_map<std::string, std::shared_ptr<ov::Node>> cloned_node_map;
    // map to fill output nodes in models:
    // { original_node_names, out_port_idx_without_orig_node_to_check }
    std::unordered_map<std::string, std::unordered_set<size_t>> model_output_nodes;
    std::map<std::string, InputInfo> model_input_info;
    ov::ParameterVector model_parameters;
    {
        // prepare map { original_op_name, cloned_node }
        size_t functional_node_cnt = 0;
        for (const auto& node : nodes) {
            auto orig_node_name = node->get_friendly_name();
            checked_ops.insert(orig_node_name);
            cloned_node_map.insert({ orig_node_name,
                                     clone_node(node, true, false, orig_node_name) });
            
            // create temporary vector to fill node output indexes
            std::vector<size_t> out_ports(node->outputs().size());
            std::iota(out_ports.begin(), out_ports.end(), 0);
            // fill by all nodes with output ports
            model_output_nodes.insert({ 
                orig_node_name, 
                std::unordered_set<size_t>(out_ports.begin(), out_ports.end()) });
            if (!ov::op::util::is_output(node) &&
                !ov::op::util::is_constant(node) &&
                !ov::op::util::is_parameter(node)) {
                ++functional_node_cnt;
            }
        }

        if (functional_node_cnt < 2) {
            throw std::runtime_error("Incorrect node number to create model!");
        }
        // replace new inputs by taken from graph if possible and
        // find input and output nodes in future subgraph
        for (const auto& node : nodes) {
            // variable to store updated input index
            int filled_input_idx = -1;
            auto cloned_node = cloned_node_map[node->get_friendly_name()];
            auto node_input_info = get_input_info_by_node(cloned_node);
            for (size_t in_idx = 0; in_idx < node->inputs().size(); ++in_idx) {
                auto orig_in_node = node->get_input_node_ptr(in_idx)->shared_from_this();
                for (size_t out_idx = 0; out_idx < orig_in_node->outputs().size(); ++out_idx) {
                    for (const auto& orig_node_to_check : orig_in_node->output(out_idx).get_target_inputs()) {
                        if (orig_node_to_check.get_node()->shared_from_this() == node) {
                            auto orig_in_node_name = orig_in_node->get_friendly_name();
                            auto cloned_in_node = cloned_node->get_input_node_shared_ptr(in_idx);
                            // if op input node is in subgraph replace parameters 
                            // in cloned node by other nodes from the map
                            if (cloned_node_map.count(orig_in_node_name)) {
                                auto orig_in_node = cloned_node_map[orig_in_node_name];
                                auto cloned_in_node_name = cloned_in_node->get_friendly_name();
                                ov::replace_output_update_name(cloned_in_node->get_default_output(), orig_in_node->output(out_idx));
                                if (ov::op::util::is_parameter(orig_in_node)) {
                                    auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(orig_in_node);
                                    model_parameters.push_back(param);
                                    node_input_info.insert({ orig_in_node->get_friendly_name(),
                                                             node_input_info[cloned_in_node_name]});
                                } else if (ov::op::util::is_constant(orig_in_node)) {
                                    auto op_to_replace = std::dynamic_pointer_cast<ov::op::v0::Constant>(orig_in_node);
                                    auto param = convert_const_to_param(op_to_replace);
                                    if (param != nullptr) {
                                        model_parameters.push_back(param);
                                    }
                                    node_input_info.insert({ orig_in_node->get_friendly_name(),
                                                             node_input_info[cloned_in_node_name]});
                                }
                                filled_input_idx++;
                                // clean up replaced node data
                                node_input_info.erase(cloned_in_node_name);
                                model_output_nodes[orig_in_node_name].erase(out_idx);
                                if (model_output_nodes[orig_in_node_name].empty()) {
                                    model_output_nodes.erase(orig_in_node_name);
                                }
                            } else if (ov::op::util::is_parameter(cloned_in_node)) {
                                auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(cloned_in_node);
                                model_parameters.push_back(param);
                            } else if (ov::op::util::is_constant(cloned_in_node)) {
                                auto op_to_replace = std::dynamic_pointer_cast<ov::op::v0::Constant>(cloned_in_node);
                                auto param = convert_const_to_param(op_to_replace);
                                if (param != nullptr) {
                                    model_parameters.push_back(param);
                                }
                            }
                            break;
                        }
                    }
                    if (filled_input_idx == in_idx) {
                        break;
                    }
                }
            }
            if (!node_input_info.empty()) {
                model_input_info.insert(node_input_info.begin(), node_input_info.end());
            }
        }
    }
    ov::ResultVector model_results;
    for (const auto& out_node_name : model_output_nodes) {
        auto out_node = cloned_node_map[out_node_name.first];
        if (ov::op::util::is_output(out_node)) {
            model_results.push_back(std::dynamic_pointer_cast<ov::op::v0::Result>(out_node));
        } else {
            for (const auto& out_port_id : out_node_name.second) {
                model_results.push_back(std::make_shared<ov::op::v0::Result>(out_node->output(out_port_id)));
            }
        }
    }
    auto model = std::make_shared<ov::Model>(model_results, model_parameters);

    // prepare unique model name based on operations from model
    std::string string_to_hash;
    for (const auto& op : model->get_ordered_ops()) {
        std::ostringstream result;
        result << op->get_type_info();
        for (const auto& in : op->inputs()) { 
            result << in.get_element_type();
            result << in.get_partial_shape().rank();
            result << in.get_partial_shape().is_static();
        }
        for (const auto& out : op->outputs()) {
            result << out.get_element_type();
            result << out.get_partial_shape().rank();
            result << out.get_partial_shape().is_static();
        }
        string_to_hash += result.str();
    }
    for (const auto& in : model_input_info) {
        string_to_hash += (in.second.is_const ? "1" : "0");
    }
    auto h1 = std::hash<std::string>{}(string_to_hash);
    model->set_friendly_name(std::to_string(h1));

    return { model, model_input_info, extractor_name };
}

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
