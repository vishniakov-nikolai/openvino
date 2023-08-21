// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

/*This test runs the following subgraph:

                      param
                        |
                        |
                      Split
                     /  |  \
                    /   |   \
                  Add  Add  Add
                    \   |   /\
                     \  |  /  \
                      Concat  Result
                     /  |  \            
                    /   |   \
                  Add  Add   Result
                   |    |
                  Add  Add
                  /     |
               Result  Result

The main purpose of the test is to check the memory sharing between result and in_place edges.
*/

using namespace InferenceEngine;
using namespace ov::test;

namespace SubgraphTestsDefinitions {

class SplitConcatAddInPlace : virtual public ov::test::SubgraphBaseTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        const auto precision = ov::element::f32;
        ov::test::InputShape input_shape{{}, {{1, 3, 3, 3}}};
        init_input_shapes({input_shape});

        auto params = ngraph::builder::makeDynamicParams(precision, inputDynamicShapes);
        auto split = ngraph::builder::makeSplit(params.front(), precision, 3, 1);
        auto add_const = ngraph::builder::makeConstant(precision, {1}, std::vector<float>({1.0f}));
        auto add_1 = ngraph::builder::makeEltwise(split->output(0), add_const, ngraph::helpers::EltwiseTypes::ADD);
        auto result_add_1 = std::make_shared<ngraph::opset3::Result>(add_1);
        auto add_2 = ngraph::builder::makeEltwise(split->output(1), add_const, ngraph::helpers::EltwiseTypes::ADD);
        auto add_3 = ngraph::builder::makeEltwise(split->output(2), add_const, ngraph::helpers::EltwiseTypes::ADD);
        auto concat = ngraph::builder::makeConcat({add_1, add_2, add_3}, 1);
        auto result_concat = std::make_shared<ngraph::opset3::Result>(concat);
        auto add_4 = ngraph::builder::makeEltwise(concat, add_const, ngraph::helpers::EltwiseTypes::ADD);
        auto add_5 = ngraph::builder::makeEltwise(concat, add_const, ngraph::helpers::EltwiseTypes::ADD);
        auto result_1 = std::make_shared<ngraph::opset3::Result>(add_4);
        auto result_2 = std::make_shared<ngraph::opset3::Result>(add_5);
        ngraph::ResultVector results = {result_1, result_2, result_add_1, result_concat};
        function = std::make_shared<ov::Model>(results, params, "Subgraph");
    }
};

TEST_F(SplitConcatAddInPlace, smoke_CompareWithRefs) {
    run();
}

} // namespace SubgraphTestsDefinitions