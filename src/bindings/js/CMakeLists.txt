# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

project(OpenVINO_JS_API DESCRIPTION "OpenVINO Runtime JS bindings")

set(js_source_dir "${CMAKE_CURRENT_SOURCE_DIR}/")
set(js_install_dir "js/")

ov_option(ENABLE_JS "Enables OpenVINO JS API build" OFF)

if(ENABLE_JS)
  install(FILES ${js_source_dir}/tsconfig.json DESTINATION ${js_install_dir})
  install(FILES ${js_source_dir}/.eslintrc-global.js DESTINATION ${js_install_dir})
  install(FILES ${js_source_dir}/.eslintrc.js DESTINATION ${js_install_dir})

  add_subdirectory(node)
endif()
