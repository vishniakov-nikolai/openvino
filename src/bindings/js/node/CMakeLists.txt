# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT TARGET nodejs_api)
    add_custom_target(nodejs_api)
endif()

set(TARGET_NAME "nodejs_api")

set(js_source_dir "${CMAKE_CURRENT_SOURCE_DIR}/")

install(DIRECTORY ${js_source_dir}/ DESTINATION "${js_install_dir}/node")

# perform building of nodejs bindings
# add_custom_target(TARGET ${TARGET_NAME}
#     COMMAND RUNTIME_DIR='../..' npm install
#     COMMENT "Running custom command after installation"
# )

# add_dependencies(${TARGET_NAME} nodejs_api)
