/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm-service/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>
#include <string>

namespace xllm_service {

constexpr char kSessionIdHeader[] = "X-Session-Id";
constexpr size_t kMaxSessionIdLength = 128;

bool is_valid_session_id(const std::string& session_id);

// affinity_key = session_id + '\0' + model
std::string make_session_affinity_key(const std::string& session_id,
                                    const std::string& model);

uint64_t hash_session_affinity_key(const std::string& affinity_key);

}  // namespace xllm_service
