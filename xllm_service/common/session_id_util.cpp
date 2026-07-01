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

#include "session_id_util.h"

#include <xxhash.h>

namespace xllm_service {

namespace {

bool is_unreserved_char(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.';
}

}  // namespace

bool is_valid_session_id(const std::string& session_id) {
  if (session_id.empty() || session_id.size() > kMaxSessionIdLength) {
    return false;
  }
  for (char c : session_id) {
    if (!is_unreserved_char(c)) {
      return false;
    }
  }
  return true;
}

std::string make_session_affinity_key(const std::string& session_id,
                                      const std::string& model) {
  std::string key;
  key.reserve(session_id.size() + 1 + model.size());
  key.append(session_id);
  key.push_back('\0');
  key.append(model);
  return key;
}

uint64_t hash_session_affinity_key(const std::string& affinity_key) {
  return XXH64(affinity_key.data(), affinity_key.size(), 0);
}

}  // namespace xllm_service
