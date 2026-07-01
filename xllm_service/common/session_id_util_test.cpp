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

#include <gtest/gtest.h>

#include <string>

namespace xllm_service {

TEST(SessionIdUtil, AcceptsValidSessionIds) {
  EXPECT_TRUE(is_valid_session_id("sess-abc123"));
  EXPECT_TRUE(is_valid_session_id("A.B_c-9"));
}

TEST(SessionIdUtil, RejectsEmptyOrOversizedSessionIds) {
  EXPECT_FALSE(is_valid_session_id(""));
  EXPECT_FALSE(is_valid_session_id(std::string(kMaxSessionIdLength + 1, 'a')));
}

TEST(SessionIdUtil, RejectsInvalidCharacters) {
  EXPECT_FALSE(is_valid_session_id("sess with space"));
  EXPECT_FALSE(is_valid_session_id("sess/id"));
  EXPECT_FALSE(is_valid_session_id("sess:id"));
}

TEST(SessionIdUtil, HashIsStableForSameAffinityKey) {
  const std::string key = make_session_affinity_key("sess-abc", "Qwen-7B");
  EXPECT_EQ(hash_session_affinity_key(key), hash_session_affinity_key(key));
  EXPECT_NE(hash_session_affinity_key(make_session_affinity_key("sess-abc", "A")),
            hash_session_affinity_key(make_session_affinity_key("sess-abc", "B")));
}

}  // namespace xllm_service
