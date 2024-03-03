#!/bin/bash
#
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -e
set -x

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MAKEFILE="${SCRIPT_DIR}/../Makefile"
readonly DOCKER_CPUS="${DOCKER_CPUS:=k8 aarch64 armv7a}"
readonly TARGETS="${TARGETS:=tests benchmarks tools examples}"

# Check if the CPU was passed to the script in DOCKER_CPUS.
# If so, build for that CPU on the specified docker image.
# Args:
#   - CPU
#   - Docker image name
function docker_build {
  local cpu="$1"
  local image="$2"

  if [[ " ${DOCKER_CPUS[@]} " =~ "${cpu}" ]]; then
    make DOCKER_IMAGE="${image}" DOCKER_CPUS="${cpu}" \
       DOCKER_TARGETS="${TARGETS}" \
       -f "${MAKEFILE}" docker-build
  else
    echo "Skipping build of ${cpu}."
  fi
}

for i in "$@"; do
  if [[ "$i" == --clean ]]; then
    make -f "${MAKEFILE}" clean
  fi
done

# Build for k8 (use Ubuntu 22.04 for compatibility with most platforms).
docker_build "k8" "ubuntu:22.04"

# Build for armv7a.
docker_build "armv7a" "debian:bookworm"

# Build for aarch64.
docker_build "aarch64" "debian:bookworm"
