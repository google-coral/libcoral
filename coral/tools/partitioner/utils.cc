/* Copyright 2019-2021 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "coral/tools/partitioner/utils.h"

#include <fstream>
#include <iostream>
#include <stack>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "glog/logging.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"

namespace coral {

namespace {
bool IsCustomOp(const tflite::Model& model, int op_index) {
  const auto* opcodes = model.operator_codes();
  CHECK_EQ(model.subgraphs()->size(), 1);
  const auto* ops = model.subgraphs()->Get(0)->operators();
  const auto* op = ops->Get(op_index);
  int opcode_index = op->opcode_index();
  const auto* opcode = opcodes->Get(opcode_index);
  return ::tflite::GetBuiltinCode(opcode) == ::tflite::BuiltinOperator_CUSTOM;
}

std::string GetOpName(const tflite::Model& model, int op_index) {
  const auto* opcodes = model.operator_codes();
  CHECK_EQ(model.subgraphs()->size(), 1);
  const auto* ops = model.subgraphs()->Get(0)->operators();

  const auto* op = ops->Get(op_index);
  int opcode_index = op->opcode_index();
  const auto* opcode = opcodes->Get(opcode_index);

  const auto builtin_code = ::tflite::GetBuiltinCode(opcode);
  if (builtin_code != ::tflite::BuiltinOperator_CUSTOM) {
    return EnumNameBuiltinOperator(builtin_code);
  } else {
    return opcode->custom_code()->str();
  }
}
}  // namespace

std::vector<Edge> BuildEdgeList(const tflite::Model& model) {
  CHECK_EQ(model.subgraphs()->size(), 1);
  const auto* ops = model.subgraphs()->Get(0)->operators();
  VLOG(1) << "Number of operators: " << ops->size();

  // Map tensor (index) to its consumers (op's indices).
  absl::flat_hash_map<int, std::vector<int>> tensor_consumers;
  // Map tensor (index) to its producers (op's indices). Ideally, there should
  // only  be one producer for each tensor.
  absl::flat_hash_map<int, std::vector<int>> tensor_producers;
  absl::flat_hash_set<int> all_tensor_indices;

  // Construct tensor's consumers and producers maps.
  for (int i = 0; i < ops->size(); ++i) {
    for (const auto& in_tensor_idx : *(ops->Get(i)->inputs())) {
      tensor_consumers[in_tensor_idx].push_back(i);
      all_tensor_indices.insert(in_tensor_idx);
    }
    for (const auto& out_tensor_idx : *(ops->Get(i)->outputs())) {
      tensor_producers[out_tensor_idx].push_back(i);
      all_tensor_indices.insert(out_tensor_idx);
    }
  }

  // Build all edges
  std::vector<Edge> result;
  for (const auto& tensor_idx : all_tensor_indices) {
    const auto& consumers_iter = tensor_consumers.find(tensor_idx);
    const auto& producers_iter = tensor_producers.find(tensor_idx);
    if ((consumers_iter != tensor_consumers.end()) &&
        (producers_iter != tensor_producers.end())) {
      for (const auto& consumer_op_idx : consumers_iter->second) {
        for (const auto& producer_op_idx : producers_iter->second) {
          result.push_back({producer_op_idx, consumer_op_idx});
        }
      }
    }
  }

  return result;
}

Graph BuildGraph(const std::vector<Edge>& edges, int num_nodes) {
  Graph graph(num_nodes);
  for (const auto& edge : edges) {
    graph[edge.first].push_back(edge.second);
  }
  return graph;
}

Graph BuildReverseGraph(const Graph& graph) {
  Graph reverse_graph(graph.size());
  for (int i = 0; i < graph.size(); ++i) {
    for (const auto& child_node : graph[i]) {
      reverse_graph[child_node].push_back(i);
    }
  }
  return reverse_graph;
}

std::vector<int> CalculateInDegree(const Graph& graph) {
  const int num_nodes = graph.size();
  std::vector<int> in_degree(num_nodes, 0);
  for (int i = 0; i < num_nodes; ++i) {
    for (const auto& child_node : graph[i]) {
      in_degree[child_node]++;
    }
  }
  return in_degree;
}

std::vector<int> CalculateOutDegree(const Graph& graph) {
  return CalculateInDegree(BuildReverseGraph(graph));
}

std::vector<int> TopologicalSort(const Graph& graph) {
  const int num_nodes = graph.size();
  // The vector to_visit serves as a stack, where back of the vector is the top
  // of the conceptual stack's top.
  std::vector<int> to_visit;
  std::vector<int> exe_order;
  exe_order.reserve(num_nodes);

  auto in_degree = CalculateInDegree(graph);
  auto out_degree = CalculateOutDegree(graph);
  // Find all nodes with 0 in-degree as starting points.
  for (int i = 0; i < num_nodes; ++i) {
    if (in_degree[i] == 0) {
      to_visit.push_back(i);
    }
  }

  while (!to_visit.empty()) {
    const int node = to_visit.back();
    to_visit.pop_back();
    exe_order.push_back(node);

    for (const auto& child_node : graph[node]) {
      in_degree[child_node]--;
      if (in_degree[child_node] == 0) {
        if (out_degree[child_node] == 0) {
          // If the node has no child, prefer to visit it as late as possible.
          to_visit.insert(to_visit.begin(), child_node);
        } else {
          to_visit.push_back(child_node);
        }
      }
    }
  }

  LOG_IF(FATAL, exe_order.size() != num_nodes) << "Graph is NOT DAG";
  return exe_order;
}

std::vector<int> TopologicalSort(const tflite::Model& model) {
  const auto& edges = BuildEdgeList(model);
  const int num_nodes = model.subgraphs()->Get(0)->operators()->size();
  const auto& graph = BuildGraph(edges, num_nodes);
  return TopologicalSort(graph);
}

absl::flat_hash_map<std::string, int> BuildTensorNameToIndexMap(
    const tflite::Model& model) {
  CHECK_EQ(model.subgraphs()->size(), 1);
  const auto* subgraph = model.subgraphs()->Get(0);
  const auto* tensors = subgraph->tensors();
  absl::flat_hash_map<std::string, int> result(tensors->size());
  for (int i = 0; i < tensors->size(); ++i) {
    const auto* tensor = tensors->Get(i);
    result.insert({tensor->name()->str(), i});
  }
  return result;
}

std::string ExtractModelSegment(const tflite::Model& src_model,
                                const std::vector<int>& target_ops,
                                const std::vector<int>& target_inputs,
                                const std::vector<int>& target_outputs) {
  CHECK_EQ(src_model.subgraphs()->size(), 1);

  VLOG(1) << "Collecting tensors and opcodes...";
  const auto* src_subgraph = src_model.subgraphs()->Get(0);
  const auto* src_ops = src_subgraph->operators();
  std::set<int> target_tensors;
  std::set<int> target_opcodes;
  for (const auto& index : target_ops) {
    const auto* src_op = src_ops->Get(index);
    target_tensors.insert(src_op->inputs()->begin(), src_op->inputs()->end());
    target_tensors.insert(src_op->outputs()->begin(), src_op->outputs()->end());
    target_opcodes.insert(src_op->opcode_index());
  }

  VLOG(1) << "Copying buffers and tensors...";
  const auto* src_tensors = src_subgraph->tensors();
  const auto* src_buffers = src_model.buffers();
  // Note: buffers[0] is always empty!!!
  // see third_party/tensorflow/lite/schema/schema_v3.fbs for details.
  //
  // What this implies is that it is not possible to predetermine how many
  // buffers will be needed as more than one tensors are allowed to point at
  // buffers[0].
  //
  // But, the size of buffers can be at most tensors.size() + 1.
  std::vector<std::unique_ptr<tflite::BufferT>> buffers_copy;
  buffers_copy.reserve(target_tensors.size() + 1);
  // Create buffers[0]
  buffers_copy.push_back(absl::WrapUnique(src_buffers->Get(0)->UnPack()));

  std::vector<std::unique_ptr<tflite::TensorT>> tensors_copy(
      target_tensors.size());
  // Key: index of tensor in subgraphs[0].tensors of src_model.
  // Value: index of tensor in subgraphs[0].tensors of model_copy.
  absl::flat_hash_map<int, int> tensors_src_to_copy_map(target_tensors.size());

  for (int i = 0; i < target_tensors.size(); ++i) {
    const int target_tensor_index = *std::next(target_tensors.begin(), i);
    const auto* src_tensor = src_tensors->Get(target_tensor_index);

    // Copy tensor.
    tensors_copy[i] = absl::WrapUnique(src_tensor->UnPack());

    // Need to update `buffer` index if not 0.
    if (src_tensor->buffer() != 0) {
      // Copy buffer.
      const auto* src_buffer = src_buffers->Get(src_tensor->buffer());
      buffers_copy.push_back(absl::WrapUnique(src_buffer->UnPack()));
      tensors_copy[i]->buffer = buffers_copy.size() - 1;
    }

    tensors_src_to_copy_map.insert({target_tensor_index, i});
  }

  VLOG(1) << "Copying opcodes...";
  std::vector<std::unique_ptr<tflite::OperatorCodeT>> opcodes_copy(
      target_opcodes.size());
  // Key: index of op_code in operator_codes (field) of src_model.
  // Value: index of op_code in operator_codes (field) of model_copy.
  absl::flat_hash_map<int, int> opcodes_src_to_copy_map(opcodes_copy.size());
  const auto* src_opcodes = src_model.operator_codes();
  for (int i = 0; i < target_opcodes.size(); ++i) {
    const int target_opcode_index = *std::next(target_opcodes.begin(), i);
    opcodes_copy[i] =
        absl::WrapUnique(src_opcodes->Get(target_opcode_index)->UnPack());
    opcodes_src_to_copy_map.insert({target_opcode_index, i});
  }

  VLOG(1) << "Copying operators...";
  std::vector<std::unique_ptr<tflite::OperatorT>> ops_copy(target_ops.size());
  for (int i = 0; i < target_ops.size(); ++i) {
    const auto* src_op = src_ops->Get(target_ops[i]);

    auto op_copy = absl::WrapUnique(src_op->UnPack());
    // Update inputs and outputs tensors indices.
    for (int j = 0; j < op_copy->inputs.size(); ++j) {
      op_copy->inputs[j] = tensors_src_to_copy_map.at(op_copy->inputs[j]);
    }
    for (int j = 0; j < op_copy->outputs.size(); ++j) {
      op_copy->outputs[j] = tensors_src_to_copy_map.at(op_copy->outputs[j]);
    }
    // Update opcode index.
    op_copy->opcode_index = opcodes_src_to_copy_map.at(op_copy->opcode_index);

    ops_copy[i] = std::move(op_copy);
  }

  VLOG(1) << "Constructing subgraph....";
  auto subgraph_copy = absl::make_unique<tflite::SubGraphT>();
  subgraph_copy->tensors = std::move(tensors_copy);
  subgraph_copy->operators = std::move(ops_copy);
  if (src_subgraph->name()) {
    subgraph_copy->name = src_subgraph->name()->str();
  }
  subgraph_copy->inputs.resize(target_inputs.size());
  subgraph_copy->outputs.resize(target_outputs.size());
  for (int i = 0; i < target_inputs.size(); ++i) {
    subgraph_copy->inputs[i] = tensors_src_to_copy_map.at(target_inputs[i]);
  }
  for (int i = 0; i < target_outputs.size(); ++i) {
    subgraph_copy->outputs[i] = tensors_src_to_copy_map.at(target_outputs[i]);
  }

  VLOG(1) << "Constructing model...";
  tflite::ModelT model_copy;
  model_copy.description = src_model.description()->str();
  model_copy.version = src_model.version();
  model_copy.buffers = std::move(buffers_copy);
  model_copy.operator_codes = std::move(opcodes_copy);
  model_copy.subgraphs.push_back(std::move(subgraph_copy));

  auto fbb = absl::make_unique<flatbuffers::FlatBufferBuilder>();
  tflite::FinishModelBuffer(*fbb, tflite::Model::Pack(*fbb, &model_copy));
  return {fbb->GetBufferPointer(), fbb->GetBufferPointer() + fbb->GetSize()};
}

std::vector<SubgraphNodes> LocateSubgraphNodes(
    const std::vector<Edge>& edges, int num_nodes,
    const std::vector<int>& exe_order_to_node_idx,
    const std::vector<int>& num_nodes_per_subgraph) {
  const auto& graph = BuildGraph(edges, num_nodes);
  const auto& reverse_graph = BuildReverseGraph(graph);
  const int num_subgraphs = num_nodes_per_subgraph.size();

  // Find the node index -> execution order map.
  std::vector<int> node_idx_to_exe_order(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    node_idx_to_exe_order[exe_order_to_node_idx[i]] = i;
  }

  std::vector<int> num_cumulative_ops(num_subgraphs + 1, 0);
  for (int i = 0; i < num_subgraphs; ++i) {
    num_cumulative_ops[i + 1] =
        num_cumulative_ops[i] + num_nodes_per_subgraph[i];
  }

  std::vector<SubgraphNodes> subgraph_nodes_list(num_subgraphs);
  // Build `subgraph_nodes_list`.
  // For input (boundary) nodes, the condition is:
  //   *) current node's execution order >= input_boundary_num_ops_limit
  //   *) parent node's execution order < input_boundary_num_ops_limit
  // For output (boundary) nodes, the condition is:
  //   *) current node's execution order < output_boundary_num_ops_limit
  //   *) child node's execution order >= output_boundary_num_ops_limit©
  for (int i = 0; i < num_subgraphs; ++i) {
    auto& subgraph_nodes = subgraph_nodes_list[i];
    const int input_boundary_num_ops_limit = num_cumulative_ops[i];
    const int output_boundary_num_ops_limit = num_cumulative_ops[i + 1];
    // `j` is in execution order.
    for (int j = input_boundary_num_ops_limit;
         j < output_boundary_num_ops_limit; ++j) {
      const auto& cur_node = exe_order_to_node_idx[j];
      subgraph_nodes.all_nodes.push_back(cur_node);
      // Check input boundary nodes.
      if (reverse_graph[cur_node].empty()) {
        // No parent nodes, must be input boundary nodes. Special value
        // `kGraphInputGenNode` is used to indicate this subgraph has the same
        // input as whole graph.
        subgraph_nodes.input_nodes.insert(kGraphInputGenNode);
      } else {
        for (const auto& parent_node : reverse_graph[cur_node]) {
          const auto& parent_exe_order = node_idx_to_exe_order[parent_node];
          if (parent_exe_order < input_boundary_num_ops_limit) {
            subgraph_nodes.input_nodes.insert(parent_node);
          }
        }
      }
      // Check output boundary nodes.
      if (graph[cur_node].empty()) {
        // No child nodes, must be output boundary nodes.
        subgraph_nodes.output_nodes.insert(cur_node);
      } else {
        for (const auto& child_node : graph[cur_node]) {
          const auto& child_exe_order = node_idx_to_exe_order[child_node];
          if (child_exe_order >= output_boundary_num_ops_limit) {
            subgraph_nodes.output_nodes.insert(cur_node);
          }
        }
      }
    }
  }

  return subgraph_nodes_list;
}

void ReadFileOrExit(const std::string& file_name, std::vector<char>* contents) {
  std::ifstream file(file_name.c_str(), std::ios::binary | std::ios::ate);
  if (!file) {
    std::cout << "Error opening file for reading: " << file_name << std::endl;
    exit(1);
  }

  auto file_size = file.tellg();
  contents->resize(file_size);
  file.seekg(0, std::ios::beg);
  file.read(contents->data(), contents->size());
  if (!file || file.bad()) {
    std::cout << "Error reading file: " << file_name << std::endl;
    exit(1);
  }
}

void WriteFileOrExit(const std::string& file_name,
                     const std::string& contents) {
  std::ofstream file(file_name, std::ios::binary);
  if (!file) {
    std::cout << "Error opening file for writing: " << file_name << std::endl;
    exit(1);
  }

  file.write(contents.c_str(), contents.size());
  file.flush();
  if (!file || file.bad()) {
    std::cout << "Error writing file: " << file_name << std::endl;
    exit(1);
  }
}

std::vector<int64_t> CalculateParameterSizes(const tflite::Model& model) {
  const int num_nodes = model.subgraphs()->Get(0)->operators()->size();
  VLOG(1) << "Total num nodes: " << num_nodes;
  const auto* ops = model.subgraphs()->Get(0)->operators();
  const auto* tensors = model.subgraphs()->Get(0)->tensors();
  const auto* buffers = model.buffers();

  // Calculate parameter sizes.
  std::vector<int64_t> parameter_sizes(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    const auto* op = ops->Get(i);
    // Do not count parameters for custom operators, which can not be
    // accelerated with Edge TPU.
    if (IsCustomOp(model, i)) {
      parameter_sizes[i] = 0;
    } else {
      // This assumes tflite model does not allocate memory for intermediate
      // tensors.
      int num_non_weight_tensors = 0;
      for (const auto& input_index : *(op->inputs())) {
        const auto* tensor = tensors->Get(input_index);
        const auto* buffer = buffers->Get(tensor->buffer());
        if (!buffer->data()) {
          num_non_weight_tensors++;
        } else {
          parameter_sizes[i] += buffer->data()->size();
        }
        VLOG(1) << "Tensor name: " << tensor->name()->str() << " Buffer size: "
                << (buffer->data() ? buffer->data()->size() : 0);
      }
      // Sanity Check, every op should have non-weights tensor.
      CHECK_GT(num_non_weight_tensors, 0);
    }
    VLOG(1) << GetOpName(model, i)
            << " parameter size (bytes): " << parameter_sizes[i];
  }
  return parameter_sizes;
}
}  // namespace coral
