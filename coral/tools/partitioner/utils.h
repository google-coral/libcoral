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

#ifndef LIBCORAL_CORAL_TOOLS_PARTITIONER_UTILS_H_
#define LIBCORAL_CORAL_TOOLS_PARTITIONER_UTILS_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace coral {

// Represents a directional edge, e.g., {1,2} is an edge pointing from node 1 to
// node 2.
using Edge = std::pair<int, int>;

// Directed graph represented as adjacency list. graph[i] gives the children
// nodes of node `i`.
using Graph = std::vector<std::vector<int>>;

// Special node index to represent the (fake) node which generates (fake) input
// edge of a graph.
constexpr int kGraphInputGenNode = -1;

// Describes nodes of a subgraph.
struct SubgraphNodes {
  // Indices of all input boundary nodes of the subgraph.
  //
  // Note that `input_nodes` refers to the node that generates the input edge
  // for the subgraph. This makes it easier to find the input tensor of the
  // subgraph because of tflite's representation. A special value of
  // `kGraphInputGenNode` is used to indicate the input is the same as input of
  // the graph.
  absl::flat_hash_set<int> input_nodes;
  // Indices of all output boundary nodes of the subgraph.
  absl::flat_hash_set<int> output_nodes;
  // Indices of all nodes of the subgraph. All of these nodes are topologically
  // sorted.
  std::vector<int> all_nodes;
};

// Builds graph represented as list of edges from tflite::Model.
std::vector<Edge> BuildEdgeList(const tflite::Model& model);

// Builds graph (represented with adjacency list) from list of edges (can be
// viewed as sparse representation of adjacency matrix). Edge is directional,
// e.g., first element (parent) -> second element (child)
Graph BuildGraph(const std::vector<Edge>& edges, int num_nodes);

// Reverses graph, by reversing the direction of every edge, i.e., child node
// becomes parent node; and parent node becomes child node.
Graph BuildReverseGraph(const Graph& graph);

// Returns in-degree for each node.
std::vector<int> CalculateInDegree(const Graph& graph);

// Returns out-degree for each node.
std::vector<int> CalculateOutDegree(const Graph& graph);

// Returns topological sort of nodes.
// The function will log-fatal if input graph is not DAG.
//
// Implemented with Kahn's algorithm as described here.
// https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
std::vector<int> TopologicalSort(const Graph& graph);

std::vector<int> TopologicalSort(const tflite::Model& model);

// Builds a map between tensor name and its index.
absl::flat_hash_map<std::string, int> BuildTensorNameToIndexMap(
    const tflite::Model& model);

// Extracts a model segment from given tflite model.
//
//  *) `target_ops` specifies all the operator indices of the segment, they
//      should be in valid topological sorted order;
//  *) `target_inputs` specifies the tensor indices of the inputs of segment;
//  *) `target_outputs` specifies the tensor indices of the outputs of segment;
//
// Returns tflite flatbuffer.
//
// Note that this function does not verify the given operators can form a valid
// subgraph, it is caller's responsibility to ensure that.
std::string ExtractModelSegment(const tflite::Model& src_model,
                                const std::vector<int>& target_ops,
                                const std::vector<int>& target_inputs,
                                const std::vector<int>& target_outputs);

// Locates nodes for each subgraph.
//
// `num_nodes_per_subgraph` specifies how many nodes each subgraph should
// contain.
//
// Returned nodes' indices are the same as `edges` indexing order.
std::vector<SubgraphNodes> LocateSubgraphNodes(
    const std::vector<Edge>& edges, int num_nodes,
    const std::vector<int>& exe_order_to_node_idx,
    const std::vector<int>& num_nodes_per_subgraph);

// Fully read file_name into contents. On error log a message and exit.
void ReadFileOrExit(const std::string& file_name, std::vector<char>* contents);

// Fully write contents into file file_name. On error log a message and exit.
void WriteFileOrExit(const std::string& file_name, const std::string& contents);

// Calculates parameter (weight) size in bytes for each operator in the model.
std::vector<int64_t> CalculateParameterSizes(const tflite::Model& model);
}  // namespace coral

#endif  // LIBCORAL_CORAL_TOOLS_PARTITIONER_UTILS_H_
