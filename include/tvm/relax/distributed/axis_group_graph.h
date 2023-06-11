/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef TVM_RELAX_DISTRIBUTED_AXIS_GROUP_GRAPH_H_
#define TVM_RELAX_DISTRIBUTED_AXIS_GROUP_GRAPH_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {
namespace distributed {



struct Axis{
  const VarNode* var = nullptr;
  int dim = 0;

  bool operator==(const Axis& other) const {
    return var == other.var && dim == other.dim;
  }
};


class AxisHash{
  public:
  size_t operator()(const Axis &axis) const
    {
        size_t const h1 ( std::hash<const VarNode*>()(axis.var) );
        size_t const h2 ( std::hash<int>()(axis.dim) );
        return h1 ^ (h2 << 1);
    }
};


using AxisGroup = std::unordered_set<Axis, AxisHash>;

class AxisGroupHash{
  public:
  size_t operator()(const AxisGroup &axis_group) const{
      size_t seed = 0;
      for(auto axis : axis_group){
        seed ^= AxisHash()(axis) + 0x9e3779b9 + (seed<<6) + (seed>>2);
      }
      return seed;
  }
};

using ShardingPlan = std::pair<DeviceMesh, Placement>;

// device mesh and the device mesh axis that the tensor axis maps to
using AxisShardingPlan = std::pair<DeviceMesh, int>;

//todo: enable cutting edge
class AxisGroupGraph {

  struct AxisGraphEdge{
    Axis src;
    Axis dst;

    bool operator==(const AxisGraphEdge& other) const {
      return src == other.src && dst == other.dst;
    }
  };

  public : AxisGroupGraph() = default;

  void JoinAxis(Axis axis1, Axis axis2){
    AddEdge(axis1, axis2);
    AddEdge(axis2, axis1);
  }

  void AddSrcShardingPoint(Axis axis, AxisShardingPlan plan){
    src_axis_sharding_plan_[axis] = plan;
  }

  void PropagateShardingPlan(){
    for(const auto& pr: src_axis_sharding_plan_){
      PropagateShardingPlan(pr.first, pr.second);
    }
  }

  void AddPropagationCutPoint(Axis axis, AxisShardingPlan plan){
    cutpoint_axis_sharding_plan_[axis] = plan;
  }

  private:
  void AddEdge(Axis src, Axis dst){
    if (!graph_.count(src)){
      graph_[src] = {};
    }
    graph_[src].push_back({src, dst});
  }
  void PropagateShardingPlan(Axis axis, AxisShardingPlan plan){
    if(cutpoint_axis_sharding_plan_.count(axis)){
      return;
    }
    if(!axis_sharding_plans_.count(axis)){
      axis_sharding_plans_[axis] = {};
    }
    axis_sharding_plans_[axis].push_back(plan);
    for(auto edge : graph_[axis]){
      PropagateShardingPlan(edge.dst, plan);
    }
  }


  //union set
  std::unordered_map<Axis, std::vector<AxisGraphEdge>, AxisHash> graph_;
  std::unordered_map<Axis, AxisShardingPlan, AxisHash> src_axis_sharding_plan_;
  std::unordered_map<Axis, AxisShardingPlan, AxisHash> cutpoint_axis_sharding_plan_;
  std::unordered_map<Axis, std::vector<AxisShardingPlan>, AxisHash> axis_sharding_plans_;
  
};

using FBuildAxisGraph = std::function<void(
    const Var& output_var, const Call& call,
    distributed::AxisGroupGraph* axis_group_graph)>;

void BuildAxisGraphUnary(const Var& output_var, const Call& call,
                           distributed::AxisGroupGraph* axis_group_graph);
void BuildAxisGraphBinary(const Var& output_var, const Call& call,
                           distributed::AxisGroupGraph* axis_group_graph);
void BuildAxisGraphReduce(const Var& output_var, const Call& call,
                           distributed::AxisGroupGraph* axis_group_graph);
void BuildAxisGraphMatmul(const Var& output_var, const Call& call,
                           distributed::AxisGroupGraph* axis_group_graph);
void BuildAxisGraphPermuteDims(const Var& output_var, const Call& call,
                           distributed::AxisGroupGraph* axis_group_graph);
void BuildAxisGraphReshape(const Var& output_var, const Call& call,
                           distributed::AxisGroupGraph* axis_group_graph);

}  // namespace distributed
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_DISTRIBUTED_AXIS_GROUP_GRAPH_H_