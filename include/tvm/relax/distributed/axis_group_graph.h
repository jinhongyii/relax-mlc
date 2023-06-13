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

#include <tvm/tir/function.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/distributed/struct_info.h>

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
class AxisShardingPlanEqual {
 public:
  bool operator()(const AxisShardingPlan& lhs, const AxisShardingPlan& rhs) const {
    return StructuralEqual()(lhs.first, rhs.first) && lhs.second == rhs.second;
  }
};

class AxisShardingPlanHash {
 public:
  size_t operator()(const AxisShardingPlan& sharding_plan) const {
    size_t seed = 0;
    seed ^= StructuralHash()(sharding_plan.first);
    seed ^= std::hash<int>()(sharding_plan.second) << 1;
    return seed;
  }
};

//todo: enable cutting edge
class AxisGroupGraph {
  public:
  enum class EdgeType { kAscend, kDescend, kSimbling };
  private:
  EdgeType ReverseEdgeType(EdgeType type){
    switch(type){
      case EdgeType::kAscend:
        return EdgeType::kDescend;
      case EdgeType::kDescend:
        return EdgeType::kAscend;
      case EdgeType::kSimbling:
        return EdgeType::kSimbling;
    }
  }

  int GetEdgePriority(EdgeType type){
    switch(type){
      case EdgeType::kAscend:
        return 0;
      case EdgeType::kDescend:
        return 2;
      case EdgeType::kSimbling:
        return 1;
    }
  }

  struct AxisGraphEdge{
    Axis src;
    Axis dst;
    EdgeType type;

    bool operator==(const AxisGraphEdge& other) const {
      return src == other.src && dst == other.dst && type == other.type;
    }
  };

  public : AxisGroupGraph() = default;

  void JoinAxis(Axis axis1, Axis axis2, EdgeType type){
    AddEdge(axis1, axis2, type);
    AddEdge(axis2, axis1, ReverseEdgeType(type));
  }

  void AddSrcShardingPoint(Axis axis, AxisShardingPlan plan){
    src_axis_sharding_plan_[axis] = plan;
  }

  void PropagateShardingPlan(){
    axis_sharding_plans_priority_.clear();
    for(const auto& pr: src_axis_sharding_plan_){
      std::unordered_set<Axis, AxisHash> visited;
      PropagateShardingPlan(pr.first, pr.second, GetEdgePriority(EdgeType::kDescend), &visited);
    }
    ChooseAxisShardingPlan();
  }

  void AddPropagationCutPoint(Axis axis, AxisShardingPlan plan){
    cutpoint_axis_sharding_plan_[axis] = plan;
  }

  std::tuple<AxisShardingPlan, bool> GetAxisShardingPlan(Axis axis) {
    if (axis_sharding_plans_priority_.count(axis)) {
      return {axis_sharding_plans_priority_[axis].begin()->first, true};
    } else {
      return {{DeviceMesh(), -1}, false};
    }
  }

 private:
  void AddEdge(Axis src, Axis dst, EdgeType type){
    if (!graph_.count(src)){
      graph_[src] = {};
    }
    graph_[src].push_back({src, dst, type});
  }

  void PropagateShardingPlan(Axis axis, AxisShardingPlan plan, int priority, std::unordered_set<Axis, AxisHash>* visited){
    if (cutpoint_axis_sharding_plan_.count(axis) ||
        (src_axis_sharding_plan_.count(axis) && !AxisShardingPlanEqual()(src_axis_sharding_plan_[axis], plan)) || visited->count(axis)) {
      return;
    }
    visited->insert(axis);
    if (!axis_sharding_plans_priority_.count(axis)) {
      axis_sharding_plans_priority_[axis] = {};
    }
    axis_sharding_plans_priority_[axis][plan] = priority;
    for(auto edge : graph_[axis]){
      PropagateShardingPlan(edge.dst, plan, std::min(priority, GetEdgePriority(edge.type)), visited);
    }
  }

  void ChooseAxisShardingPlan(){
    for(auto& pr : axis_sharding_plans_priority_){
      auto& axis = pr.first;
      auto& plans = pr.second;
      int min_priority = std::numeric_limits<int>::max();
      for(auto& pr2 : plans){
        min_priority = std::min(min_priority, pr2.second);
      }
      for(auto it = plans.begin(); it != plans.end();){
        if(it->second != min_priority){
          it = plans.erase(it);
        }else{
          it++;
        }
      }
      ICHECK(plans.size() == 1) << "multiple possible sharding for axis " << axis.var->name_hint() << " dim " << axis.dim;
    }
  }

  //union set
  std::unordered_map<Axis, std::vector<AxisGraphEdge>, AxisHash> graph_;
  std::unordered_map<Axis, AxisShardingPlan, AxisHash> src_axis_sharding_plan_;
  std::unordered_map<Axis, AxisShardingPlan, AxisHash> cutpoint_axis_sharding_plan_;
  std::unordered_map<Axis, std::unordered_map<AxisShardingPlan, int, AxisShardingPlanHash, AxisShardingPlanEqual>, AxisHash> axis_sharding_plans_priority_;
  
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
//assume output must be a tensor/dtensor (not tuple)
void BuildAxisGraphCallTIR(const Var& output_var, const Call& call, const tir::PrimFunc& func,
                           distributed::AxisGroupGraph* axis_group_graph);

}  // namespace distributed
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_DISTRIBUTED_AXIS_GROUP_GRAPH_H_