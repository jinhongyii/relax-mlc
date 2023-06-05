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

//todo: enable cutting edge
class AxisGroupGraph {


  public:
  AxisGroupGraph() = default;

  void JoinAxis(Axis axis1, Axis axis2){
    LOG(INFO)<<"join ("<<axis1.var->name_hint()<<", "<<axis1.dim<<") and ("<<axis2.var->name_hint()<<", "<<axis2.dim<<")";
    Axis parent1 = FindParent(axis1);
    Axis parent2 = FindParent(axis2);
    if(parent1 == parent2){
      return;
    }
    if (axis2group_.count(parent1)){
      parent_[parent2] = parent1;
      if(axis2group_.count(parent2)){
        axis2group_[parent1].insert(axis2group_[parent2].begin(), axis2group_[parent2].end());
      } else {
        axis2group_[parent1].insert(parent2);
      }
    } else if(axis2group_.count(parent2)){
      parent_[parent1] = parent2;
      axis2group_[parent2].insert(parent1);
    } else {
      parent_[parent1] = parent2;
      axis2group_[parent2] = AxisGroup{parent1, parent2};
    }
  }

  AxisGroup GetAxisGroup(Axis axis){
    Axis parent = FindParent(axis);
    if(axis2group_.count(parent)){
      return axis2group_[parent];
    } else {
      return {};
    }
  }


  private:
  Axis FindParent(Axis axis){
    if(parent_.count(axis)){
      axis = parent_[axis] = FindParent(parent_[axis]);
    }
    return axis;
  }

  //union set
  std::unordered_map<Axis, Axis, AxisHash> parent_;
  std::unordered_map<Axis, AxisGroup, AxisHash> axis2group_;

  
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

}  // namespace distributed
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_DISTRIBUTED_AXIS_GROUP_GRAPH_H_