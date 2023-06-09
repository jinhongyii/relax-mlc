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
#include <numeric>

#include <tvm/relax/distributed/axis_group_graph.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/manipulate.h>
#include <tvm/relax/attrs/linear_algebra.h>
#include <tvm/relax/attrs/statistical.h>
#include <tvm/relax/attrs/distributed.h>

namespace tvm {
namespace relax {
namespace distributed {

const TensorStructInfoNode* GetTensorStructInfo(Var var){
  const auto* tensor_sinfo = GetStructInfoAs<TensorStructInfoNode>(var);
  if(tensor_sinfo){
    return tensor_sinfo;
  }
  const auto* dtensor_sinfo = GetStructInfoAs<DTensorStructInfoNode>(var);
  if(dtensor_sinfo){
    return dtensor_sinfo->tensor_sinfo.get();
  }
  LOG(FATAL)<<var->name_hint()<<" must be either Tensor or DTensor";
  throw;
}

void UnaryOpHelper(Array<Var> var_list,
                        distributed::AxisGroupGraph* axis_group_graph){
    int n_dim = GetTensorStructInfo(var_list[0])->ndim;
    for(const auto& var: var_list){
        ICHECK(GetTensorStructInfo(var)->ndim == n_dim);
    }
    for (int i = 0; i < n_dim;i++){
        for (int j = 0; j < var_list.size()-1;j++){
        axis_group_graph->JoinAxis({var_list[j].get(), i}, {var_list[j+1].get(), i});
        }
    }
}

void BuildAxisGraphUnary(const Var& output_var, const Call& call,
                        distributed::AxisGroupGraph* axis_group_graph){
  Array<Var> var_list; //vars in param and output
  if(call->args[0]->IsInstance<VarNode>()){
    var_list.push_back(Downcast<Var>(call->args[0]));
  }
  var_list.push_back(output_var);
  UnaryOpHelper(var_list, axis_group_graph);
}



void BuildAxisGraphBinary(const Var& output_var, const Call& call,
                           distributed::AxisGroupGraph* axis_group_graph){
                            Array<Var> var_list; //vars in param and output
  if(call->args[0]->IsInstance<VarNode>()){
    var_list.push_back(Downcast<Var>(call->args[0]));
  }
  if(call->args[1]->IsInstance<VarNode>()){
    var_list.push_back(Downcast<Var>(call->args[1]));
  }
  var_list.push_back(output_var);
  if(var_list.size()<=2){
    UnaryOpHelper(var_list, axis_group_graph);
    return;
  }
  const auto* x1_sinfo = GetTensorStructInfo(var_list[0]);
  const auto* x2_sinfo = GetTensorStructInfo(var_list[1]);
  int x1_ndim = x1_sinfo->ndim;
  int x2_ndim = x2_sinfo->ndim;
  const auto* x1_shape = x1_sinfo->shape.as<ShapeExprNode>();
  const auto* x2_shape = x2_sinfo->shape.as<ShapeExprNode>();
  ICHECK(x1_shape && x2_shape);
    arith::Analyzer analyzer;
    for (int i = 1; i <= std::min(x1_ndim, x2_ndim); ++i) {
      const PrimExpr& dim0 = x1_shape->values[x1_ndim - i];
      const PrimExpr& dim1 = x2_shape->values[x2_ndim - i];
      if (analyzer.CanProveEqual(dim0, dim1)) {
        //join batch dim
        axis_group_graph->JoinAxis({var_list[0].get(), x1_ndim - i}, {var_list[2].get(), std::max(x1_ndim, x2_ndim) - i});
        axis_group_graph->JoinAxis({var_list[1].get(), x2_ndim - i}, {var_list[2].get(), std::max(x1_ndim, x2_ndim) - i});
      } else if(analyzer.CanProveEqual(dim0, 1)){
        axis_group_graph->JoinAxis({var_list[1].get(), x2_ndim - i}, {var_list[2].get(), std::max(x1_ndim, x2_ndim) - i});
      } else if(analyzer.CanProveEqual(dim1, 1)){
        axis_group_graph->JoinAxis({var_list[0].get(), x1_ndim - i}, {var_list[2].get(), std::max(x1_ndim, x2_ndim) - i});
      } else {
        LOG(FATAL)<<"Invalid broadcast, dim0: "<< dim0 <<", dim1: "<<dim1;
      }
    }
  int n_dim = GetStructInfoAs<TensorStructInfoNode>(var_list[0])->ndim;
  for(const auto& var: var_list){
    ICHECK(GetStructInfoAs<TensorStructInfoNode>(var)->ndim == n_dim);
  }
  for (int i = 0; i < n_dim;i++){
    for (int j = 0; j < var_list.size()-1;j++){
      axis_group_graph->JoinAxis({var_list[j].get(), i}, {var_list[j+1].get(), i});
    }
  }
}

void BuildAxisGraphReduce(const Var& output_var, const Call& call,
                           distributed::AxisGroupGraph* axis_group_graph){
    ICHECK(call->args[0]->IsInstance<VarNode>());
    Var input_var = Downcast<Var>(call->args[0]);
    const auto* attrs = call->attrs.as<StatisticalAttrs>();
    ICHECK(attrs);
    int ndim = GetTensorStructInfo(input_var)->ndim;

    if(attrs->axis.defined()){
        std::unordered_set<int> normalized_axes;
        for (const Integer& i : attrs->axis.value()) {
            int val = i->value;
            ICHECK(val < ndim && val >= -ndim);
            if (val < 0) {
                val = ndim + val;
            }
            normalized_axes.insert(val);
        }
        if(attrs->keepdims){
            for (int i = 0; i < ndim;i++){
                if(!normalized_axes.count(i)){
                    axis_group_graph->JoinAxis({input_var.get(), i}, {output_var.get(), i});
                }
            }
        } else {
            for (int i = 0, j = 0; i < ndim;i++){
                if(!normalized_axes.count(i)){
                    axis_group_graph->JoinAxis({input_var.get(), i}, {output_var.get(), j});
                    j++;
                }
            } 
        }
    }
}

void BuildAxisGraphMatmul(const Var& output_var, const Call& call,
                           distributed::AxisGroupGraph* axis_group_graph){
    Var x1 = Downcast<Var>(call->args[0]);
    Var x2 = Downcast<Var>(call->args[1]);
    Var x3 = output_var;
    const auto* x1_sinfo = GetTensorStructInfo(x1);
    const auto* x2_sinfo = GetTensorStructInfo(x2);
    int x1_ndim = x1_sinfo->ndim;
    int x2_ndim = x2_sinfo->ndim;
    ICHECK(x1_ndim > 0 && x2_ndim > 0);
    int x1_prepended = 0;
    int x2_appended = 0;
    if (x1_ndim == 1) {
      x1_ndim = 2;
      x1_prepended = 1;
    }
    if (x2_ndim == 1) {
      x2_ndim = 2;
      x2_appended = 1;
    }
    const auto* x1_shape = x1_sinfo->shape.as<ShapeExprNode>();
    const auto* x2_shape = x2_sinfo->shape.as<ShapeExprNode>();
    ICHECK(x1_shape && x2_shape);
    Array<PrimExpr> x1_shape_prefix{x1_shape->values.begin(),
                                    x1_shape->values.end() - 2 + x1_prepended};
    Array<PrimExpr> x2_shape_prefix{x2_shape->values.begin(),
                                    x2_shape->values.end() - 2 + x2_appended};
    
    int x1_prefix_ndim = x1_shape_prefix.size();
    int x2_prefix_ndim = x2_shape_prefix.size();
    arith::Analyzer analyzer;
    for (int i = 1; i <= std::min(x1_prefix_ndim, x2_prefix_ndim); ++i) {
      const PrimExpr& dim0 = x1_shape_prefix[x1_prefix_ndim - i];
      const PrimExpr& dim1 = x2_shape_prefix[x2_prefix_ndim - i];
        //join batch dim
      if (analyzer.CanProveEqual(dim0, dim1)) {
        axis_group_graph->JoinAxis({x1.get(), x1_prefix_ndim - i}, {x3.get(), std::max(x1_prefix_ndim, x2_prefix_ndim) - i});
        axis_group_graph->JoinAxis({x2.get(), x2_prefix_ndim - i}, {x3.get(), std::max(x1_prefix_ndim, x2_prefix_ndim) - i});
      } else if (analyzer.CanProveEqual(dim0, 1)){
        axis_group_graph->JoinAxis({x2.get(), x2_prefix_ndim - i}, {x3.get(), std::max(x1_prefix_ndim, x2_prefix_ndim) - i});
      } else if (analyzer.CanProveEqual(dim1, 1)){
        axis_group_graph->JoinAxis({x1.get(), x1_prefix_ndim - i}, {x3.get(), std::max(x1_prefix_ndim, x2_prefix_ndim) - i});
      } else {
        LOG(FATAL) << "Cannot broadcast " << dim0 << " and " << dim1;
      }
    }
    //join reduction dim
    axis_group_graph->JoinAxis({x1.get(), x1_sinfo->ndim - 1}, {x2.get(), x2_ndim - 2});
    //join lhs_spatial dim and rhs_spatial dim
    if (!x1_prepended) {
      axis_group_graph->JoinAxis({x1.get(), x1_ndim - 2}, {x3.get(), std::max(x1_prefix_ndim, x2_prefix_ndim)});
      if(!x2_appended){
        axis_group_graph->JoinAxis({x2.get(), x2_ndim - 1}, {x3.get(), std::max(x1_prefix_ndim, x2_prefix_ndim) + 1});
      }
    } else if(!x2_appended){
      axis_group_graph->JoinAxis({x2.get(), x2_ndim - 1}, {x3.get(), std::max(x1_prefix_ndim, x2_prefix_ndim)});
    }
}

void BuildAxisGraphPermuteDims(const Var& output_var, const Call& call,
                           distributed::AxisGroupGraph* axis_group_graph){
    Var input_var = Downcast<Var>(call->args[0]);
    const auto* attrs = call->attrs.as<PermuteDimsAttrs>();
    ICHECK(attrs);
    int ndim = GetTensorStructInfo(input_var)->ndim;
    std::vector<int> normalized_axes;
    if(attrs->axes.defined()){
      for (const Integer& i : attrs->axes.value()) {
        int val = i->value;
        ICHECK(val < ndim && val >= -ndim);
        if (val < 0) {
          val = ndim + val;
        }
        normalized_axes.push_back(val);
      }
    } else {
      normalized_axes.resize(ndim);
      std::iota(normalized_axes.rbegin(), normalized_axes.rend(), 0);
    }
    for(int i = 0; i < ndim; i++){
      axis_group_graph->JoinAxis({input_var.get(), normalized_axes[i]}, {output_var.get(), i});
    }
}

} // namespace distributed
}  // namespace relax
}  // namespace tvm