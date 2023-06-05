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

/*!
 * \file tvm/relax/distributed/transform/propagate_sharding.cc
 * \brief Pass for propagating sharding information.
 */
#include <numeric>

#include <tvm/relax/analysis.h>
#include <tvm/relax/distributed/transform.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/distributed/axis_group_graph.h>
#include <tvm/relax/distributed/utils.h>
#include <tvm/relax/attrs/manipulate.h>
#include <tvm/relax/attrs/linear_algebra.h>
#include <tvm/relax/attrs/statistical.h>
#include <tvm/relax/attrs/distributed.h>

#include "../../op/distributed/distributed.h"

namespace tvm {
namespace relax {
namespace distributed {

void CollectAxisGraphBinary(const VarBindingNode* binding, const CallNode* call, AxisGroupGraph* axis_group_graph){
  const std::vector<std::string> binary_op_names = {
      "add",   "subtract", "multiply",      "divide", "power", "floor_divide",
      "equal", "greater",  "greater_equal", "less",   "less_equal", "not_equal", "minimum", "maximum"};
  for(const auto& op_name: binary_op_names){
    static const Op& binary_op = Op::Get("relax."+op_name);
    if(call->op.same_as(binary_op)){
      Array<Var> var_list; //vars in param and output
      if(call->args[0]->IsInstance<VarNode>()){
        var_list.push_back(Downcast<Var>(call->args[0]));
      }
      if(call->args[1]->IsInstance<VarNode>()){
        var_list.push_back(Downcast<Var>(call->args[1]));
      }
      var_list.push_back(binding->var);
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
  }
}

void CollectAxisGraphUnary(const VarBindingNode* binding, const CallNode* call, AxisGroupGraph* axis_group_graph){
  const std::vector<std::string> unary_op_names = {"abs", "acos", "acosh", "asin", "asinh", "atan",
                                                   "atanh", "ceil", "cos", "cosh", "exp", 
                                                   "floor", "log", "negative", "relu",
                                                   "round", "rqrt", "sigmoid", "sign", "sin", "sinh", "square", "sqrt",
                                                   "tan", "tanh","clip", "isfinite","isinf","isnan", "annotate_sharding","gelu"};
  for(const auto& op_name: unary_op_names){
    static const Op& unary_op = Op::Get("relax."+op_name);
    if(call->op.same_as(unary_op)){
      Array<Var> var_list; //vars in param and output
      if(call->args[0]->IsInstance<VarNode>()){
        var_list.push_back(Downcast<Var>(call->args[0]));
      }
      var_list.push_back(binding->var);
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
  }
}

void CollectAxisGraphReduction(const VarBindingNode* binding, const CallNode* call, AxisGroupGraph* axis_group_graph){
  const std::vector<std::string> reduction_op_names = {"sum", "max", "min", "prod", "mean", "std", "variance"};
  for(const auto& op_name: reduction_op_names){
    static const Op& reduction_op = Op::Get("relax."+op_name);
    if(call->op.same_as(reduction_op)){
      ICHECK(call->args[0]->IsInstance<VarNode>());
      Var input_var = Downcast<Var>(call->args[0]);
      Var output_var = binding->var;
      const auto* attrs = call->attrs.as<StatisticalAttrs>();
      ICHECK(attrs);
      int ndim = GetStructInfoAs<TensorStructInfoNode>(input_var)->ndim;

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
        } else{
          for (int i = 0, j = 0; i < ndim;i++){
            if(!normalized_axes.count(i)){
              axis_group_graph->JoinAxis({input_var.get(), i}, {output_var.get(), j});
              j++;
            }
          } 
        }
      }
    }
  }
}

void CollectAxisGraphMatmul(const VarBindingNode* binding, const CallNode* call, AxisGroupGraph* axis_group_graph){
  static const Op& matmul_op = Op::Get("relax.matmul");
  if(call->op.same_as(matmul_op)){
    Var x1 = Downcast<Var>(call->args[0]);
    Var x2 = Downcast<Var>(call->args[1]);
    Var x3 = binding->var;
    const auto* x1_sinfo = GetStructInfoAs<TensorStructInfoNode>(x1);
    const auto* x2_sinfo = GetStructInfoAs<TensorStructInfoNode>(x2);
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
      if (analyzer.CanProveEqual(dim0, dim1)) {
        //join batch dim
        axis_group_graph->JoinAxis({x1.get(), x1_prefix_ndim - i}, {x2.get(), x2_prefix_ndim - i});
        axis_group_graph->JoinAxis({x1.get(), x1_prefix_ndim - i}, {x3.get(), std::max(x1_prefix_ndim, x2_prefix_ndim) - i});
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
}

void CollectAxisGraphPermuteDims(const VarBindingNode* binding, const CallNode* call, AxisGroupGraph* axis_group_graph){
  static const Op& permute_dims_op = Op::Get("relax.permute_dims");
  if(call->op.same_as(permute_dims_op)){
    Var input_var = Downcast<Var>(call->args[0]);
    Var output_var = binding->var;
    const auto* attrs = call->attrs.as<PermuteDimsAttrs>();
    ICHECK(attrs);
    int ndim = GetStructInfoAs<TensorStructInfoNode>(input_var)->ndim;
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
}

void CollectAxisGraphForDeviceMesh(const VarBindingNode* binding, const CallNode* call, AxisGroupGraph* axis_group_graph){
  Array<Var> var_list;
  for (const auto& arg : call->args) {
    if(arg.as<VarNode>()){
      var_list.push_back(Downcast<Var>(arg));
    }
  }
  var_list.push_back(binding->var);
  for (int i = 0; i < var_list.size() - 1; i++) {
    axis_group_graph->JoinAxis({var_list[i].get(),-1}, {var_list[i + 1].get(), -1});
  }
}

class AxisGroupGraphBuilder: public ExprVisitor{

  public:
  static void BuildAxisGroupGraph(AxisGroupGraph* axis_group_graph, const Function& func){
    AxisGroupGraphBuilder builder(axis_group_graph);
    builder.VisitExpr(func);
  }

  private:
  explicit AxisGroupGraphBuilder(AxisGroupGraph* axis_group_graph): axis_group_graph_(axis_group_graph){}
    
  void VisitBinding_(const VarBindingNode* binding, const CallNode* val){
    CollectAxisGraphBinary(binding, val, axis_group_graph_);
    CollectAxisGraphUnary(binding, val, axis_group_graph_);
    CollectAxisGraphReduction(binding, val, axis_group_graph_);
    CollectAxisGraphMatmul(binding, val, axis_group_graph_);
    CollectAxisGraphPermuteDims(binding, val, axis_group_graph_);
    CollectAxisGraphForDeviceMesh(binding, val, axis_group_graph_);
    ExprVisitor::VisitBinding_(binding, val);
  }
  AxisGroupGraph* axis_group_graph_;
  
};



class ShardingAnnotationCollector : public ExprVisitor {
  public: 
  static void CollectShardingAnnotation(AxisGroupGraph* axis_group_graph, AxisGroupToShardingPlanSetMap* axis_group_to_sharding_plan, const Function& func){
    ShardingAnnotationCollector collector(axis_group_graph, axis_group_to_sharding_plan);
    collector.VisitExpr(func);
  }

  private:
  ShardingAnnotationCollector(AxisGroupGraph* axis_group_graph, AxisGroupToShardingPlanSetMap* axis_group_to_distribution_plan): axis_group_graph_(axis_group_graph), axis_group_to_sharding_plan_set_(axis_group_to_distribution_plan){}
  void VisitBinding_(const VarBindingNode* binding, const CallNode* val){
    static const Op& annotate_sharding_op = Op::Get("relax.annotate_sharding");
    if(val->op.same_as(annotate_sharding_op)){
      const auto* attrs = val->attrs.as<DistributionAttrs>();
      ICHECK(attrs);
      for(const PlacementSpec& placement_spec: attrs->placement->dim_specs){
        if(placement_spec->kind == PlacementSpecKind::kSharding){
          AxisGroup axis_group = axis_group_graph_->GetAxisGroup({binding->var.get(), placement_spec->axis});
          AddShardingPlan(axis_group, {attrs->device_mesh, placement_spec});
        }
      }

      //FIXME(hongyi): should represent src nodes in axis group graph, and cutting graph will reflect on the device mesh choice of unassigned tensors
      AxisGroup device_mesh_axis_group = axis_group_graph_->GetAxisGroup({binding->var.get(), -1});
      AddShardingPlan(device_mesh_axis_group, {attrs->device_mesh, PlacementSpec::Replica()});
    }
    ExprVisitor::VisitBinding_(binding, val);
  }

  void AddShardingPlan(const AxisGroup& axis_group, const AxisShardingPlan& sharding_plan){
    if(!axis_group_to_sharding_plan_set_->count(axis_group)){
      (*axis_group_to_sharding_plan_set_)[axis_group] = {};
    }
    (*axis_group_to_sharding_plan_set_)[axis_group].insert(sharding_plan);

  }

  AxisGroupGraph* axis_group_graph_;
  AxisGroupToShardingPlanSetMap* axis_group_to_sharding_plan_set_;
};

class ShardingConflictHandler: public ExprVisitor{

  public:
  static void HandleShardingConflict(AxisGroupGraph* axis_group_graph, AxisGroupToShardingPlanSetMap* axis_group_to_sharding_plan, Function function){
    ShardingConflictHandler handler(axis_group_graph, axis_group_to_sharding_plan);
    handler.VisitExpr(function);
    for(const Var& var: function->params){
      if(GetStructInfoAs<TensorStructInfoNode>(var)){
        handler.CheckTensorShardingCompatible(var);
      }
    }
  }

  private:
  ShardingConflictHandler(AxisGroupGraph* axis_group_graph, AxisGroupToShardingPlanSetMap* axis_group_to_sharding_plan): axis_group_graph_(axis_group_graph), axis_group_to_sharding_plan_set_(axis_group_to_sharding_plan){}

  void CheckTensorShardingCompatible(Var var){
    const auto* sinfo = GetStructInfoAs<TensorStructInfoNode>(var);
    ICHECK(sinfo);
    int ndim = sinfo->ndim;
    std::unordered_set<int> sharded_mesh_dim;
    Optional<DeviceMesh> device_mesh;
    for (int i = -1; i < ndim; i++) {
      AxisGroup axis_group = axis_group_graph_->GetAxisGroup({var.get(), i});
      if(axis_group_to_sharding_plan_set_->count(axis_group)){
        AxisShardingPlanSet sharding_plan_set = axis_group_to_sharding_plan_set_->at(axis_group);
        for(const auto& sharding_plan: sharding_plan_set){
          LOG(INFO) << sharding_plan.first << Placement({sharding_plan.second});
        }
        if(sharding_plan_set.size() > 1){
          ICHECK(false) << "Sharding conflict detected for tensor " << var->name_hint() << ": Multiple sharding plans for axis " << i << ". Conflict Handling logic will be added in the future.";
        }
        AxisShardingPlan sharding_plan = *sharding_plan_set.begin();
        if(i>=0){
          const PlacementSpec& placement_spec = sharding_plan.second;
          ICHECK(placement_spec->kind == PlacementSpecKind::kSharding);
          ICHECK(sharded_mesh_dim.count(placement_spec->axis) == 0) << "Sharding conflict detected for tensor " << var->name_hint() << ": Replicate sharding axis " << placement_spec->axis
                                                                    << ". Conflict Handling logic will be added in the future.";
          sharded_mesh_dim.insert(placement_spec->axis);
        }
        if(device_mesh.defined()){
          ICHECK(StructuralEqual()(device_mesh.value(), sharding_plan.first)) << "Sharding conflict detected for tensor " << var->name_hint() << ": Device Mesh mismatch"
                                                                                              << ". Conflict Handling logic will be added in the future.";
        } else {
          device_mesh = sharding_plan.first;
        }
      }
    }
  }

  void VisitBinding_(const VarBindingNode* binding){
    if(GetStructInfoAs<TensorStructInfoNode>(binding->var)){
      CheckTensorShardingCompatible(binding->var);
    }
    ExprVisitor::VisitBinding_(binding);
  }
  
  AxisGroupGraph* axis_group_graph_;
  AxisGroupToShardingPlanSetMap* axis_group_to_sharding_plan_set_;
};

class DistributedIRBuilder: public ExprMutator{
  
  public: 
  explicit DistributedIRBuilder(const IRModule& module) : ExprMutator(module) {}

  IRModule BuildDistributedIR(){
    auto mod = builder_->GetContextIRModule();
    for(const auto& [gv, base_func] : mod->functions){
      const auto* func_ = base_func.as<FunctionNode>();
      if (func_ == nullptr) {
        continue;
      }
      Function func = RewriteFunction(GetRef<Function>(func_));
      builder_->UpdateFunction(gv, func);
    }
    return builder_->GetContextIRModule();
  }

  private:
  Var RewriteInputTensor(Var param){
    const auto* sinfo = GetStructInfoAs<TensorStructInfoNode>(param);
    int ndim = sinfo->ndim;
    DeviceMesh device_mesh;
    Array<PlacementSpec> placement_specs;
    for (int i = 0;i<ndim;i++){
      AxisGroup axis_group = axis_group_graph_.GetAxisGroup({param.get(), i});
      visited_axis_group_.insert(axis_group);
      if(axis_group_to_sharding_plan_set_.count(axis_group)){
        const PlacementSpec& placement_spec = axis_group_to_sharding_plan_set_.at(axis_group).begin()->second;
        ICHECK(placement_spec->kind == PlacementSpecKind::kSharding);
        placement_specs.push_back(placement_spec);
      }
    } 
    AxisGroup device_mesh_axis_group = axis_group_graph_.GetAxisGroup({param.get(), -1});
    device_mesh = axis_group_to_sharding_plan_set_.at(device_mesh_axis_group).begin()->first;
    Var new_param(param->name_hint(), DTensorStructInfo(GetRef<TensorStructInfo>(sinfo), device_mesh, Placement(placement_specs)));
    return new_param;
  }

  Function RewriteFunction(Function func){
    // Step 1. Construct AxisGroupGraph
    AxisGroupGraphBuilder::BuildAxisGroupGraph(&axis_group_graph_, func);
    // Step 2. Collect Sharding Annotation
    ShardingAnnotationCollector::CollectShardingAnnotation(&axis_group_graph_, &axis_group_to_sharding_plan_set_, func);
    // Step 3. Handle Sharding Conflict
    ShardingConflictHandler::HandleShardingConflict(&axis_group_graph_, &axis_group_to_sharding_plan_set_, func);
    // Step 4. Rewrite Function
    Array<Var> new_params;
    for(const Var& var: func->params){
      if(GetStructInfoAs<TensorStructInfoNode>(var)){
        Var new_param = RewriteInputTensor(var);
        input_tensor_remap_.Set(var, new_param);
        new_params.push_back(new_param);
      }else{
        new_params.push_back(var);
      }
    }
    auto new_body = VisitWithNewScope(func->body, new_params);
    Function new_func(new_params, new_body, func->ret_struct_info, func->is_pure, func->attrs);
    return new_func;
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* val){
    static const Op& annotate_sharding_op = Op::Get("relax.annotate_sharding");
    Call new_call = Downcast<Call>(this->VisitExpr(binding->value));
    if(val->op.same_as(annotate_sharding_op)){
      ReEmitBinding(binding, new_call->args[0]);
      return;
    }
    const auto* sinfo = GetStructInfoAs<TensorStructInfoNode>(binding->var);
    int ndim = sinfo->ndim;
    for (int i = 0; i < ndim;i++){
      AxisGroup axis_group = axis_group_graph_.GetAxisGroup({binding->var.get(), i});
      Array<PlacementSpec> placement_specs;
      bool insert_redistribute = false;
      DeviceMesh device_mesh;
      if (axis_group_to_sharding_plan_set_.count(axis_group)) {
        if(!visited_axis_group_.count(axis_group)){
          insert_redistribute = true;
          visited_axis_group_.insert(axis_group);
        } 
        AxisShardingPlan sharding_plan = *axis_group_to_sharding_plan_set_.at(axis_group).begin();
        device_mesh = sharding_plan.first;
        placement_specs.push_back(sharding_plan.second);
      } else {
        placement_specs.push_back(PlacementSpec::Replica());
      }
      if(insert_redistribute){
        Expr new_value= redistribute(new_call, device_mesh, Placement(placement_specs));
        new_value = builder_->Normalize(new_value);
        ReEmitBinding(binding, new_value);
      } else {
        ReEmitBinding(binding, new_call);
      }
    } 
  }

  Expr VisitExpr_(const VarNode* var) final {
    auto it = input_tensor_remap_.find(GetRef<Var>(var));
    if (it != input_tensor_remap_.end()) {
      return (*it).second;
    }
    return ExprMutator::VisitExpr_(var);
  }

  Map<Var, Var> input_tensor_remap_;
  AxisGroupGraph axis_group_graph_;
  AxisGroupToShardingPlanSetMap axis_group_to_sharding_plan_set_;
  std::unordered_set<AxisGroup, AxisGroupHash> visited_axis_group_;
};
namespace transform {

Pass PropagateSharding() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return DistributedIRBuilder(m).BuildDistributedIR(); };
  return CreateModulePass(pass_func, 1, "PropagateSharding", {});
}
TVM_REGISTER_GLOBAL("relax.distributed.transform.PropagateSharding").set_body_typed(PropagateSharding);
}  // namespace transform

}  // namespace distributed
}  // namespace relax
}  // namespace tvm