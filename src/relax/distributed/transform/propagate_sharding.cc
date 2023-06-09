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
    const Op& binary_op = Op::Get("relax."+op_name);
    if(call->op.same_as(binary_op)){
      BuildAxisGraphBinary(binding->var, GetRef<Call>(call), axis_group_graph);
      break;
    }
  }
}

void CollectAxisGraphUnary(const VarBindingNode* binding, const CallNode* call, AxisGroupGraph* axis_group_graph){
  const std::vector<std::string> unary_op_names = {"abs", "acos", "acosh", "asin", "asinh", "atan",
                                                   "atanh", "ceil", "cos", "cosh", "exp", 
                                                   "floor", "log", "negative", "nn.relu",
                                                   "round", "rsqrt", "sigmoid", "sign", "sin", "sinh", "square", "sqrt",
                                                   "tan", "tanh","clip", "isfinite","isinf","isnan", "distributed.annotate_sharding","nn.gelu"};
  for(const auto& op_name: unary_op_names){
    const Op& unary_op = Op::Get("relax."+op_name);
    if(call->op.same_as(unary_op)){
      LOG(INFO) << "match unary op " << op_name;
      BuildAxisGraphUnary(binding->var, GetRef<Call>(call), axis_group_graph);
    }
  }
}

void CollectAxisGraphReduce(const VarBindingNode* binding, const CallNode* call, AxisGroupGraph* axis_group_graph){
  const std::vector<std::string> reduction_op_names = {"sum", "max", "min", "prod", "mean", "std", "variance"};
  for(const auto& op_name: reduction_op_names){
    const Op& reduction_op = Op::Get("relax."+op_name);
    if(call->op.same_as(reduction_op)){
      BuildAxisGraphReduce(binding->var, GetRef<Call>(call), axis_group_graph);
      break;
    }
  }
}

void CollectAxisGraphMatmul(const VarBindingNode* binding, const CallNode* call, AxisGroupGraph* axis_group_graph){
  static const Op& matmul_op = Op::Get("relax.matmul");
  if(call->op.same_as(matmul_op)){
    BuildAxisGraphMatmul(binding->var, GetRef<Call>(call), axis_group_graph);
  }
}

void CollectAxisGraphPermuteDims(const VarBindingNode* binding, const CallNode* call, AxisGroupGraph* axis_group_graph){
  static const Op& permute_dims_op = Op::Get("relax.permute_dims");
  if(call->op.same_as(permute_dims_op)){
    BuildAxisGraphPermuteDims(binding->var, GetRef<Call>(call), axis_group_graph);
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
    CollectAxisGraphReduce(binding, val, axis_group_graph_);
    CollectAxisGraphMatmul(binding, val, axis_group_graph_);
    CollectAxisGraphPermuteDims(binding, val, axis_group_graph_);
    CollectAxisGraphForDeviceMesh(binding, val, axis_group_graph_);
    ExprVisitor::VisitBinding_(binding, val);
  }
  AxisGroupGraph* axis_group_graph_;
  
};


//todo: wrong understanding on placement definition. need to modify the logic
class ShardingAnnotationCollector : public ExprVisitor {
  public: 
  static void CollectShardingAnnotation(AxisGroupGraph* axis_group_graph, AxisGroupToShardingPlanSetMap* axis_group_to_sharding_plan, const Function& func){
    ShardingAnnotationCollector collector(axis_group_graph, axis_group_to_sharding_plan);
    collector.VisitExpr(func);
  }

  private:
  ShardingAnnotationCollector(AxisGroupGraph* axis_group_graph, AxisGroupToShardingPlanSetMap* axis_group_to_distribution_plan): axis_group_graph_(axis_group_graph), axis_group_to_sharding_plan_set_(axis_group_to_distribution_plan){}
  void VisitBinding_(const VarBindingNode* binding, const CallNode* val){
    static const Op& annotate_sharding_op = Op::Get("relax.distributed.annotate_sharding");
    if(val->op.same_as(annotate_sharding_op)){
      const auto* attrs = val->attrs.as<DistributionAttrs>();
      ICHECK(attrs);

      for (int i = 0;i < attrs->placement->dim_specs.size(); i++) {
        const PlacementSpec& placement_spec = attrs->placement->dim_specs[i];
        if (placement_spec->kind == PlacementSpecKind::kSharding) {
          AxisGroup axis_group = axis_group_graph_->GetAxisGroup({binding->var.get(), placement_spec->axis});
          LOG(INFO)<<"add sharding plan: "<<i<<", axis group:"<<binding->var->name_hint()<<", "<<placement_spec->axis;
          AddShardingPlan(axis_group, {attrs->device_mesh, i});
        }
      }

      //FIXME(hongyi): should represent src nodes in axis group graph, and cutting graph will reflect on the device mesh choice of unassigned tensors
      AxisGroup device_mesh_axis_group = axis_group_graph_->GetAxisGroup({binding->var.get(), -1});
      AddShardingPlan(device_mesh_axis_group, {attrs->device_mesh, -1});
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
          LOG(INFO) << sharding_plan.first << sharding_plan.second;
        }
        if(sharding_plan_set.size() > 1){
          ICHECK(false) << "Sharding conflict detected for tensor " << var->name_hint() << ": Multiple sharding plans for tensor axis " << i << ". Conflict Handling logic will be added in the future.";
        }
        AxisShardingPlan sharding_plan = *sharding_plan_set.begin();
        if(i>=0){
          int sharding_dim = sharding_plan.second;
          ICHECK(sharded_mesh_dim.count(sharding_dim) == 0) << "Sharding conflict detected for tensor " << var->name_hint() << ": Replicate sharding device mesh axis " << sharding_dim
                                                                    << ". Conflict Handling logic will be added in the future.";
          sharded_mesh_dim.insert(sharding_dim);
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
    AxisGroup device_mesh_axis_group = axis_group_graph_.GetAxisGroup({param.get(), -1});
    DeviceMesh device_mesh = axis_group_to_sharding_plan_set_.at(device_mesh_axis_group).begin()->first;
    Array<PlacementSpec> placement_specs(std::vector<PlacementSpec>(device_mesh->shape.size(), PlacementSpec::Replica()));
    for (int i = 0;i<ndim;i++){
      AxisGroup axis_group = axis_group_graph_.GetAxisGroup({param.get(), i});
      visited_axis_group_.insert(axis_group);
      if(axis_group_to_sharding_plan_set_.count(axis_group)){
        int sharding_dim = axis_group_to_sharding_plan_set_.at(axis_group).begin()->second;
        placement_specs.Set(sharding_dim, PlacementSpec::Sharding(i));
      }
    } 
    Var new_param(param->name_hint(), DTensorStructInfo(GetRef<TensorStructInfo>(sinfo), device_mesh, Placement(placement_specs)));
    return new_param;
  }

  Function RewriteFunction(Function func){
    // Step 1. Construct AxisGroupGraph
    AxisGroupGraphBuilder::BuildAxisGroupGraph(&axis_group_graph_, func);
    LOG(INFO) << "step 1 complete";
    // Step 2. Collect Sharding Annotation
    ShardingAnnotationCollector::CollectShardingAnnotation(&axis_group_graph_, &axis_group_to_sharding_plan_set_, func);
    LOG(INFO)<<"step 2 complete";
    // Step 3. Handle Sharding Conflict
    ShardingConflictHandler::HandleShardingConflict(&axis_group_graph_, &axis_group_to_sharding_plan_set_, func);
    LOG(INFO)<<"step 3 complete";
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
    LOG(INFO) << "rewrite all input tensors";
    auto new_body = VisitWithNewScope(func->body, new_params);
    LOG(INFO) << "visiting complete";
    Function new_func(new_params, new_body, func->ret_struct_info, func->is_pure, func->attrs);
    return new_func;
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* val){
    LOG(INFO) << "visit " << binding->var;
    static const Op& annotate_sharding_op = Op::Get("relax.distributed.annotate_sharding");
    Call new_call = Downcast<Call>(this->VisitExpr(binding->value));
    if(val->op.same_as(annotate_sharding_op)){
      ReEmitBinding(binding, new_call->args[0]);
      return;
    }
    const auto* sinfo = GetStructInfoAs<TensorStructInfoNode>(binding->var);
    int ndim = sinfo->ndim;
    bool insert_redistribute = false;
    AxisGroup device_mesh_axis_group = axis_group_graph_.GetAxisGroup({binding->var.get(), -1});
    DeviceMesh device_mesh = axis_group_to_sharding_plan_set_.at(device_mesh_axis_group).begin()->first;
    Array<PlacementSpec> placement_specs(std::vector<PlacementSpec>(device_mesh->shape.size(), PlacementSpec::Replica()));
    for (int i = 0; i < ndim;i++){
      AxisGroup axis_group = axis_group_graph_.GetAxisGroup({binding->var.get(), i});
      if (axis_group_to_sharding_plan_set_.count(axis_group)) {
        if(!visited_axis_group_.count(axis_group)){
          insert_redistribute = true;
          visited_axis_group_.insert(axis_group);
        } 
        AxisShardingPlan sharding_plan = *axis_group_to_sharding_plan_set_.at(axis_group).begin();
        placement_specs.Set(sharding_plan.second, PlacementSpec::Sharding(i));
      }
    } 
    LOG(INFO) << "insert redistribute:" << insert_redistribute;
    if(insert_redistribute){
      Expr new_value= redistribute(new_call, device_mesh, Placement(placement_specs));
      new_value = builder_->Normalize(new_value);
      ReEmitBinding(binding, new_value);
    } else {
      ReEmitBinding(binding, new_call);
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