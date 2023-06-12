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
 * \file tvm/relax/distributed/utils.h
 * \brief Utility for distributed Relax
 */

#ifndef TVM_RELAX_DISTRIBUTED_UTILS_H_
#define TVM_RELAX_DISTRIBUTED_UTILS_H_

#include <tvm/relax/distributed/axis_group_graph.h>
#include <tvm/relax/distributed/struct_info.h>

namespace tvm {
namespace relax {
namespace distributed {




using AxisShardingPlanSet = std::unordered_set<AxisShardingPlan, AxisShardingPlanHash, AxisShardingPlanEqual>;
using AxisGroupToShardingPlanSetMap = std::unordered_map<AxisGroup, AxisShardingPlanSet, AxisGroupHash>;


}  // namespace distributed
}  // namespace relax
}  // namespace tvm

#endif //TVM_RELAX_DISTRIBUTED_UTILS_H_