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

#ifndef TVM_RUNTIME_DISCO_CUDA_IPC_IPC_ALLOCATOR_H_
#define TVM_RUNTIME_DISCO_CUDA_IPC_IPC_ALLOCATOR_H_

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/memory/memory_manager.h>

#include "../../memory/pooled_allocator.h"
#include "../nccl/disco_nccl.h"

namespace tvm {
namespace runtime {
namespace cuda_ipc {



using tvm::runtime::memory::Buffer;

class CUDAIPCAllocator final: public memory::PooledAllocator {
  public:
  explicit CUDAIPCAllocator(Device dev, size_t page_size = kDefaultPageSize): PooledAllocator(dev, page_size) {
    ICHECK(dev.device_type == kDLCUDA);
    CUDA_CALL(cudaStreamCreateWithFlags(&cpu_comm_stream_, cudaStreamNonBlocking));
  }

  Buffer Alloc(size_t nbytes, size_t alignment, DLDataType type_hint) override {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    size_t size = ((nbytes + page_size_ - 1) / page_size_) * page_size_;
    auto&& it = memory_pool_.find(size);
    if (it != memory_pool_.end() && !it->second.empty()) {
    auto&& pool = it->second;
    auto ret = pool.back();
    pool.pop_back();
    return ret;
    }
    Buffer buf;
    buf.device = device_;
    buf.size = size;
    buf.alloc_type = memory::kPooled;
    try {
    buf.data = AllocDataSpace(device_, size, alignment, type_hint);
    } catch (InternalError& err) {
    LOG(WARNING) << "PooledAllocator got InternalError during allocation: " << err.message();
    LOG(WARNING) << "Trying to release all unused memory and reallocate...";
    ReleaseAll();
    buf.data = AllocDataSpace(device_, size, alignment, type_hint);
    }

    used_memory_.fetch_add(size, std::memory_order_relaxed);
    VLOG(1) << "allocate " << size << " B, used memory " << used_memory_ << " B";
    return buf;
  }

  Buffer Alloc(ShapeTuple shape, DLDataType type_hint, const std::string& mem_scope) override {
    if (mem_scope.empty() || mem_scope == "cuda_ipc") {
      NDArray::Container container(nullptr, shape, type_hint, device_);
      size_t size = DeviceAPI::Get(device_)->GetDataSize(container.dl_tensor);
      size_t alignment = GetDataAlignment(container.dl_tensor);
      return Alloc(size, alignment, type_hint);
    }
    LOG(FATAL) << "This alloc should be implemented";
    return {};
  }

  std::vector<void*> GetIPCRemoteMemPtr(void* ptr) {
    ICHECK(ipc_remote_mem.count(ptr));
    return ipc_remote_mem.at(ptr);
  }

  private:

  inline size_t GetDataAlignment(const DLTensor& arr) {
    size_t align = (arr.dtype.bits / 8) * arr.dtype.lanes;
    if (align < kAllocAlignment) return kAllocAlignment;
    return align;
  }

  void ReleaseAll() override{
    std::lock_guard<std::recursive_mutex> lock(mu_);
    for (auto const& it : memory_pool_) {
      auto const& pool = it.second;
      for (auto const& buf : pool) {
       FreeDataSpace(buf.device, buf.data);
      }
    }
    memory_pool_.clear();
    used_memory_ = 0;
    VLOG(1) << "release all buffers";
  }

  void* AllocDataSpace(Device dev, size_t size, size_t alignment, DLDataType type_hint) {
    // alloc local buffer
    ICHECK(dev.device_type == kDLCUDA);
    void* ptr;
    CUDA_CALL(cudaSetDevice(dev.device_id));
    CUDA_CALL(cudaMalloc(&ptr, size));
    // create ipc handle
    cudaIpcMemHandle_t localHandle;
    CUDA_CALL(cudaIpcGetMemHandle(&localHandle, ptr));
    // all gather ipc handle
    nccl::CCLThreadLocalContext* ctx = nccl::CCLThreadLocalContext::Get();
    void *d_src, *d_dst;
    CUDA_CALL(cudaMalloc(&d_src, CUDA_IPC_HANDLE_SIZE));
    CUDA_CALL(cudaMalloc(&d_dst, CUDA_IPC_HANDLE_SIZE * ctx->worker->num_workers));
    CUDA_CALL(cudaMemcpyAsync(d_src, &localHandle, CUDA_IPC_HANDLE_SIZE, cudaMemcpyDefault, cpu_comm_stream_));
    NCCL_CALL(ncclAllGather(d_src, d_dst, CUDA_IPC_HANDLE_SIZE, ncclChar, ctx->comm, cpu_comm_stream_));
    CUDA_CALL(cudaStreamSynchronize(cpu_comm_stream_));
    std::vector<char> serialHandles(CUDA_IPC_HANDLE_SIZE * ctx->worker->num_workers, 0);
    CUDA_CALL(cudaMemcpy(serialHandles.data(), d_dst, CUDA_IPC_HANDLE_SIZE * ctx->worker->num_workers, cudaMemcpyDefault));
    std::vector<cudaIpcMemHandle_t> handles(ctx->worker->num_workers);
    for(int i = 0; i < ctx->worker->num_workers; i++) {
      memcpy(handles[i].reserved, &serialHandles[i * CUDA_IPC_HANDLE_SIZE], CUDA_IPC_HANDLE_SIZE);
    }
    std::vector<void*> mCommPtrs(ctx->worker->num_workers);
    for (size_t nodeId = 0; nodeId < handles.size(); nodeId++){
      if ((int) nodeId == ctx->worker->worker_id){
          mCommPtrs[nodeId] = ptr;
      }
      else {
        uint8_t* foreignBuffer;
        CUDA_CALL(cudaIpcOpenMemHandle(
            reinterpret_cast<void**>(&foreignBuffer), handles[nodeId], cudaIpcMemLazyEnablePeerAccess));
        mCommPtrs[nodeId] = foreignBuffer;
      }
    }
    CUDA_CALL(cudaFree(d_src));
    CUDA_CALL(cudaFree(d_dst));
    ipc_remote_mem[ptr] = mCommPtrs;
    return ptr;
  }

  void FreeDataSpace(Device dev, void* ptr) {
    ICHECK(dev.device_type == kDLCUDA);
    CUDA_CALL(cudaSetDevice(dev.device_id));
    //cpu barrier
    nccl::CCLThreadLocalContext* ctx = nccl::CCLThreadLocalContext::Get();
    void* d_src, *d_dst;
    CUDA_CALL(cudaMalloc(&d_src, sizeof(int)));
    CUDA_CALL(cudaMalloc(&d_dst, sizeof(int)));
    NCCL_CALL(ncclAllReduce(d_src, d_dst, 1, ncclInt, ncclSum, ctx->comm, cpu_comm_stream_));
    CUDA_CALL(cudaStreamSynchronize(cpu_comm_stream_));
    CUDA_CALL(cudaFree(d_src));
    CUDA_CALL(cudaFree(d_dst));
    // free local buffer
    CUDA_CALL(cudaFree(ptr));
    // free ipc handle
    for (int i = 0; i< static_cast<int>(ipc_remote_mem[ptr].size()); i++){
      if(i != ctx->worker->worker_id){
        CUDA_CALL(cudaIpcCloseMemHandle(ipc_remote_mem[ptr][i]));
      }
    }
    ipc_remote_mem.erase(ptr);
  }

  private:
   cudaStream_t cpu_comm_stream_;
   std::unordered_map<void*, std::vector<void*>> ipc_remote_mem;

};

extern CUDAIPCAllocator* ipc_alloc;

}  // namespace cuda_ipc
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_DISCO_CUDA_IPC_IPC_ALLOCATOR_H_