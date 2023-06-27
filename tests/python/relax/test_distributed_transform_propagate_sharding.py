#  type: ignore

from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
import tvm
from tvm import relax

@I.ir_module
class TestModule:
    I.module_attrs({"device_num": 10}) 
    I.module_global_infos(
        {
            "mesh": [
                R.device_mesh((2,), I.Range(0, 2)),  # mesh[0] 
                R.device_mesh((1,), I.Range(4, 5)),  # mesh[1] 
            ]
        }
    )
    
    @R.function
    def foo(
        x: R.Tensor((128, 128), "float32"), weight1: R.Tensor((128, 128), "float32"), weight2: R.Tensor((128, 128), "float32")
    ) -> R.Tensor((128, 128), "float32"): 
        lv0 = R.matmul(x, weight1)
        lv1 = R.nn.gelu(lv0)
        lv2 = R.annotate_sharding(lv1, device_mesh="mesh[0]", placement="S[1]" )
        lv3 = R.matmul(lv2, weight2)
        return lv3
    
@I.ir_module
class LlamaDecoderLayer:
    I.module_attrs({"device_num": 10}) 
    I.module_global_infos(
        {
            "mesh": [
                R.device_mesh((2,), I.Range(0, 2)),  # mesh[0] 
                R.device_mesh((1,), I.Range(4, 5)),  # mesh[1] 
            ]
        }
    )
    
    @T.prim_func
    def min_max_triu_te(var_make_diag_mask_te: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        
        make_diag_mask_te = T.match_buffer(var_make_diag_mask_te, (256, 256), "float16")
        # with T.block("root"):
        for i, j in T.grid(256, 256):
            with T.block("make_diag_mask_te"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads()
                T.writes(make_diag_mask_te[v_i, v_j])
                make_diag_mask_te[v_i, v_j] = T.Select(v_i < v_j, T.float16(-65504), T.float16(65504))
                
    @T.prim_func
    def extend_te(var_A: T.handle, var_concat_te: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        
        A = T.match_buffer(var_A, (T.int64(1), T.int64(1), 256, 256), "float16")
        
        concat_te = T.match_buffer(var_concat_te, (T.int64(1), T.int64(1), 256, 256), "float16")
        # with T.block("root"):
        for b, _, i, j in T.grid(T.int64(1), T.int64(1), 256, 256):
            with T.block("concat_te"):
                v_b, v__, v_i, v_j = T.axis.remap("SSSS", [b, _, i, j])
                T.reads(A[v_b, v__, v_i, v_j + 256 - 256])
                T.writes(concat_te[v_b, v__, v_i, v_j])
                concat_te[v_b, v__, v_i, v_j] = T.if_then_else(v_j < 256 - 256, T.float16(65504), A[v_b, v__, v_i, v_j + 256 - 256])
                
    @T.prim_func
    def rms_norm(var_A: T.handle, B: T.Buffer((T.int64(4096),), "float16"), var_rms_norm: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        
        A = T.match_buffer(var_A, (T.int64(1), 256, T.int64(4096)), "float16")
        rms_norm_1 = T.match_buffer(var_rms_norm, (T.int64(1), 256, T.int64(4096)), "float16")
        # with T.block("root"):
        Ared_temp = T.alloc_buffer((T.int64(1), 256))
        for bsz, i, k in T.grid(T.int64(1), 256, T.int64(4096)):
            with T.block("Ared_temp"):
                v_bsz, v_i, v_k = T.axis.remap("SSR", [bsz, i, k])
                T.reads(A[v_bsz, v_i, v_k])
                T.writes(Ared_temp[v_bsz, v_i])
                with T.init():
                    Ared_temp[v_bsz, v_i] = T.float32(0)
                Ared_temp[v_bsz, v_i] = Ared_temp[v_bsz, v_i] + T.Cast("float32", A[v_bsz, v_i, v_k]) * T.Cast("float32", A[v_bsz, v_i, v_k])
        for bsz, i, k in T.grid(T.int64(1), 256, T.int64(4096)):
            with T.block("rms_norm"):
                v_bsz, v_i, v_k = T.axis.remap("SSS", [bsz, i, k])
                T.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp[v_bsz, v_i])
                T.writes(rms_norm_1[v_bsz, v_i, v_k])
                rms_norm_1[v_bsz, v_i, v_k] = T.Cast("float16", T.Cast("float32", B[v_k]) * (T.Cast("float32", A[v_bsz, v_i, v_k]) / T.sqrt(Ared_temp[v_bsz, v_i] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))))
                
    
    @T.prim_func
    def rotary_embedding(var_A: T.handle, B: T.Buffer((T.int64(2048), T.int64(128)), "float16"), C: T.Buffer((T.int64(2048), T.int64(128)), "float16"), var_rotary: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        
        A = T.match_buffer(var_A, (T.int64(1), 256, T.int64(32), T.int64(128)), "float16")
        rotary = T.match_buffer(var_rotary, (T.int64(1), 256, T.int64(32), T.int64(128)), "float16")
        # with T.block("root"):
        for i0, i1, i2, i3 in T.grid(T.int64(1), 256, T.int64(32), T.int64(128)):
            with T.block("rotary"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(B[256 + v_i1 - 256, v_i3], A[v_i0, v_i1, v_i2, v_i3 - T.int64(64):v_i3 - T.int64(64) + T.int64(129)], C[256 + v_i1 - 256, v_i3])
                T.writes(rotary[v_i0, v_i1, v_i2, v_i3])
                rotary[v_i0, v_i1, v_i2, v_i3] = B[256 + v_i1 - 256, v_i3] * A[v_i0, v_i1, v_i2, v_i3] + C[256 + v_i1 - 256, v_i3] * T.Select(T.int64(64) <= v_i3, A[v_i0, v_i1, v_i2, v_i3 - T.int64(64)], A[v_i0, v_i1, v_i2, v_i3 + T.int64(64)] * T.float16(-1))

    @R.function
    def foo(
       input_tokens: R.Tensor((1, 256, 4096), dtype="float16"), mask:R.Tensor((1, 1, 256, 256), dtype="float16"), div_const:R.Tensor((1, 32, 256, 256), dtype="float16"), maximum_const: R.Tensor((1, 32, 256, 256), dtype="float16"), kv_cache: R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object), linear_weight: R.Tensor((4096, 4096), dtype="float16"), linear_weight1: R.Tensor((4096, 4096), dtype="float16"), linear_weight2: R.Tensor((4096, 4096), dtype="float16"), linear_weight3: R.Tensor((4096, 4096), dtype="float16"), rms_norm_weight: R.Tensor((4096,), dtype="float16"), cos_cached: R.Tensor((2048, 128), dtype="float16"), sin_cached: R.Tensor((2048, 128), dtype="float16")
    ) : 
        R.func_attr({"num_input": 3, "tir_var_upper_bound": {"256": 2048, "256": 2048}})
        cls = LlamaDecoderLayer
        with R.dataflow():
            lv6 = R.call_tir(cls.rms_norm, (input_tokens, rms_norm_weight), out_sinfo=R.Tensor((1, 256, 4096), dtype="float16"))
            lv7: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(linear_weight, axes=None)
            lv7_copy: R.Tensor((4096, 4096), dtype="float16") = R.annotate_sharding(lv7, "mesh[0]", "S[1]")
            lv8: R.Tensor((1, 256, 4096), dtype="float16") = R.matmul(lv6, lv7_copy, out_dtype="void")
            lv9: R.Tensor((1, 256, 32, 128), dtype="float16") = R.reshape(lv8, R.shape([1, 256, 32, 128]))
            lv10: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(linear_weight1, axes=None)
            lv10_copy: R.Tensor((4096, 4096), dtype="float16") = R.annotate_sharding(lv10, "mesh[0]", "S[1]")
            lv11: R.Tensor((1, 256, 4096), dtype="float16") = R.matmul(lv6, lv10_copy, out_dtype="void")
            lv12: R.Tensor((1, 256, 32, 128), dtype="float16") = R.reshape(lv11, R.shape([1, 256, 32, 128]))
            lv13: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(linear_weight2, axes=None)
            lv13_copy: R.Tensor((4096, 4096), dtype="float16") = R.annotate_sharding(lv13, "mesh[0]", "S[1]")
            lv14: R.Tensor((1, 256, 4096), dtype="float16") = R.matmul(lv6, lv13_copy, out_dtype="void")
            lv15: R.Tensor((1, 256, 32, 128), dtype="float16") = R.reshape(lv14, R.shape([1, 256, 32, 128]))
            lv16 = R.call_tir(cls.rotary_embedding, (lv9, cos_cached, sin_cached), out_sinfo=R.Tensor((1, 256, 32, 128), dtype="float16"), tir_vars=R.shape([256]))
            lv17 = R.call_tir(cls.rotary_embedding, (lv12, cos_cached, sin_cached), out_sinfo=R.Tensor((1, 256, 32, 128), dtype="float16"), tir_vars=R.shape([256]))
            lv18: R.Tensor((256, 32, 128), dtype="float16") = R.reshape(lv17, R.shape([256, 32, 128]))
            lv19: R.Tensor((256, 32, 128), dtype="float16") = R.reshape(lv15, R.shape([256, 32, 128]))
            lv20: R.Object = kv_cache[0]
            lv21: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv20, lv18, sinfo_args=(R.Object,))
            lv22: R.Object = kv_cache[1]
            lv23: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", lv22, lv19, sinfo_args=(R.Object,))
            lv24: R.Tensor((256, 32, 128), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv21, R.shape([256, 32, 128]), sinfo_args=(R.Tensor((256, 32, 128), dtype="float16"),))
            lv25: R.Tensor((256, 32, 128), dtype="float16") = R.call_packed("vm.builtin.attention_kv_cache_view", lv23, R.shape([256, 32, 128]), sinfo_args=(R.Tensor((256, 32, 128), dtype="float16"),))
            lv26: R.Tensor((1, 256, 32, 128), dtype="float16") = R.reshape(lv24, R.shape([1, 256, 32, 128]))
            lv27: R.Tensor((1, 256, 32, 128), dtype="float16") = R.reshape(lv25, R.shape([1, 256, 32, 128]))
            lv28: R.Tensor((1, 32, 256, 128), dtype="float16") = R.permute_dims(lv16, axes=[0, 2, 1, 3])
            lv29: R.Tensor((1, 32, 256, 128), dtype="float16") = R.permute_dims(lv26, axes=[0, 2, 1, 3])
            lv30: R.Tensor((1, 32, 256, 128), dtype="float16") = R.permute_dims(lv27, axes=[0, 2, 1, 3])
            lv31: R.Tensor((1, 32, 128, 256), dtype="float16") = R.permute_dims(lv29, axes=[0, 1, 3, 2])
            lv32: R.Tensor((1, 32, 256, 256), dtype="float16") = R.matmul(lv28, lv31, out_dtype="void")
            # constant is not currently supported
            lv33: R.Tensor((1, 32, 256, 256), dtype="float16") = R.divide(lv32, div_const)
            lv34: R.Tensor((1, 32, 256, 256), dtype="float16") = R.maximum(lv33, maximum_const)
            lv35: R.Tensor((1, 32, 256, 256), dtype="float16") = R.minimum(lv34, mask)
            # lv36: R.Tensor((1, 32, 256, 256), dtype="float32") = R.astype(lv35, dtype="float32")
            lv37: R.Tensor((1, 32, 256, 256), dtype="float16") = R.nn.softmax(lv35, axis=-1)
            # lv38: R.Tensor((1, 32, 256, 256), dtype="float16") = R.astype(lv37, dtype="float16")
            lv39: R.Tensor((1, 32, 256, 128), dtype="float16") = R.matmul(lv37, lv30, out_dtype="void")
            lv40: R.Tensor((1, 256, 32, 128), dtype="float16") = R.permute_dims(lv39, axes=[0, 2, 1, 3])
            lv41: R.Tensor((1, 256, 4096), dtype="float16") = R.reshape(lv40, R.shape([1, 256, 4096]))
            lv42: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(linear_weight3, axes=None)
            lv43: R.Tensor((1, 256, 4096), dtype="float16") = R.matmul(lv41, lv42, out_dtype="void")
            lv44: R.Tensor((1, 256, 4096), dtype="float16") = R.add(input_tokens, lv43)
            gv = lv44
            R.output(gv)
        return gv
    

def test_simple_annotate_from_mid():
    after = relax.distributed.transform.PropagateSharding()(TestModule)
    print(after)
        
def test_decoder_layer():
    after = relax.distributed.transform.PropagateSharding()(LlamaDecoderLayer)
    print(after)

if __name__ == "__main__":
    # test_simple_annotate_from_mid()
    test_decoder_layer()