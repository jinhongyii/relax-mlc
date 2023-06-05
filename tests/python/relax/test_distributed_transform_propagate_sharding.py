from tvm.script.parser import ir as I
from tvm.script.parser import relax as R
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
        lv2 = R.annotate_sharding(lv1, device_mesh="mesh[0]", placement="R, S[0]" )
        lv3 = R.matmul(lv2, weight2)
        return lv3
    
    

def test_simple_annotate_from_mid():
    after = relax.distributed.transform.PropagateSharding()(TestModule)
    print(after)
        
if __name__ == "__main__":
    test_simple_annotate_from_mid()