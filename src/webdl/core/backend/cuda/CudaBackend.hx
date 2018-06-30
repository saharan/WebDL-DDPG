package webdl.core.backend.cuda;
import pycuda.AutoInit;
import webdl.core.Operation;
import webdl.core.Tensor;
import webdl.core.TensorData;
import webdl.core.backend.Backend;
import webdl.core.backend.cuda.operation.CudaAbsOperation;
import webdl.core.backend.cuda.operation.CudaActivationOperation;
import webdl.core.backend.cuda.operation.CudaAdamUpdateOperation;
import webdl.core.backend.cuda.operation.CudaAddConstOperation;
import webdl.core.backend.cuda.operation.CudaAddOperation;
import webdl.core.backend.cuda.operation.CudaAssignOperation;
import webdl.core.backend.cuda.operation.CudaBiasAddOperation;
import webdl.core.backend.cuda.operation.CudaDivOperation;
import webdl.core.backend.cuda.operation.CudaExpOperation;
import webdl.core.backend.cuda.operation.CudaGradientsOperation;
import webdl.core.backend.cuda.operation.CudaLinCombOperation;
import webdl.core.backend.cuda.operation.CudaLogOperation;
import webdl.core.backend.cuda.operation.CudaMatMulOperation;
import webdl.core.backend.cuda.operation.CudaMergeOperation;
import webdl.core.backend.cuda.operation.CudaMulConstOperation;
import webdl.core.backend.cuda.operation.CudaMulOperation;
import webdl.core.backend.cuda.operation.CudaPowConstOperation;
import webdl.core.backend.cuda.operation.CudaPowOperation;
import webdl.core.backend.cuda.operation.CudaReduceMeanOperation;
import webdl.core.backend.cuda.operation.CudaReduceSumOperation;
import webdl.core.backend.cuda.operation.CudaSplitOperation;
import webdl.core.backend.cuda.operation.CudaSubOperation;
import webdl.core.backend.cuda.operation.CudaTensorDotOperation;
import webdl.core.backend.cuda.operation.CudaWhereOperation;
import webdl.core.nn.Activation;


/**
 * ...
 */
class CudaBackend implements Backend {
	var disposedData:Array<CudaTensorData>;

	public function new() {
		var init = AutoInit;
		disposedData = [];
	}

	public function requestTensorData(size:Int):TensorData {
		for (d in disposedData) {
			if (d.isPreferableSize(size)) {
				disposedData.remove(d);
				//trace("data reused: size = " + d.maxSize);
				return d;
			}
		}
		return new CudaTensorData(size);
	}

	public function disposeTensorData(data:TensorData):Void {
		if (!Std.is(data, CudaTensorData)) throw "backends mismatch";
		disposedData.push(cast data);
	}

	public function add(a:Tensor, b:Tensor, dst:Tensor):Operation {
		return new CudaAddOperation(a, b, dst);
	}

	public function addConst(a:Tensor, b:Float, dst:Tensor):Operation {
		return new CudaAddConstOperation(a, b, dst);
	}

	public function sub(a:Tensor, b:Tensor, dst:Tensor):Operation {
		return new CudaSubOperation(a, b, dst);
	}

	public function linComb(a:Tensor, b:Tensor, dst:Tensor, aScale:Float, bScale:Float):Operation {
		return new CudaLinCombOperation(a, b, dst, aScale, bScale);
	}

	public function mul(a:Tensor, b:Tensor, dst:Tensor):Operation {
		return new CudaMulOperation(a, b, dst);
	}

	public function mulConst(a:Tensor, b:Float, dst:Tensor):Operation {
		return new CudaMulConstOperation(a, b, dst);
	}

	public function div(a:Tensor, b:Tensor, dst:Tensor):Operation {
		return new CudaDivOperation(a, b, dst);
	}

	public function pow(a:Tensor, b:Tensor, dst:Tensor):Operation {
		return new CudaPowOperation(a, b, dst);
	}

	public function powConst(a:Tensor, b:Float, dst:Tensor):Operation {
		return new CudaPowConstOperation(a, b, dst);
	}

	public function matMul(a:Tensor, b:Tensor, dst:Tensor):Operation {
		return new CudaMatMulOperation(a, b, dst);
	}

	public function tensorDot(a:Tensor, b:Tensor, dst:Tensor, count:Int, axes:Array<Array<Int>>):Operation {
		return new CudaTensorDotOperation(a, b, dst, count, axes);
	}

	public function abs(a:Tensor, dst:Tensor):Operation {
		return new CudaAbsOperation(a, dst);
	}

	public function log(a:Tensor, dst:Tensor):Operation {
		return new CudaLogOperation(a, dst);
	}

	public function exp(a:Tensor, dst:Tensor):Operation {
		return new CudaExpOperation(a, dst);
	}

	public function biasAdd(a:Tensor, b:Tensor, dst:Tensor):Operation {
		return new CudaBiasAddOperation(a, b, dst);
	}

	public function activation(a:Tensor, dst:Tensor, activation:Activation):Operation {
		return new CudaActivationOperation(a, dst, activation);
	}

	public function reduceSum(a:Tensor, dst:Tensor, axis:Int, keepDim:Bool):Operation {
		return new CudaReduceSumOperation(a, dst, axis, keepDim);
	}

	public function reduceMean(a:Tensor, dst:Tensor, axis:Int, keepDim:Bool):Operation {
		return new CudaReduceMeanOperation(a, dst, axis, keepDim);
	}

	public function split(a:Tensor, dsts:Array<Tensor>, axis:Int, sizes:Array<Int>):Operation {
		return new CudaSplitOperation(a, dsts, axis, sizes);
	}

	public function merge(as:Array<Tensor>, dst:Tensor, axis:Int):Operation {
		return new CudaMergeOperation(as, dst, axis);
	}

	public function gradients(y:Tensor, xs:Array<Tensor>, dsts:Array<Tensor>, gradY:Tensor):Operation {
		return new CudaGradientsOperation(y, xs, dsts, gradY);
	}

	public function assign(a:Tensor, dummyDst:Tensor, dst:Tensor):Operation {
		return new CudaAssignOperation(a, dummyDst, dst);
	}

	public function where(cond:Tensor, a:Tensor, b:Tensor, dst:Tensor):Operation {
		return new CudaWhereOperation(cond, a, b, dst);
	}

	public function adamUpdate(count:Tensor, t:Tensor, g:Tensor, m:Tensor, v:Tensor, alpha:Tensor, beta1:Tensor, beta2:Tensor, epsilon:Tensor, l2Decay:Tensor, dummyDst:Tensor):Operation {
		return new CudaAdamUpdateOperation(count, t, g, m, v, alpha, beta1, beta2, epsilon, l2Decay, dummyDst);
	}

}
