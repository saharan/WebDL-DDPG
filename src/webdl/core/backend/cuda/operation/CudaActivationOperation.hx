package webdl.core.backend.cuda.operation;
import webdl.core.backend.cuda.CudaAtomicOperation;
import webdl.core.nn.Activation;

/**
 * ...
 */
class CudaActivationOperation extends CudaOperation {
	var a:Tensor;
	var dst:Tensor;

	public function new(a:Tensor, dst:Tensor, activation:Activation) {
		super([a], [dst]);
		this.a = a;
		var activForward:String = switch (activation) {
			case Linear: "a";
			case Tangent: "tanhf(a)";
			case Sigmoid: "sigmoid(a)";
			case Relu: "relu(a)";
		}
		var activBackward:String = switch (activation) {
			case Linear: "1";
			case Tangent: "tanhGrad(a)";
			case Sigmoid: "sigmoidGrad(a)";
			case Relu: "reluGrad(a)";
		}
		this.dst = dst;
		this.forwardOps = [
			new CudaAtomicOperation([dst, a], "activation_forward", '
				float a = val(1, idx4);
				val(0, idx4) = $activForward;
			')
		];
		this.backwardOps = [
			new CudaAtomicOperation([a, dst], "activation_backward", '
				float a = val(0, idx4);
				float d = dif(1, idx4);
				dif(0, idx4) += d * ($activBackward);
			')
		];
	}

	override function shapeCheck():Void {
		dst.assignShape(a.actualShape);
	}

}
