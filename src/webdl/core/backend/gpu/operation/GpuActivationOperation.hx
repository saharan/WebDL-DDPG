package webdl.core.backend.gpu.operation;
import webdl.core.nn.Activation;

/**
 * ...
 */
class GpuActivationOperation extends GpuOperation {
	var a:Tensor;
	var dst:Tensor;

	public function new(backend:GpuBackend, a:Tensor, dst:Tensor, activation:Activation) {
		super(backend, [a], [dst]);
		this.a = a;
		var activForward:String = switch (activation) {
			case Linear: "a.value";
			case Tangent: "tanh(a.value)";
			case Sigmoid: "sigmoid(a.value)";
			case Relu: "relu(a.value)";
		}
		var activBackward:String = switch (activation) {
			case Linear: "1.0";
			case Tangent: "tanhGrad(a.value)";
			case Sigmoid: "sigmoidGrad(a.value)";
			case Relu: "reluGrad(a.value)";
		}
		this.dst = dst;
		this.forwardOps = [
			fop([a], dst, "activation_forward", '
				float run(ivec4 idx4) {
					elem a = src1(idx4);
					return $activForward;
				}
			')
		];
		this.backwardOps = [
			bop([dst, a], a, "activation_backward", '
				float run(ivec4 idx4) {
					elem dst = src1(idx4);
					elem a = src2(idx4);
					return dst.diff * $activBackward;
				}
			')
		];
	}

	override function shapeCheck():Void {
		dst.assignShape(a.actualShape);
	}

}
