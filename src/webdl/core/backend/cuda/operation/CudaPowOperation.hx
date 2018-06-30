package webdl.core.backend.cuda.operation;
import webdl.core.Tensor;
import webdl.core.backend.cuda.CudaAtomicOperation;

/**
 * ...
 */
class CudaPowOperation extends CudaOperation {
	var a:Tensor;
	var b:Tensor;
	var dst:Tensor;

	public function new(a:Tensor, b:Tensor, dst:Tensor) {
		super([a, b], [dst]);
		this.a = a;
		this.b = b;
		this.dst = dst;
		this.forwardOps = [
			new CudaAtomicOperation([dst, a, b], "pow_forward", '
				val(0, idx4) = powf(val(1, idx4), val(2, idx4));
			')
		];
		var broadcastedA:Tensor = addBroadcastBackward(a, dst, ShapeInference.getBroadcastedAxes(dst.shape, a.shape));
		var broadcastedB:Tensor = addBroadcastBackward(b, dst, ShapeInference.getBroadcastedAxes(dst.shape, b.shape));
		this.backwardOps = [
			new CudaAtomicOperation([broadcastedA, dst, a, b], "pow_backward_a", '
				float dst = val(1, idx4);
				float d   = dif(1, idx4);
				float a   = val(2, idx4);
				float b   = val(3, idx4);
				dif(0, idx4) += d * b * powf(a, b - 1);
			'),
			new CudaAtomicOperation([broadcastedB, dst, a], "pow_backward_b", '
				float dst = val(1, idx4);
				float d   = dif(1, idx4);
				float a   = val(2, idx4);
				dif(0, idx4) += d * logf(a) * dst;
			')
		];
	}

	override function shapeCheck():Void {
		dst.assignShape(ShapeInference.broadcast(a.shape, b.shape, a.actualShape, b.actualShape));
	}

}
