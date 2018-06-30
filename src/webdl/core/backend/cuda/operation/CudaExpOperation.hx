package webdl.core.backend.cuda.operation;
import webdl.core.backend.cuda.CudaAtomicOperation;

/**
 * ...
 */
class CudaExpOperation extends CudaOperation {
	var a:Tensor;
	var dst:Tensor;

	public function new(a:Tensor, dst:Tensor) {
		super([a], [dst]);
		this.a = a;
		this.dst = dst;
		this.forwardOps = [
			new CudaAtomicOperation([dst, a], "exp_forward", '
				val(0, idx4) = expf(val(1, idx4));
			')
		];
		this.backwardOps = [
			new CudaAtomicOperation([a, dst], "exp_backward", '
				float dst = val(1, idx4);
				float d = dif(1, idx4);
				dif(0, idx4) += d * dst;
			')
		];
	}

	override function shapeCheck():Void {
		dst.assignShape(a.actualShape);
	}

}
