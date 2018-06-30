package webdl.core.backend.cuda.operation;
import webdl.core.backend.cuda.CudaAtomicOperation;

/**
 * ...
 */
class CudaAbsOperation extends CudaOperation {
	var a:Tensor;
	var dst:Tensor;

	public function new(a:Tensor, dst:Tensor) {
		super([a], [dst]);
		this.a = a;
		this.dst = dst;
		this.forwardOps = [
			new CudaAtomicOperation([dst, a], "abs_forward", '
				val(0, idx4) = fabsf(val(1, idx4));
			')
		];
		this.backwardOps = [
			new CudaAtomicOperation([a, dst], "abs_backward", '
				float a = val(0, idx4);
				float d = dif(1, idx4);
				dif(0, idx4) += a < 0 ? -d : d;
			')
		];
	}

	override function shapeCheck():Void {
		dst.assignShape(a.actualShape);
	}

}
