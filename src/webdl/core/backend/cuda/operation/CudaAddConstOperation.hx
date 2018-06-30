package webdl.core.backend.cuda.operation;
import webdl.core.backend.cuda.CudaAtomicOperation;

/**
 * ...
 */
class CudaAddConstOperation extends CudaOperation {
	var a:Tensor;
	var dst:Tensor;

	public function new(a:Tensor, b:Float, dst:Tensor) {
		super([a], [dst]);
		this.a = a;
		this.dst = dst;
		this.forwardOps = [
			new CudaAtomicOperation([dst, a], "addconst_forward", '
				val(0, idx4) = val(1, idx4) + $b;
			')
		];
		this.backwardOps = [
			new CudaAtomicOperation([a, dst], "addconst_backward", '
				float d = dif(1, idx4);
				dif(0, idx4) += d;
			')
		];
	}

	override function shapeCheck():Void {
		dst.assignShape(a.actualShape);
	}

}
