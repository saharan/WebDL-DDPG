package webdl.core.backend.cuda.operation;
import webdl.core.backend.cuda.CudaAtomicOperation;

/**
 * ...
 */
class CudaLogOperation extends CudaOperation {
	var a:Tensor;
	var dst:Tensor;

	public function new(a:Tensor, dst:Tensor) {
		super([a], [dst]);
		this.a = a;
		this.dst = dst;
		this.forwardOps = [
			new CudaAtomicOperation([dst, a], "log_forward", '
				val(0, idx4) = logf(val(1, idx4));
			')
		];
		this.backwardOps = [
			new CudaAtomicOperation([a, dst], "log_backward", '
				float a = val(0, idx4);
				float d = dif(1, idx4);
				dif(0, idx4) += d / a;
			')
		];
	}

	override function shapeCheck():Void {
		dst.assignShape(a.actualShape);
	}

}
