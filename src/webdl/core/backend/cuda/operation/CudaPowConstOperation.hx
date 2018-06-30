package webdl.core.backend.cuda.operation;
import webdl.core.backend.cuda.CudaAtomicOperation;

/**
 * ...
 */
class CudaPowConstOperation extends CudaOperation {
	var a:Tensor;
	var dst:Tensor;

	public function new(a:Tensor, b:Float, dst:Tensor) {
		super([a], [dst]);
		this.a = a;
		this.dst = dst;
		this.forwardOps = [
			new CudaAtomicOperation([dst, a], "powconst_forward", '
				val(0, idx4) = powf(val(1, idx4), $b);
			')
		];
		this.backwardOps = [
			new CudaAtomicOperation([a, dst], "powconst_backward", '
				float a   = val(0, idx4);
				float dst = val(1, idx4);
				float d   = dif(1, idx4);
				dif(0, idx4) += d * $b * powf(a, $b - 1);
			')
		];
	}

	override function shapeCheck():Void {
		dst.assignShape(a.actualShape);
	}

}
