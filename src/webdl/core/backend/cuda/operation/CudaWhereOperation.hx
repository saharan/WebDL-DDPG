package webdl.core.backend.cuda.operation;
import webdl.core.Tensor;
import webdl.core.backend.cuda.CudaAtomicOperation;

/**
 * ...
 */
class CudaWhereOperation extends CudaOperation {
	var cond:Tensor;
	var a:Tensor;
	var b:Tensor;
	var dst:Tensor;

	public function new(cond:Tensor, a:Tensor, b:Tensor, dst:Tensor) {
		super([cond, a, b], [dst]);
		this.cond = cond;
		this.a = a;
		this.b = b;
		this.dst = dst;

		this.forwardOps = [
			new CudaAtomicOperation([dst, cond, a, b], "where_forward", '
				float cond = val(1, idx4);
				float a    = val(2, idx4);
				float b    = val(3, idx4);
				val(0, idx4) = cond > 0.5 ? a : b;
			')
		];

		this.backwardOps = [
			new CudaAtomicOperation([a, dst, cond], "where_backward_a", '
				float d    = dif(1, idx4);
				float cond = val(2, idx4);
				dif(0, idx4) += cond > 0.5 ? d : 0;
			'),
			new CudaAtomicOperation([b, dst, cond], "where_backward_b", '
				float d    = dif(1, idx4);
				float cond = val(2, idx4);
				dif(0, idx4) += cond > 0.5 ? 0 : d;
			')
		];
	}

	override function shapeCheck():Void {
		shapeEq(a.actualShape, b.actualShape);
		dst.assignShape(a.actualShape.toArray());
	}

}
