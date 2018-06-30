package webdl.core.backend.cuda.operation;
import webdl.core.backend.cuda.CudaAtomicOperation;

/**
 * ...
 */
class CudaBiasAddOperation extends CudaOperation {
	var a:Tensor;
	var b:Tensor;
	var dst:Tensor;

	public function new(a:Tensor, b:Tensor, dst:Tensor) {
		if (a.rank == 0 || b.rank == 0) {
			throw "ranks must be greater than 0";
		}
		super([a, b], [dst]);
		this.a = a;
		this.b = b;
		this.dst = dst;
		this.forwardOps = [
			new CudaAtomicOperation([dst, a, b], "biasadd_forward", '
				val(0, idx4) = val(1, idx4) + val(2, idx4);
			')
		];
		this.backwardOps = [
			new CudaAtomicOperation([a, dst], "biasadd_backward_a", '
				float d = dif(1, idx4);
				dif(0, idx4) += d;
			'),
			new CudaAtomicOperation([b, dst], "biasadd_backward_b", '
				def_idx4(dstIdx4);
				float sum = 0;
				for (int i = 0; i < shape1[1]; i++) {
					for (int j = 0; j < shape1[2]; j++) {
						for (int k = 0; k < shape1[3]; k++) {
							dstIdx4[1] = i;
							dstIdx4[2] = j;
							dstIdx4[3] = k;
							float d = dif(1, dstIdx4);
							sum += d;
						}
					}
				}
				dif(0, idx4) += sum;
			')
		];
	}

	override function shapeCheck():Void {
		if (a.actualShape[a.rank - 1] != b.actualShape[b.rank - 1]) throw "dimensions mismatch";
		dst.assignShape(a.actualShape);
	}

}
