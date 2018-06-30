package webdl.core.backend.cuda.operation;
import webdl.core.backend.cuda.CudaAtomicOperation;

/**
 * ...
 */
class CudaMatMulOperation extends CudaOperation {
	var a:Tensor;
	var b:Tensor;
	var dst:Tensor;

	public function new(a:Tensor, b:Tensor, dst:Tensor) {
		super([a, b], [dst]);
		this.a = a;
		this.b = b;
		this.dst = dst;

		// c := dst
		//
		// c_ij = a_ik * b_kj
		//
		// ↓ diff
		//
		// ∂c_ij/∂a_ik = b_kj
		// ∂c_ij/∂b_kj = a_ik
		//
		// ↓ backprop
		//
		// ∂a_ik <- ∂c_ij * b_kj
		// ∂b_kj <- ∂c_ij * a_ik
		//
		// ↓ replace indices
		//
		// ∂a.xy <- ∂c.iy * b.ix
		// ∂b.xy <- ∂c.xi * a.yi

		this.forwardOps = [
			new CudaAtomicOperation([dst, a, b], "matmul_forward", '
				def_idx4(aIdx4);
				def_idx4(bIdx4);

				int num = shape1[0]; // num cols of `a`
				float sum = 0;
				for (int i = 0; i < num; i++) {
					aIdx4[0] = i;
					bIdx4[1] = i;
					float a = val(1, aIdx4);
					float b = val(2, bIdx4);
					sum += a * b;
				}

				val(0, idx4) = sum;
			')
		];

		this.backwardOps = [
			// ∂a.xy <- ∂c.iy * b.ix
			new CudaAtomicOperation([a, dst, b], "matmul_backward_a", '
				def_idx4(dstIdx4);
				def_idx4(bIdx4);

				// dst = {*, 1, 2, 3}
				// b   = {*, 0, 2, 3}
				bIdx4[1] = bIdx4[0];

				int num = shape1[0];
				float sum = 0;
				for (int i = 0; i < num; i++) {
					dstIdx4[0] = i;
					bIdx4[0] = i;
					float d = dif(1, dstIdx4);
					float b = val(2, bIdx4);
					sum += d * b;
				}

				dif(0, idx4) += sum;
			'),
			// ∂b.xy <- ∂c.xi * a.yi
			new CudaAtomicOperation([b, dst, a], "matmul_backward_b", '
				def_idx4(dstIdx4);
				def_idx4(aIdx4);

				// dst = {0, *, 2, 3}
				// a   = {1, *, 2, 3}
				aIdx4[0] = aIdx4[1];

				int num = shape1[1];
				float sum = 0;
				for (int i = 0; i < num; i++) {
					dstIdx4[1] = i;
					aIdx4[1] = i;
					float d = dif(1, dstIdx4);
					float a = val(2, aIdx4);
					sum += d * a;
				}

				dif(0, idx4) += sum;
			')
		];
	}

	override function shapeCheck():Void {
		var aLast:Int = a.rank - 1;
		var bLast:Int = b.rank - 1;
		shapeEq(a.actualShape, b.actualShape, 2, 2);

		if (a.actualShape[aLast] != b.actualShape[bLast - 1]) throw "cannot multiply matrices";
		var dstShape:Array<Int> = a.actualShape.toArray();
		dstShape[bLast] = b.actualShape[bLast];

		dst.assignShape(dstShape);
	}
}
