package webdl.core.backend.gpu.operation;
import webdl.core.backend.gpu.ShaderConsts.*;

/**
 * ...
 */
class GpuMatMulOperation extends GpuOperation {
	var a:Tensor;
	var b:Tensor;
	var dst:Tensor;

	public function new(backend:GpuBackend, a:Tensor, b:Tensor, dst:Tensor) {
		super(backend, [a, b], [dst]);
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
			fop([a, b], dst, "matmul_forward", '
				float run(ivec4 idx4) {
					ivec4 aIdx4 = replaceX(idx4, 0);
					ivec4 bIdx4 = replaceY(idx4, 0);

					int aIdx1Offset = index4To1(aIdx4, $U_SRC_SHAPE1);
					int bIdx1Offset = index4To1(bIdx4, $U_SRC_SHAPE2);

					ivec2 idx1Offsets = ivec2(aIdx1Offset, bIdx1Offset);
					float sum = 0.0;
				' +
					GpuShader.loopOverDimensions("idx1s", "idx1Offsets", [1, 2], [["x"], ["y"]], '
						elem a = src1(idx1s.x);
						elem b = src2(idx1s.y);
						sum += a.value * b.value;
					')
				+ '
					return sum;
				}
			')
		];

		this.backwardOps = [
			// ∂a.xy <- ∂c.iy * b.ix
			bop([dst, b], a, "matmul_backward_a", '
				float run(ivec4 idx4) {
					ivec4 dstIdx4 = ivec4(0, idx4.yzw);
					ivec4 bIdx4 = ivec4(0, idx4.xzw);

					int dstIdx1Offset = index4To1(dstIdx4, $U_SRC_SHAPE1);
					int bIdx1Offset = index4To1(bIdx4, $U_SRC_SHAPE2);

					ivec2 idx1Offsets = ivec2(dstIdx1Offset, bIdx1Offset);
					float sum = 0.0;
				' +
					GpuShader.loopOverDimensions("idx1s", "idx1Offsets", [1, 2], [["x"], ["x"]], '
						elem dst = src1(idx1s.x);
						elem b = src2(idx1s.y);
						sum += dst.diff * b.value;
					')
				+ '
					return sum;
				}
			'),
			// ∂b.xy <- ∂c.xi * a.yi
			bop([dst, a], b, "matmul_backward_b", '
				float run(ivec4 idx4) {
					ivec4 dstIdx4 = ivec4(idx4.x, 0, idx4.zw);
					ivec4 aIdx4 = ivec4(idx4.y, 0, idx4.zw);

					int dstIdx1Offset = index4To1(dstIdx4, $U_SRC_SHAPE1);
					int aIdx1Offset = index4To1(aIdx4, $U_SRC_SHAPE2);

					ivec2 idx1Offsets = ivec2(dstIdx1Offset, aIdx1Offset);
					float sum = 0.0;
				' +
					GpuShader.loopOverDimensions("idx1s", "idx1Offsets", [1, 2], [["y"], ["y"]], '
						elem dst = src1(idx1s.x);
						elem a = src2(idx1s.y);
						sum += dst.diff * a.value;
					')
				+ '
					return sum;
				}
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
