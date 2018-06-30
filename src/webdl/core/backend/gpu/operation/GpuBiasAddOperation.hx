package webdl.core.backend.gpu.operation;
import webdl.core.backend.gpu.ShaderConsts.*;

/**
 * ...
 */
class GpuBiasAddOperation extends GpuOperation {
	var a:Tensor;
	var b:Tensor;
	var dst:Tensor;

	public function new(backend:GpuBackend, a:Tensor, b:Tensor, dst:Tensor) {
		super(backend, [a, b], [dst]);
		this.a = a;
		this.b = b;
		this.dst = dst;
		this.forwardOps = [
			fop([a, b], dst, "biasadd_forward", '
				float run(ivec4 idx4) {
					elem a = src1(idx4);
					elem b = src2(ivec4(idx4.x, 0, 0, 0));
					return a.value + b.value;
				}
			')
		];
		this.backwardOps = [
			bop([dst], a, "biasadd_backward_a", '
				float run(ivec4 idx4) {
					elem dst = src1(idx4);
					return dst.diff;
				}
			'),
			bop([dst], b, "biasadd_backward_b", '
				float run(ivec4 idx4) {
					// sum up all the higher dimensions in dst
					idx4.yzw = ivec3(0);
					int idx1Offset = index4To1(idx4, $U_SRC_SHAPE1);
					float sum = 0.0;
				' +
					GpuShader.loopOverDimensions("idx1", "idx1Offset", [1], [["y", "z", "w"]], '
						elem dst = src1(idx1);
						sum += dst.diff;
					')
				+ '
					return sum;
				}
			')
		];
	}

	override function shapeCheck():Void {
		if (a.actualShape[a.rank - 1] != b.actualShape[b.rank - 1]) throw "dimensions mismatch";
		dst.assignShape(a.actualShape);
	}

}
