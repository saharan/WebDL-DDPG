package webdl.core.backend.gpu.operation;
import webdl.core.Tensor;
import webdl.core.backend.gpu.ShaderConsts.*;

/**
 * ...
 */
class GpuWhereOperation extends GpuOperation {
	var cond:Tensor;
	var a:Tensor;
	var b:Tensor;
	var dst:Tensor;

	public function new(backend:GpuBackend, cond:Tensor, a:Tensor, b:Tensor, dst:Tensor) {
		super(backend, [cond, a, b], [dst]);
		this.cond = cond;
		this.a = a;
		this.b = b;
		this.dst = dst;

		this.forwardOps = [
			fop([cond, a, b], dst, "where_forward", '
				float run(ivec4 idx4) {
					elem cond = src1(idx4);
					elem a = src2(idx4);
					elem b = src3(idx4);
					return cond.value > 0.5 ? a.value : b.value;
				}
			')
		];

		this.backwardOps = [
			bop([dst, cond], a, "where_backward_a", '
				float run(ivec4 idx4) {
					elem dst = src1(idx4);
					elem cond = src2(idx4);
					return cond.value > 0.5 ? dst.diff : 0.0;
				}
			'),
			bop([dst, cond], b, "where_backward_b", '
				float run(ivec4 idx4) {
					elem dst = src1(idx4);
					elem cond = src2(idx4);
					return cond.value > 0.5 ? 0.0 : dst.diff;
				}
			')
		];
	}

	override function shapeCheck():Void {
		shapeEq(a.actualShape, b.actualShape);
		dst.assignShape(a.actualShape.toArray());
	}

}
