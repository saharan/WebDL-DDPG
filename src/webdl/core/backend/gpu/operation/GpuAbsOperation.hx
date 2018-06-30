package webdl.core.backend.gpu.operation;

/**
 * ...
 */
class GpuAbsOperation extends GpuOperation {
	var a:Tensor;
	var dst:Tensor;

	public function new(backend:GpuBackend, a:Tensor, dst:Tensor) {
		super(backend, [a], [dst]);
		this.a = a;
		this.dst = dst;
		this.forwardOps = [
			fop([a], dst, "abs_forward", '
				float run(ivec4 idx4) {
					elem a = src1(idx4);
					return abs(a.value);
				}
			')
		];
		this.backwardOps = [
			bop([dst, a], a, "abs_backward", '
				float run(ivec4 idx4) {
					elem dst = src1(idx4);
					elem a = src2(idx4);
					if (a.value < 0.0) {
						return -dst.diff;
					} else {
						return dst.diff;
					}
				}
			')
		];
	}

	override function shapeCheck():Void {
		dst.assignShape(a.actualShape);
	}

}
