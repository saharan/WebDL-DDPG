package webdl.core.backend.gpu.operation;

/**
 * ...
 */
class GpuLogOperation extends GpuOperation {
	var a:Tensor;
	var dst:Tensor;

	public function new(backend:GpuBackend, a:Tensor, dst:Tensor) {
		super(backend, [a], [dst]);
		this.a = a;
		this.dst = dst;
		this.forwardOps = [
			fop([a], dst, "log_forward", '
				float run(ivec4 idx4) {
					elem a = src1(idx4);
					return log(a.value);
				}
			')
		];
		this.backwardOps = [
			bop([dst, a], a, "log_backward", '
				float run(ivec4 idx4) {
					elem dst = src1(idx4);
					elem a = src2(idx4);
					return dst.diff / a.value;
				}
			')
		];
	}

	override function shapeCheck():Void {
		dst.assignShape(a.actualShape);
	}

}
