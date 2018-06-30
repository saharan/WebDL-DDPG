package webdl.core.backend.gpu.operation;

/**
 * ...
 */
class GpuExpOperation extends GpuOperation {
	var a:Tensor;
	var dst:Tensor;

	public function new(backend:GpuBackend, a:Tensor, dst:Tensor) {
		super(backend, [a], [dst]);
		this.a = a;
		this.dst = dst;
		this.forwardOps = [
			fop([a], dst, "exp_forward", '
				float run(ivec4 idx4) {
					elem a = src1(idx4);
					return exp(a.value);
				}
			')
		];
		this.backwardOps = [
			bop([dst], a, "exp_backward", '
				float run(ivec4 idx4) {
					elem dst = src1(idx4);
					return dst.diff * dst.value;
				}
			')
		];
	}

	override function shapeCheck():Void {
		dst.assignShape(a.actualShape);
	}

}
