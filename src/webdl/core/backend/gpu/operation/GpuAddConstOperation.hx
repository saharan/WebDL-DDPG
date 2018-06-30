package webdl.core.backend.gpu.operation;

/**
 * ...
 */
class GpuAddConstOperation extends GpuOperation {
	var a:Tensor;
	var dst:Tensor;

	public function new(backend:GpuBackend, a:Tensor, b:Float, dst:Tensor) {
		super(backend, [a], [dst]);
		this.a = a;
		this.dst = dst;
		this.forwardOps = [
			fop([a], dst, "addconst_forward", '
				float run(ivec4 idx4) {
					elem a = src1(idx4);
					return a.value + float($b);
				}
			')
		];
		this.backwardOps = [
			bop([dst], a, "addconst_backward", '
				float run(ivec4 idx4) {
					elem dst = src1(idx4);
					return dst.diff;
				}
			')
		];
	}

	override function shapeCheck():Void {
		dst.assignShape(a.actualShape);
	}

}
