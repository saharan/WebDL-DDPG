package webdl.core.backend.gpu.operation;

/**
 * ...
 */
class GpuPowConstOperation extends GpuOperation {
	var a:Tensor;
	var dst:Tensor;

	public function new(backend:GpuBackend, a:Tensor, b:Float, dst:Tensor) {
		super(backend, [a], [dst]);
		this.a = a;
		this.dst = dst;
		this.forwardOps = [
			fop([a], dst, "powconst_forward", '
				float run(ivec4 idx4) {
					elem a = src1(idx4);
					return safePow(a.value, float($b));
				}
			')
		];
		this.backwardOps = [
			bop([dst, a], a, "powconst_backward", '
				float run(ivec4 idx4) {
					elem dst = src1(idx4);
					elem a = src2(idx4);
					return dst.diff * float($b) * safePow(a.value, float($b) - 1.0);
				}
			')
		];
	}

	override function shapeCheck():Void {
		dst.assignShape(a.actualShape);
	}

}
