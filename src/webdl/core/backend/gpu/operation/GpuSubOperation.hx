package webdl.core.backend.gpu.operation;

/**
 * ...
 */
class GpuSubOperation extends GpuOperation {
	var a:Tensor;
	var b:Tensor;
	var dst:Tensor;

	public function new(backend:GpuBackend, a:Tensor, b:Tensor, dst:Tensor) {
		super(backend, [a, b], [dst]);
		this.a = a;
		this.b = b;
		this.dst = dst;
		this.forwardOps = [
			fop([a, b], dst, "sub_forward", '
				float run(ivec4 idx4) {
					elem a = src1(idx4);
					elem b = src2(idx4);
					return a.value - b.value;
				}
			')
		];
		var broadcastedA:Tensor = addBroadcastBackward(a, dst, ShapeInference.getBroadcastedAxes(dst.shape, a.shape));
		var broadcastedB:Tensor = addBroadcastBackward(b, dst, ShapeInference.getBroadcastedAxes(dst.shape, b.shape));
		this.backwardOps = [
			bop([dst], broadcastedA, "sub_backward_a", '
				float run(ivec4 idx4) {
					elem dst = src1(idx4);
					return dst.diff;
				}
			'),
			bop([dst], broadcastedB, "sub_backward_b", '
				float run(ivec4 idx4) {
					elem dst = src1(idx4);
					return -dst.diff;
				}
			')
		];
	}

	override function shapeCheck():Void {
		dst.assignShape(ShapeInference.broadcast(a.shape, b.shape, a.actualShape, b.actualShape));
	}

}
