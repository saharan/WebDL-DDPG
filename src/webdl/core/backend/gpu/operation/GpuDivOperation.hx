package webdl.core.backend.gpu.operation;

/**
 * ...
 */
class GpuDivOperation extends GpuOperation {
	var a:Tensor;
	var b:Tensor;
	var dst:Tensor;

	public function new(backend:GpuBackend, a:Tensor, b:Tensor, dst:Tensor) {
		super(backend, [a, b], [dst]);
		this.a = a;
		this.b = b;
		this.dst = dst;
		this.forwardOps = [
			fop([a, b], dst, "div_forward", '
				float run(ivec4 idx4) {
					elem a = src1(idx4);
					elem b = src2(idx4);
					return a.value / b.value;
				}
			')
		];
		var broadcastedA:Tensor = addBroadcastBackward(a, dst, ShapeInference.getBroadcastedAxes(dst.shape, a.shape));
		var broadcastedB:Tensor = addBroadcastBackward(b, dst, ShapeInference.getBroadcastedAxes(dst.shape, b.shape));
		this.backwardOps = [
			bop([dst, b], broadcastedA, "div_backward_a", '
				float run(ivec4 idx4) {
					elem dst = src1(idx4);
					elem b = src2(idx4);
					return dst.diff / b.value;
				}
			'),
			bop([dst, a, b], broadcastedB, "div_backward_b", '
				float run(ivec4 idx4) {
					elem dst = src1(idx4);
					elem a = src2(idx4);
					elem b = src3(idx4);
					return -dst.diff * a.value / (b.value * b.value);
				}
			')
		];
	}

	override function shapeCheck():Void {
		dst.assignShape(ShapeInference.broadcast(a.shape, b.shape, a.actualShape, b.actualShape));
	}

}
