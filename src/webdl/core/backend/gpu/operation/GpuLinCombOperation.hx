package webdl.core.backend.gpu.operation;

/**
 * ...
 */
class GpuLinCombOperation extends GpuOperation {
	var a:Tensor;
	var b:Tensor;
	var dst:Tensor;

	public function new(backend:GpuBackend, a:Tensor, b:Tensor, dst:Tensor, aScale:Float, bScale:Float) {
		super(backend, [a, b], [dst]);
		this.a = a;
		this.b = b;
		this.dst = dst;
		this.forwardOps = [
			fop([a, b], dst, "lincomb_forward", '
				float run(ivec4 idx4) {
					elem a = src1(idx4);
					elem b = src2(idx4);
					return a.value * float($aScale) + b.value * float($bScale);
				}
			')
		];
		var broadcastedA:Tensor = addBroadcastBackward(a, dst, ShapeInference.getBroadcastedAxes(dst.shape, a.shape));
		var broadcastedB:Tensor = addBroadcastBackward(b, dst, ShapeInference.getBroadcastedAxes(dst.shape, b.shape));
		this.backwardOps = [
			bop([dst], broadcastedA, "lincomb_backward_a", '
				float run(ivec4 idx4) {
					elem dst = src1(idx4);
					return dst.diff * float($aScale);
				}
			'),
			bop([dst], broadcastedB, "lincomb_backward_b", '
				float run(ivec4 idx4) {
					elem dst = src1(idx4);
					return dst.diff * float($bScale);
				}
			')
		];
	}

	override function shapeCheck():Void {
		dst.assignShape(ShapeInference.broadcast(a.shape, b.shape, a.actualShape, b.actualShape));
	}

}
