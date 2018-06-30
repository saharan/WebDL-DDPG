package webdl.core.backend.cuda.operation;
import webdl.core.Tensor;
import webdl.core.backend.cuda.CudaAtomicOperation;

/**
 * ...
 */
class CudaLinCombOperation extends CudaOperation {
	var a:Tensor;
	var b:Tensor;
	var dst:Tensor;

	public function new(a:Tensor, b:Tensor, dst:Tensor, aScale:Float, bScale:Float) {
		super([a, b], [dst]);
		this.a = a;
		this.b = b;
		this.dst = dst;
		this.forwardOps = [
			new CudaAtomicOperation([dst, a, b], "lincomb_forward", '
				val(0, idx4) = val(1, idx4) * $aScale + val(2, idx4) * $bScale;
			')
		];
		var broadcastedA:Tensor = addBroadcastBackward(a, dst, ShapeInference.getBroadcastedAxes(dst.shape, a.shape));
		var broadcastedB:Tensor = addBroadcastBackward(b, dst, ShapeInference.getBroadcastedAxes(dst.shape, b.shape));
		this.backwardOps = [
			new CudaAtomicOperation([broadcastedA, dst], "lincomb_backward_a", '
				float d = dif(1, idx4);
				dif(0, idx4) += d * $aScale;
			'),
			new CudaAtomicOperation([broadcastedB, dst], "lincomb_backward_b", '
				float d = dif(1, idx4);
				dif(0, idx4) += d * $bScale;
			')
		];
	}

	override function shapeCheck():Void {
		dst.assignShape(ShapeInference.broadcast(a.shape, b.shape, a.actualShape, b.actualShape));
	}

}
