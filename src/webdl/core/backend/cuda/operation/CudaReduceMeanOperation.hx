package webdl.core.backend.cuda.operation;
import haxe.ds.Vector;

/**
 * ...
 */
class CudaReduceMeanOperation extends CudaOperation {
	var a:Tensor;
	var dst:Tensor;
	var axis:Int;
	var keepDim:Bool;

	public function new(a:Tensor, dst:Tensor, axis:Int, keepDim:Bool) {
		super([a], [dst]);
		this.a = a;
		this.dst = dst;
		this.keepDim = keepDim;

		this.axis = axis;
		if (axis < 0 || axis >= a.rank) throw "invalid axis";
		var axisIndex:Int = a.rank - 1 - axis;

		if (keepDim) {
			this.forwardOps = [
				new CudaAtomicOperation([dst, a], "reducemean_forward_kd", '
					def_idx4(aIdx4);

					int num = shape1[$axisIndex];
					float sum = 0;
					for (int i = 0; i < num; i++) {
						aIdx4[$axisIndex] = i;
						sum += val(1, aIdx4);
					}
					val(0, idx4) = sum / num;
				')
			];
			this.backwardOps = [
				new CudaAtomicOperation([a, dst], "reducemean_backward_kd", '
					def_idx4(dstIdx4);
					dstIdx4[$axisIndex] = 0;

					int num = shape0[$axisIndex];
					float d = dif(1, dstIdx4);
					dif(0, idx4) += d / num;
				')
			];
		} else {
			this.forwardOps = [
				new CudaAtomicOperation([dst, a], "reducemean_forward", '
					def_idx4(aIdx4);
					ins(aIdx4, $axisIndex, 0);

					int num = shape1[$axisIndex];
					float sum = 0;
					for (int i = 0; i < num; i++) {
						aIdx4[$axisIndex] = i;
						sum += val(1, aIdx4);
					}
					val(0, idx4) = sum / num;
				')
			];
			this.backwardOps = [
				new CudaAtomicOperation([a, dst], "reducemean_backward", '
					def_idx4(dstIdx4);
					del(dstIdx4, $axisIndex);

					int num = shape0[$axisIndex];
					float d = dif(1, dstIdx4);
					dif(0, idx4) += d / num;
				')
			];
		}
	}

	override function shapeCheck():Void {
		var dstShape:Array<Int> = [];

		if (keepDim) {
			dstShape = a.actualShape.toArray();
			dstShape[axis] = 1;
			dst.assignShape(dstShape);
		} else {
			for (i in 0...a.rank) {
				if (axis != i) {
					dstShape.push(a.actualShape[i]);
				}
			}
			dst.assignShape(dstShape);
		}
	}

}
