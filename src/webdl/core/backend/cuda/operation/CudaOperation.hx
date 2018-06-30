package webdl.core.backend.cuda.operation;
import haxe.ds.Vector;
import webdl.core.Operation;
import webdl.core.Tensor;
import webdl.core.backend.cuda.CudaAtomicOperation;

/**
 * ...
 */
class CudaOperation extends Operation {
	var forwardOps:Array<CudaAtomicOperation>;
	var backwardOps:Array<CudaAtomicOperation>;
	var broadcasts:Array<BroadcastBackpropData>;

	public function new(inputs:Array<Tensor>, outputs:Array<Tensor>) {
		super(inputs, outputs);
		forwardOps = [];
		backwardOps = [];
		broadcasts = [];
	}

	function shapeEq(a:Vector<Int>, b:Vector<Int>, shrinkA:Int = 0, shrinkB:Int = 0):Void {
		var res:Bool = true;
		if (a.length - shrinkA != b.length - shrinkB) {
			res = false;
		} else {
			var num:Int = a.length - shrinkA;
			for (i in 0...num) {
				var sizeA:Int = a[i];
				var sizeB:Int = b[i];
				if (sizeA == -1 || sizeB == -1) throw "no data assigned";
				res = res && sizeA == sizeB;
			}
		}
		shapeAssert(res);
	}

	@:extern
	inline function shapeAssert(result:Bool):Void {
		if (!result) {
			throw "shapes mismatch";
		}
	}

	/**
	 * Returns broadcasted src.
	 */
	function addBroadcastBackward(src:Tensor, dst:Tensor, axes:Array<Int>):Tensor {
		if (axes.length == 0) return src; // no broadcasting
		var shape:Array<Int> = dst.shape.toArray();
		var prevTensor:Tensor = WebDL.tensorOfShape(shape);
		var intermediates:Array<Tensor> = [prevTensor];
		var ops:Array<CudaAtomicOperation> = [];
		for (i in 0...axes.length) {
			var axis:Int = axes[i];
			shape[axis] = 1;
			var nextTensor:Tensor = i == axes.length - 1 ? src : WebDL.tensorOfShape(shape);
			intermediates.push(nextTensor);
			var axisIndex:Int = dst.rank - 1 - axis;
			ops.push(new CudaAtomicOperation([nextTensor, prevTensor], "broadcast_backward" + i, '
				float sum = 0;
				// loop over the axis index
				for (int i = 0; i < shape1[$axisIndex]; i++) {
					idx4[$axisIndex] = i;
					sum += diff1[idx4to1(idx4, shape1)];
				}
				// assign the result
				diff0[idx1] ${nextTensor == src ? "+=" : "="} sum;
			'));
			prevTensor = nextTensor;
		}
		broadcasts.push(new BroadcastBackpropData(src, dst, axes, intermediates, ops, src.rank == 0));
		return intermediates[0];
	}

	function shapeCheck():Void {
	}

	function broadcastBackwardShapeCheck():Void {
		for (b in broadcasts) {
			var dst:Tensor = b.dst;
			var intermediates:Array<Tensor> = b.intermediates;
			var axes:Array<Int> = b.axes;
			var shape:Array<Int> = dst.actualShape.toArray();
			intermediates[0].assignShape(shape); // broadcasted <- dst
			intermediates[0].fillDiff(0);
			for (j in 1...intermediates.length) {
				shape[axes[j - 1]] = 1;
				if (j == intermediates.length - 1 && b.scalar) {
					intermediates[j].assignShape([]);
				} else {
					intermediates[j].assignShape(shape);
				}
			}
		}
	}

	override public function run():Void {
		shapeCheck();
		for (forwardOp in forwardOps) {
			forwardOp.run();
		}
	}

	override function backwardRun():Void {
		broadcastBackwardShapeCheck();
		for (backwardOp in backwardOps) {
			backwardOp.run();
		}
		for (b in broadcasts) {
			for (op in b.ops) {
				op.run();
			}
		}
	}
}
