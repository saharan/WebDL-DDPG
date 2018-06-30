package webdl.core.backend.gpu.operation;
import haxe.ds.Vector;
import js.html.webgl.GL;
import webdl.core.Operation;
import webdl.core.Tensor;
import webdl.core.backend.Backend;
import webdl.core.backend.gpu.GpuAtomicOperation;
import webdl.core.backend.gpu.GpuBackend;
import webdl.core.backend.gpu.GpuShader;
import webdl.core.backend.gpu.ShaderConsts.*;

/**
 * ...
 */
class GpuOperation extends Operation {
	var backend:GpuBackend;
	var forwardOps:Array<GpuAtomicOperation>;
	var backwardOps:Array<GpuAtomicOperation>;
	var broadcasts:Array<BroadcastBackpropData>;

	public function new(backend:GpuBackend, inputs:Array<Tensor>, outputs:Array<Tensor>) {
		super(inputs, outputs);
		this.backend = backend;
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

	function fop(inputs:Array<Tensor>, output:Tensor, name:String, fragmentSource:String):GpuAtomicOperation {
		return new GpuAtomicOperation(inputs, output, name, new GpuShader(backend.gl, inputs.length, GpuShader.FORWARD, fragmentSource));
	}

	function bop(inputs:Array<Tensor>, output:Tensor, name:String, fragmentSource:String, accumulate:Bool = true):GpuAtomicOperation {
		return new GpuAtomicOperation(inputs, output, name, new GpuShader(backend.gl, inputs.length, accumulate ? GpuShader.BACKWARD_ACCUMULATE : GpuShader.BACKWARD_OVERWRITE, fragmentSource));
	}

	/**
	 * Returns broadcasted src.
	 */
	function addBroadcastBackward(src:Tensor, dst:Tensor, axes:Array<Int>):Tensor {
		if (axes.length == 0) return src; // no broadcasting
		var shape:Array<Int> = dst.shape.toArray();
		var prevTensor:Tensor = WebDL.tensorOfShape(shape);
		var intermediates:Array<Tensor> = [prevTensor];
		var ops:Array<GpuAtomicOperation> = [];
		for (i in 0...axes.length) {
			var axis:Int = axes[i];
			shape[axis] = 1;
			var nextTensor:Tensor = i == axes.length - 1 ? src : WebDL.tensorOfShape(shape);
			intermediates.push(nextTensor);
			var axisComp:String = "xyzw".charAt(dst.rank - 1 - axis);
			ops.push(
				new GpuAtomicOperation([prevTensor], nextTensor, "broadcast_backward" + i, new GpuShader(backend.gl, 1, nextTensor == src ? GpuShader.BACKWARD_ACCUMULATE : GpuShader.BACKWARD_OVERWRITE, '
					float run(ivec4 idx4) {
						int idx1Offset = index4To1(idx4, $U_SRC_SHAPE1);
						float sum = 0.0;
					' +
						GpuShader.loopOverDimensions("idx1", "idx1Offset", [1], [[axisComp]], '
							elem a = src1(idx1);
							sum += a.diff;
						')
					+ '
						return sum;
					}
				'))
			);
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
			backend.runAtomicOperation(forwardOp);
		}
	}

	override function backwardRun():Void {
		broadcastBackwardShapeCheck();
		for (backwardOp in backwardOps) {
			backend.runAtomicOperation(backwardOp);
		}
		for (b in broadcasts) {
			for (op in b.ops) {
				backend.runAtomicOperation(op);
			}
		}
	}

}
