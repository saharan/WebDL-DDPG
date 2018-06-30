package webdl.core.backend.cuda.operation;
import webdl.core.Tensor;
import webdl.core.backend.cuda.CudaAtomicOperation;
import webdl.core.graph.Graph;

/**
 * ...
 */
class CudaGradientsOperation extends CudaOperation {
	var y:Tensor;
	var xs:Array<Tensor>;
	var dsts:Array<Tensor>;
	var gradY:Tensor;

	public function new(y:Tensor, xs:Array<Tensor>, dsts:Array<Tensor>, gradY:Tensor) {
		super(xs.concat(gradY == null ? [y] : [y, gradY]), dsts);
		this.y = y;
		this.xs = xs;
		this.dsts = dsts;
		this.gradY = gradY;

		if (xs.length != dsts.length) throw "invalid argument";

		this.forwardOps = [for (i in 0...xs.length) {
			new CudaAtomicOperation([dsts[i], xs[i]], "gradient_forward_" + i, '
				value0[idx1] = diff1[idx1];
			');
		}];
		this.backwardOps = gradY == null ? [] : [
			new CudaAtomicOperation([y, gradY], "gradient_init", '
				diff0[idx1] = value1[idx1];
			')
		];
	}

	override function run():Void {
		//trace("BACKPROP");
		if (gradY == null) {
			y.fillDiff(1.0);
		} else {
			super.backwardRun(); // run backward to init y's grad
		}
		Graph.backprop(y.node);
		super.run();
	}

	override public function backwardRun():Void {
		throw "gradient operation is not differentiable";
	}

	override function shapeCheck():Void {
		for (i in 0...dsts.length) {
			dsts[i].assignShape(xs[i].actualShape);
		}
	}

}
