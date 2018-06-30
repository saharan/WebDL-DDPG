package webdl.core.backend.gpu.operation;
import webdl.core.Tensor;
import webdl.core.graph.Graph;

/**
 * ...
 */
class GpuGradientsOperation extends GpuOperation {
	var y:Tensor;
	var xs:Array<Tensor>;
	var dsts:Array<Tensor>;
	var gradY:Tensor;

	public function new(backend:GpuBackend, y:Tensor, xs:Array<Tensor>, dsts:Array<Tensor>, gradY:Tensor) {
		super(backend, xs.concat(gradY == null ? [y] : [y, gradY]), dsts);
		this.y = y;
		this.xs = xs;
		this.dsts = dsts;
		this.gradY = gradY;

		if (xs.length != dsts.length) throw "invalid argument";

		this.forwardOps = [for (i in 0...xs.length) {
			fop([xs[i]], dsts[i], "gradient_forward_" + i, '
				float run(ivec4 idx4) {
					elem a = src1(idx4);
					return a.diff;
				}
			');
		}];
		this.backwardOps = gradY == null ? [] : [
			bop([gradY], y, "gradient_init", '
				float run(ivec4 idx4) {
					elem gradY = src1(idx4);
					return gradY.value;
				}
			', false) // do not accumulate
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
