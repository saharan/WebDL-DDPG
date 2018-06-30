package webdl.core.optimizer;
import webdl.core.Tensor;
import webdl.core.graph.Graph;
using webdl.core.WebDL;

/**
 * Gradient descent optimizer.
 */
class GradientDescentOptimizer extends Optimizer {
	var updatedTrainables:Array<Tensor>;

	public function new(trainables:Array<Tensor>, grads:Array<Tensor>, learningRate:Float = 0.01) {
		super(trainables, grads);

		updatedTrainables = [];
		var num:Int = trainables.length;
		for (i in 0...num) {
			var t:Tensor = trainables[i];
			var g:Tensor = grads[i];
			var updatedT:Tensor = t.assign(WebDL.linComb(t, g, 1, -learningRate));
			updatedTrainables.push(updatedT);
		}
	}

	override public function run():Void {
		WebDL.run(updatedTrainables);
	}

	override public function exportData():Array<Float> {
		return [];
	}

	override public function importData(data:Array<Float>):Void {
	}
}
