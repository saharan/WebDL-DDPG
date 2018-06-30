package webdl.core.optimizer;
import webdl.core.Tensor;
import webdl.core.graph.Graph;
using webdl.core.WebDL;

/**
 * Adam optimizer.
 */
class AdamOptimizer extends Optimizer {
	var adamUpdates:Array<Tensor>;
	var l2Decay:Tensor;

	var internalTensors:Array<Tensor>;

	public function new(trainables:Array<Tensor>, grads:Array<Tensor>, alpha:Float = 0.001, beta1:Float = 0.9, beta2:Float = 0.999, epsilon:Float = 1e-8) {
		super(trainables, grads);

		adamUpdates = [];

		var count:Tensor = WebDL.tensorOfValue(0);
		var updatedCount:Tensor = count.assign(count.addConst(1));
		var alphaTensor:Tensor = WebDL.tensorOfValue(alpha);
		var beta1Tensor:Tensor = WebDL.tensorOfValue(beta1);
		var beta2Tensor:Tensor = WebDL.tensorOfValue(beta2);
		var epsilonTensor:Tensor = WebDL.tensorOfValue(epsilon);
		var zero:Tensor = WebDL.tensorOfValue(0);
		l2Decay = WebDL.tensorOfValue(0);

		internalTensors = [count, alphaTensor, beta1Tensor, beta2Tensor, epsilonTensor, l2Decay];

		var num:Int = trainables.length;
		for (i in 0...num) {
			var t:Tensor = trainables[i];
			var g:Tensor = grads[i];
			var m:Tensor = WebDL.tensorLike(t);
			var v:Tensor = WebDL.tensorLike(t);

			internalTensors.push(m);
			internalTensors.push(v);

			var update:Tensor = WebDL.adamUpdate(updatedCount, t, g, m, v, alphaTensor, beta1Tensor, beta2Tensor, epsilonTensor, t.doNotDecay ? zero : l2Decay);
			adamUpdates.push(update);
		}
	}

	public function setL2Decay(decay:Float):Void {
		l2Decay.set0D(decay);
	}

	override public function run():Void {
		WebDL.run(adamUpdates);
	}

	override public function exportData():Array<Float> {
		return internalTensors.exportElements();
	}

	override public function importData(data:Array<Float>):Void {
		internalTensors.importElements(data);
	}
}
