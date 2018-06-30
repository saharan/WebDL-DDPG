package webdl.core.optimizer;
import haxe.ds.Vector;

/**
 * ...
 */
class Optimizer {
	var trainables:Array<Tensor>;
	var grads:Array<Tensor>;

	public function new(trainables:Array<Tensor>, grads:Array<Tensor>) {
		this.trainables = trainables;
		this.grads = grads;
		if (trainables.length != grads.length) throw "invalid arguments";
	}

	public function run():Void {
	}

	/**
	 * Exports internal variables needed to restart optimization.
	 */
	public function exportData():Array<Float> {
		return [];
	}

	/**
	 * Imports interval variables needed to restart optimization.
	 */
	public function importData(data:Array<Float>):Void {
	}

}
