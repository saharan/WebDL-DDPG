package webdl.rl.ddpg;
import haxe.ds.Vector;
import webdl.core.Tensor;
import webdl.core.WebDL;
using webdl.core.WebDL;

/**
 * On-line mean and standard deviation tracker.
 */
class MeanStdTracker {
	var sum:Tensor;
	var sumSq:Tensor;
	var count:Tensor;

	public var mean:Tensor;
	public var std:Tensor;
	var sample:Tensor;
	var updates:Array<Tensor>;

	public function new(shape:Array<Int>) {
		sample = shape.tensorOfShape();
		sum = shape.tensorOfShape();
		sumSq = shape.tensorOfShape();
		mean = shape.tensorOfShape();
		std = shape.tensorOfShape();
		count = WebDL.tensorOfValue(0);
		mean.fill(0);
		std.fill(1);
		mean.shouldBeSaved = true;
		std.shouldBeSaved = true;

		var newSum:Tensor = sum.assign(sum.add(sample));
		var newSumSq:Tensor = sumSq.assign(sumSq.add(sample.square()));
		var newCount:Tensor = count.assign(count.addConst(1));
		var newMean:Tensor = mean.assign(newSum.div(newCount));
		var newStd:Tensor = std.assign(newSumSq.div(newCount).sub(newMean.square()).addConst(1e-6).powConst(0.5));
		updates = [newMean, newStd];
	}

	public function update(sampleDataArray:Array<Float>):Array<Tensor> {
		sample.setArray(sampleDataArray);
		return updates;
	}

}
