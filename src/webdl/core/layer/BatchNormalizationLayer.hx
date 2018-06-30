package webdl.core.layer;
import webdl.core.Tensor;
import webdl.core.nn.Activation;
import webdl.core.nn.HeInitializer;
import webdl.core.nn.Initializer;
import webdl.core.nn.XavierInitializer;
using webdl.core.WebDL;

/**
 * A batch normalization layer.
 */
class BatchNormalizationLayer extends Layer {
	var beta:Tensor;
	var scale:Tensor;
	var popMean:Tensor;
	var popVariance:Tensor;
	var batchMean:Tensor;
	var batchVariance:Tensor;
	var isTraining:Tensor;

	var initializer:Initializer;
	var kernelIn:Int;
	var out:Tensor;

	/**
	 * Normalizes `a` across the batch axis (NOT feature axis) `batchAxis`.
	 *
	 * `isTraining` is the tensor which should be set to `1.0` during the training,
	 * `0.0` during the testing.
	 *
	 * Statistics of mean and standard deviation are updated with learning rate
	 * `1 - momentum`, every time `this.updates` is evaluated.
	 *
	 * `epsilon` is a small number to avoid division by zero.
	 */
	public function new(a:Tensor, isTraining:Tensor, batchAxis:Int, momentum:Float = 0.95, epsilon:Float = 1e-6) {
		super();
		input = a;

		batchMean = a.reduceMean(batchAxis, true);
		batchVariance = a.sub(batchMean).square().reduceMean(batchAxis, true);
		popMean = WebDL.tensorLike(batchMean);
		popVariance = WebDL.tensorLike(batchMean);
		beta = WebDL.tensorLike(batchMean);
		scale = WebDL.tensorLike(batchMean);
		popMean.fill(0);
		popVariance.fill(1);
		beta.fill(0);
		scale.fill(1);

		var trainMean:Tensor = popMean.assign(WebDL.linComb(popMean, batchMean, momentum, 1 - momentum));
		var trainVariance:Tensor = popVariance.assign(WebDL.linComb(popVariance, batchVariance, momentum, 1 - momentum));

		updates = [trainMean, trainVariance];

		popMean.shouldBeSaved = true;
		popVariance.shouldBeSaved = true;
		scale.trainable = true;
		scale.shouldBeSaved = true;
		beta.trainable = true;
		beta.doNotDecay = true;
		beta.shouldBeSaved = true;

		var out1:Tensor = a.sub(batchMean).mul(batchVariance.addConst(epsilon).powConst(-0.5)).mul(scale).add(beta);
		var out2:Tensor = a.sub(popMean).mul(popVariance.addConst(epsilon).powConst(-0.5)).mul(scale).add(beta);

		output = WebDL.where(isTraining, out1, out2);
	}

	override public function init():Void {
		popMean.fill(0);
		popVariance.fill(1);
	}

}
