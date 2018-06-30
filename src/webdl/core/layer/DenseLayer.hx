package webdl.core.layer;
import webdl.core.nn.Activation;
import webdl.core.nn.HeInitializer;
import webdl.core.nn.Initializer;
import webdl.core.nn.XavierInitializer;

/**
 * A fully-connected layer.
 */
class DenseLayer extends Layer {
	/**
	 * The kernel of the layer. This is the transpose of the weight matrix.
	 */
	public var kernel:Tensor;

	/**
	 * The bias of the layer.
	 */
	public var bias:Tensor;

	var initializer:Initializer;
	var kernelIn:Int;

	public function new(a:Tensor, unit:Int, activation:Activation = Linear, initializer:Initializer = null, useBias:Bool = true) {
		super();
		input = a;
		if (initializer == null) {
			initializer = activation == Relu ? new HeInitializer() : new XavierInitializer();
		}
		this.initializer = initializer;

		var outputShape:Array<Int> = a.shape.toArray();
		outputShape[a.rank - 1] = unit;
		output = WebDL.tensorOfShape(outputShape);

		kernelIn = a.shape[a.rank - 1];
		var kernelOut:Int = unit;

		if (kernelIn == -1) throw "the size of the last dimension of the input tensor must be known at the layer constructor";

		kernel = WebDL.tensorOfShape([kernelIn, kernelOut]);
		kernel.trainable = true;
		kernel.shouldBeSaved = true;
		if (useBias) {
			bias = WebDL.tensorOfShape([kernelOut]);
			bias.trainable = true;
			bias.shouldBeSaved = true;
			bias.doNotDecay = true;
			output = WebDL.biasAdd(WebDL.tensorDot(a, kernel, 1), bias);
		} else {
			bias = null;
			output = WebDL.tensorDot(a, kernel, 1);
		}
		if (activation != Linear) {
			output = WebDL.activation(output, activation);
		}
	}

	override public function init():Void {
		initializer.numInputs = kernelIn;
		kernel.fillByGenerator(initializer.next);
		if (bias != null) {
			bias.fillByGenerator(initializer.next);
		}
	}

}
