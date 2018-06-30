package webdl.rl.ddpg;
import webdl.core.Tensor;
import webdl.core.WebDL;
import webdl.core.layer.BatchNormalizationLayer;
import webdl.core.layer.DenseLayer;
import webdl.core.layer.Layer;
import webdl.core.nn.UniformInitializer;
using webdl.core.WebDL;

/**
 * An actor has information of actor network and its target network.
 */
class Actor {
	/**
	 * The actor network.
	 */
	public var network(default, null):ActorNetwork;

	/**
	 * The target network.
	 */
	public var targetNetwork(default, null):ActorNetwork;

	/**
	 * The dimension of the observable state space.
	 */
	public var stateDim(default, null):Int;

	/**
	 * The dimension of the action space.
	 */
	public var actionDim(default, null):Int;

	/**
	 * Creates actor with `network` and `targetNetwork`. They must have exactly the same structure,
	 * but must not the same instance.
	 */
	public function new(network:ActorNetwork, targetNetwork:ActorNetwork) {
		this.network = network;
		this.targetNetwork = targetNetwork;

		if (network == targetNetwork) throw "network and target network must not be the same instance";

		if (
			network.stateDim != targetNetwork.stateDim ||
			network.actionDim != targetNetwork.actionDim ||
			network.trainables.length != targetNetwork.trainables.length ||
			network.tensorsToSave.length != targetNetwork.tensorsToSave.length
		) throw "dimensions mismatch between network and target network";

		stateDim = network.stateDim;
		actionDim = network.actionDim;
	}

	/**
	 * Returns an actor with the specified network setting.
	 */
	public static function createActor(stateDim:Int, actionDim:Int, numNeuronsInHiddenLayers:Array<Int>, actionRanges:Array<Array<Float>>, finalLayerWeight:Float = 0.003, batchNormalization:Bool = false):Actor {
		var network:ActorNetwork = createNetwork(stateDim, actionDim, numNeuronsInHiddenLayers, actionRanges, finalLayerWeight, batchNormalization);
		var targetNetwork:ActorNetwork = createNetwork(stateDim, actionDim, numNeuronsInHiddenLayers, actionRanges, finalLayerWeight, batchNormalization);
		return new Actor(network, targetNetwork);
	}

	static function createNetwork(stateDim:Int, actionDim:Int, numNeuronsInHiddenLayers:Array<Int>, actionRanges:Array<Array<Float>>, finalLayerWeight:Float, batchNormalization:Bool):ActorNetwork {
		var inState:Tensor = WebDL.tensorOfShape([-1, stateDim]);
		var layers:Array<DenseLayer> = [];
		var layer:DenseLayer;
		var outTensor:Tensor = inState;
		var inDim:Int = stateDim;
		var isTraining:Tensor = WebDL.tensorOfValue(0);
		var updates:Array<Tensor> = [];
		for (unit in numNeuronsInHiddenLayers) {
			if (batchNormalization) {
				layer = new DenseLayer(outTensor, unit, Linear, new UniformInitializer(1 / Math.sqrt(inDim)), false);
				layers.push(layer);
				outTensor = layer.output;
				var bn:BatchNormalizationLayer = new BatchNormalizationLayer(outTensor, isTraining, 0);
				updates = updates.concat(bn.updates);
				outTensor = bn.output.activation(Relu);

				// TODO: debug
				batchNormalization = false;
			} else {
				layer = new DenseLayer(outTensor, unit, Relu, new UniformInitializer(1 / Math.sqrt(inDim)));
				layers.push(layer);
				outTensor = layer.output;
			}

			inDim = unit;
		}
		layer = new DenseLayer(outTensor, actionDim, Tangent, new UniformInitializer(finalLayerWeight));
		layers.push(layer);
		outTensor = layer.output;
		var outBias:Tensor = WebDL.tensorOfValue([[for(range in actionRanges) 0.5 * (range[0] + range[1])]]);
		var outScale:Tensor = WebDL.tensorOfValue([[for (range in actionRanges) 0.5 * (range[1] - range[0])]]);
		var outAction:Tensor = outTensor.mul(outScale).add(outBias);
		return new ActorNetwork(layers, inState, outAction, stateDim, actionDim, updates, isTraining);
	}
}
