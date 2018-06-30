package webdl.rl.ddpg;
import webdl.core.Tensor;
import webdl.core.WebDL;
import webdl.core.layer.BatchNormalizationLayer;
import webdl.core.layer.DenseLayer;
import webdl.core.nn.Activation;
import webdl.core.nn.UniformInitializer;
using webdl.core.WebDL;

/**
 * A critic has information of actor network and its target network.
 */
class Critic {
	/**
	 * The critic network.
	 */
	public var network(default, null):CriticNetwork;

	/**
	 * The target network.
	 */
	public var targetNetwork(default, null):CriticNetwork;

	/**
	 * The dimension of the observable state space.
	 */
	public var stateDim(default, null):Int;

	/**
	 * The dimension of the action space.
	 */
	public var actionDim(default, null):Int;

	/**
	 * Creates critic with `network` and `targetNetwork`. They must have exactly the same structure,
	 * but must not the same instance.
	 */
	public function new(network:CriticNetwork, targetNetwork:CriticNetwork) {
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
	 * Returns a critic with the specified network setting.
	 */
	public static function createCritic(stateDim:Int, actionDim:Int, numNeuronsInHiddenLayers:Array<Int>, finalLayerWeight:Float = 0.003, batchNormalization:Bool = false):Critic {
		var network:CriticNetwork = createNetwork(stateDim, actionDim, numNeuronsInHiddenLayers, finalLayerWeight, batchNormalization);
		var targetNetwork:CriticNetwork = createNetwork(stateDim, actionDim, numNeuronsInHiddenLayers, finalLayerWeight, batchNormalization);
		return new Critic(network, targetNetwork);
	}

	static function createNetwork(stateDim:Int, actionDim:Int, numNeuronsInHiddenLayers:Array<Int>, finalLayerWeight:Float, batchNormalization:Bool):CriticNetwork {
		var inState:Tensor = WebDL.tensorOfShape([-1, stateDim]);
		var inAction:Tensor = WebDL.tensorOfShape([-1, actionDim]);
		var layers:Array<DenseLayer> = [];
		var layer:DenseLayer;
		var outTensor:Tensor = inState;
		var inDim:Int = stateDim;
		var isTraining:Tensor = WebDL.tensorOfValue(0);
		var updates:Array<Tensor> = [];
		for (i in 0...numNeuronsInHiddenLayers.length) {
			var unit:Int = numNeuronsInHiddenLayers[i];
			var activ:Activation = batchNormalization ? Linear : Relu;
			if (i == 1 || i == 0 && numNeuronsInHiddenLayers.length == 1) {
				// combine action
				inDim += actionDim;
				layer = new DenseLayer([outTensor, inAction].merge(1), unit, activ, new UniformInitializer(1 / Math.sqrt(inDim)), !batchNormalization);
			} else {
				layer = new DenseLayer(outTensor, unit, activ, new UniformInitializer(1 / Math.sqrt(inDim)), !batchNormalization);
			}
			layers.push(layer);
			outTensor = layer.output;

			if (batchNormalization) {
				var bn:BatchNormalizationLayer = new BatchNormalizationLayer(outTensor, isTraining, 0);
				updates = updates.concat(bn.updates);
				outTensor = bn.output.activation(Relu);

				// TODO: debug
				batchNormalization = false;
			}

			inDim = unit;
		}
		layer = new DenseLayer(outTensor, 1, Linear, new UniformInitializer(finalLayerWeight));
		layers.push(layer);
		var outValue:Tensor = layer.output.reduceSum(1); // [N, 1] -> [N]
		return new CriticNetwork(layers, inState, inAction, outValue, stateDim, actionDim, updates, isTraining);
	}
}
