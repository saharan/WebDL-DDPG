package webdl.rl.ddpg;
import webdl.core.Tensor;
import webdl.core.WebDL;
import webdl.core.layer.DenseLayer;

/**
 * A critic network (or the target network of a critic network).
 */
class CriticNetwork {
	public var layers(default, null):Array<DenseLayer>;
	public var inState(default, null):Tensor;
	public var inAction(default, null):Tensor;
	public var outValue(default, null):Tensor;
	public var trainables(default, null):Array<Tensor>;
	public var tensorsToSave(default, null):Array<Tensor>;
	public var stateDim(default, null):Int;
	public var actionDim(default, null):Int;
	public var updates(default, null):Array<Tensor>;
	public var isTraining(default, null):Tensor;

	public function new(layers:Array<DenseLayer>, inState:Tensor, inAction:Tensor, outValue:Tensor, stateDim:Int, actionDim:Int, updates:Array<Tensor>, isTraining:Tensor) {
		this.layers = layers;
		this.inState = inState;
		this.inAction = inAction;
		this.outValue = outValue;
		this.stateDim = stateDim;
		this.actionDim = actionDim;
		this.updates = updates;
		this.isTraining = isTraining;
		trainables = WebDL.getTrainableVariables([outValue]);
		tensorsToSave = WebDL.getTensorsToSave([outValue]);
		for (layer in layers) layer.init();
	}

	public function dump():Void {
		trace("critic network info");
		for (layer in layers) {
			trace("kernel: " + layer.kernel.print());
			trace("bias: " + layer.bias.print());
		}
	}
}
