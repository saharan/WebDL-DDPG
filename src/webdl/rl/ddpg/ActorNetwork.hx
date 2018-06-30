package webdl.rl.ddpg;
import webdl.core.Tensor;
import webdl.core.WebDL;
import webdl.core.layer.DenseLayer;

/**
 * An actor network (or the target network of an actor network).
 */
class ActorNetwork {
	public var layers(default, null):Array<DenseLayer>;
	public var inState(default, null):Tensor;
	public var outAction(default, null):Tensor;
	public var trainables(default, null):Array<Tensor>;
	public var tensorsToSave(default, null):Array<Tensor>;
	public var stateDim(default, null):Int;
	public var actionDim(default, null):Int;
	public var updates(default, null):Array<Tensor>;
	public var isTraining(default, null):Tensor;

	public function new(layers:Array<DenseLayer>, inState:Tensor, outAction:Tensor, stateDim:Int, actionDim:Int, updates:Array<Tensor>, isTraining:Tensor) {
		this.layers = layers;
		this.inState = inState;
		this.outAction = outAction;
		this.stateDim = stateDim;
		this.actionDim = actionDim;
		this.updates = updates;
		this.isTraining = isTraining;
		trainables = WebDL.getTrainableVariables([outAction]);
		tensorsToSave = WebDL.getTensorsToSave([outAction]);
		for (layer in layers) layer.init();
	}

	public function dump():Void {
		trace("actor network info");
		for (layer in layers) {
			trace("kernel: " + layer.kernel.print());
			trace("bias: " + layer.bias.print());
		}
	}
}
