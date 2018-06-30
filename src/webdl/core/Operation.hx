package webdl.core;
import haxe.ds.Vector;
import webdl.core.graph.Edge;

/**
 * ...
 */
class Operation {
	public var inputs:Vector<Tensor>;
	public var outputs:Vector<Tensor>;
	public var name:String;
	@:allow(webdl.core)
	var edge:Edge;

	public function new(inputs:Array<Tensor>, outputs:Array<Tensor>) {
		this.inputs = Vector.fromArrayCopy(inputs);
		this.outputs = Vector.fromArrayCopy(outputs);
		edge = new Edge(this);
		name = "op";
	}

	/**
	 * Runs the operation. Do not directly call this (use `WebDL.run`).
	 */
	public function run():Void {
	}

	/**
	 * Called when backprop of the operation is required.
	 */
	@:allow(webdl.core)
	function backwardRun():Void {
		throw "backward operation is not supported";
	}

	/**
	 * Called after all operation's `run` are called.
	 */
	@:allow(webdl.core)
	function onAfterRun():Void {
	}

}
