package webdl.core.graph;
import haxe.ds.Vector;
import webdl.core.Operation;

/**
 * ...
 */
class Edge {
	var inputs:Vector<Node>;
	var outputs:Vector<Node>;
	var operation:Operation;
	var time:Int;

	public function new(operation:Operation) {
		this.operation = operation;
		inputs = operation.inputs.map((t) -> t.node);
		outputs = operation.outputs.map((t) -> t.node);

		for (n in inputs) {
			if (n.outputs.indexOf(this) == -1) {
				n.outputs.push(this);
			}
		}
		for (n in outputs) {
			if (n.input != null) {
				throw "outputs conflict";
			}
			n.input = this;
		}

		time = -1;
	}

	@:allow(webdl.core)
	function collectForward(time:Int, nodes:Array<Node>, edges:Array<Edge>):Void {
		if (this.time == time) return;
		this.time = time;

		// look into output tensors as well as inputs in order to capture all tensors concerned
		for (n in outputs) {
			n.collectForward(time, nodes, edges);
		}

		for (n in inputs) {
			n.collectForward(time, nodes, edges);
		}
		edges.push(this);
	}

	@:allow(webdl.core)
	function collectBackward(time:Int, queue:Array<Node>, edges:Array<Edge>):Void {
		if (this.time == time) return;
		this.time = time;

		edges.push(this);
		for (n in inputs) {
			if (!n.inQueue) {
				queue.push(n);
				n.inQueue = true;
			}
		}
	}

	@:allow(webdl.core)
	function run():Void {
		operation.run();
	}

	@:allow(webdl.core)
	function onAfterRun():Void {
		operation.onAfterRun();
	}

	@:allow(webdl.core)
	function backward():Void {
		operation.backwardRun();
	}

}
