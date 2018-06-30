package webdl.core.graph;
import webdl.core.Tensor;

/**
 * A node of a computation graph.
 */
class Node {
	@:allow(webdl.core)
	var input:Edge;
	@:allow(webdl.core)
	var outputs:Array<Edge>;
	@:allow(webdl.core)
	var inQueue:Bool;
	@:allow(webdl.core)
	var time:Int;
	@:allow(webdl.core)
	var tensor:Tensor;

	public function new(tensor:Tensor) {
		this.tensor = tensor;
		input = null;
		outputs = [];
		inQueue = false;
		time = -1;
	}

	@:allow(webdl.core)
	function collectForward(time:Int, nodes:Array<Node>, edges:Array<Edge>):Void {
		if (this.time == time) return;
		this.time = time;

		inQueue = false;
		nodes.push(this);

		if (input != null) {
			input.collectForward(time, nodes, edges);
		}
	}

	@:allow(webdl.core)
	function collectBackward(time:Int, queue:Array<Node>, edges:Array<Edge>):Void {
		if (this.time == time) return;
		this.time = time;

		if (input != null) {
			input.collectBackward(time, queue, edges);
		}
	}

}
