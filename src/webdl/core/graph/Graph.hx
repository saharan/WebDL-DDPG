package webdl.core.graph;
import haxe.ds.Vector;
import webdl.core.Tensor;

/**
 * ...
 */
class Graph {
	static var time:Int = 0; // global time

	/**
	 * Computes values of `nodes`.
	 */
	@:overload(function(nodes:Vector<Node>):Void {})
	public static function run(nodes:Array<Node>):Void {
		var unused:Array<Node> = [];
		var edges:Array<Edge> = [];

		time++;
		for (n in nodes) {
			n.collectForward(time, unused, edges);
		}

		for (e in edges) {
			e.run();
		}

		// for assignments
		for (e in edges) {
			e.onAfterRun();
		}
	}

	/**
	 * `y`'s diff must already be initialized
	 */
	@:allow(webdl.core)
	static function backprop(y:Node):Void {
		var nodes:Array<Node> = [];
		var unused:Array<Edge> = [];
		var edges:Array<Edge> = [];

		time++;
		y.collectForward(time, nodes, unused);

		time++;
		var queue:Array<Node> = [];
		queue.push(y);
		while (queue.length > 0) {
			var n:Node = queue.shift();
			n.collectBackward(time, queue, edges);
		}

		for (n in nodes) {
			if (n != y) n.tensor.fillDiff(0);
		}

		for (e in edges) {
			e.backward();
		}
	}

	/**
	 * Returns nodes concerned with `nodes`.
	 */
	public static function collectNodesConcerned(nodes:Array<Node>):Array<Node> {
		var res:Array<Node> = [];
		var unused:Array<Edge> = [];

		time++;
		for (n in nodes) {
			n.collectForward(time, res, unused);
		}

		return res;
	}
}
