package webdl.rl.ddpg.er;
import haxe.ds.Vector;

/**
 * ...
 */
class FenwickTree {
	var maxSize:Int;
	public var tree:Vector<Float>;
	public var rawValues:Vector<Float>;

	public function new(maxSize:Int) {
		this.maxSize = maxSize;
		tree = new Vector(maxSize);
		rawValues = new Vector(maxSize);
		for (i in 0...maxSize) {
			tree[i] = 0;
			rawValues[i] = 0;
		}
	}

	public function set(i:Int, value:Float):Void {
		var diff:Float = value - rawValues[i];
		rawValues[i] = value;
		i++;
		while (i <= maxSize) {
			tree[i - 1] += diff;
			i += i & -i;
		}
	}

	public function get(i:Int):Float {
		return rawValues[i];
	}

	public function getSum(i:Int):Float {
		i++;
		var sum:Float = 0;
		while (i >= 1) {
			sum += tree[i - 1];
			i -= i & -i;
		}
		return sum;
	}

	public function refresh():Void {
		for (i in 0...maxSize) {
			tree[i] = rawValues[i];
		}
		for (i in 1...maxSize) {
			tree[i - 1 + (i & -i)] += tree[i - 1];
		}
	}

}
