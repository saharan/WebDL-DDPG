package webdl.rl.ddpg.er;
import haxe.ds.Vector;
import webdl.rl.ddpg.er.FenwickTree;

/**
 * buggy, do not use :(
 */
class PrioritizedReplayBuffer {
	public static inline var ALPHA:Float = 0.6;
	public static inline var EPSILON:Float = 0.01;
	public var size(default, null):Int;
	var bufferSize:Int;
	var exps:Vector<Experience>;
	var tree:FenwickTree;
	var pointer:Int;
	var sampleCount:Int;
	var filled:Bool;

	public function new(bufferSize:Int) {
		this.bufferSize = bufferSize;
		exps = new Vector(bufferSize);
		tree = new FenwickTree(bufferSize);
		pointer = 0;
		size = 0;
		sampleCount = 0;
		filled = false;
	}

	public function push(e:Experience):Void {
		if (pointer == bufferSize) {
			filled = true;
			pointer = 0;
		}
		exps[pointer] = e;
		e.index = pointer;
		e.p = Math.pow(e.absTdError + EPSILON, ALPHA);
		tree.set(pointer, e.p);
		size = filled ? bufferSize : pointer;
		pointer++;
	}

	public function update(es:Array<Experience>):Void {
		for (e in es) {
			e.p = Math.pow(e.absTdError + EPSILON, ALPHA);
			tree.set(e.index, e.p);
		}
	}

	public function sample(num:Int):Array<Experience> {
		if (++sampleCount % 500 == 0) {
			tree.refresh(); // get rid of floating point errors
		}
		var res:Array<Experience> = [];
		var invSum:Float = 1 / tree.getSum(size);
		for (i in 0...num) {
			var x:Float = Math.random() * tree.getSum(size);
			// find minimum n that satisfies Σ_{i=1}^n p_i > x by binary search
			var min:Int = -1; // exclusive
			var max:Int = size - 1; // inclusive
			while (max - min > 1) {
				var mid:Int = min + max >> 1;
				if (tree.getSum(mid) > x) {
					max = mid; // ok, set inclusive
				} else {
					min = mid; // ng, set exclusive
				}
			}
			res.push(exps[max]);
			tree.set(max, 0); // set value to 0 so as not to select an experiment twice
		}
		for (e in res) {
			tree.set(e.index, e.p); // restore value
		}

		return res;
	}

}
