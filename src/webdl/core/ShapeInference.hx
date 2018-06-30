package webdl.core;
import haxe.ds.Vector;

/**
 * Utility class to infer shapes of tensors.
 */
class ShapeInference {
	/**
	 * Returns the shape of the tensor obtained by a broadcastable binary operation of tensors whose shapes are `a` and `b`.
	 */
	public static function inferShapeBinOp(a:Vector<Int>, b:Vector<Int>):Array<Int> {
		return broadcast(a, b);
	}

	/**
	 * Returns the shape of the tensor obtained by a matrix multiplication of tensors whose shapes are `a` and `b`.
	 */
	public static function inferShapeMatMul(a:Vector<Int>, b:Vector<Int>):Array<Int> {
		if (a.length < 2 || b.length < 2) throw "insufficient ranks";
		if (a.length != b.length) throw "ranks mismatch";

		var aLast:Int = a.length - 1;
		var bLast:Int = b.length - 1;
		if (a[aLast] != -1 && b[bLast - 1] != -1 && a[aLast] != b[bLast - 1]) throw "cannot multiply matrices";

		var res:Array<Int> = a.toArray();
		res[aLast - 1] = a[aLast - 1];
		res[bLast] = b[bLast];
		for (i in 0...a.length - 2) {
			res[i] = a[i] == -1 ? b[i] : a[i];
		}
		return res;
	}

	/**
	 * Returns the shape of the tensor obtained by a tensor contraction of tensors whose shapes are `a` and `b`.
	 */
	public static function inferShapeTensorDot(a:Vector<Int>, b:Vector<Int>, count:Int = -1, axes:Array<Array<Int>> = null):Array<Int> {
		if (count == -1 && axes == null || count != -1 && axes != null) throw "specify either count or axes";

		var res:Array<Int> = [];
		if (count != -1) {
			if (a.length < count || b.length < count) throw "insufficient ranks";
			for (i in 0...count) {
				var adummySize:Int = a[a.length - count + i];
				var bdummySize:Int = b[i];
				if (adummySize != -1 && bdummySize != -1 && adummySize != bdummySize) throw "cannot multiply tensors";
			}

			for (i in 0...a.length - count) {
				res.push(a[i]);
			}
			for (i in count...b.length) {
				res.push(b[i]);
			}
		} else {
			if (axes.length != 2) throw "invalid argument";
			if (axes[0].length != axes[1].length) throw "invalid axes";
			for (i in 0...axes[0].length) {
				var adummyAxis:Int = axes[0][i];
				var bdummyAxis:Int = axes[1][i];
				if (adummyAxis < 0 || adummyAxis >= a.length) throw "invalid axes";
				if (bdummyAxis < 0 || bdummyAxis >= b.length) throw "invalid axes";
				var adummySize:Int = a[adummyAxis];
				var bdummySize:Int = b[bdummyAxis];
				if (adummySize != -1 && bdummySize != -1 && adummySize != bdummySize) throw "cannot multiply tensors";
			}

			for (i in 0...a.length) {
				if(axes[0].indexOf(i) == -1) res.push(a[i]);
			}
			for (i in 0...b.length) {
				if(axes[1].indexOf(i) == -1) res.push(b[i]);
			}
		}
		return res;
	}

	/**
	 * Returns the reduced shape of `a` along the axis `axis`.
	 */
	public static function inferShapeReduce(a:Vector<Int>, axis:Int, keepDim:Bool):Array<Int> {
		if (axis < 0 || axis >= a.length) throw "invalid axis";

		var res:Array<Int>;

		if (keepDim) {
			res = a.toArray();
			res[axis] = 1;
			return res;
		}

		res = [];
		for (i in 0...a.length) {
			if (i != axis) res.push(a[i]);
		}
		return res;
	}

	/**
	 * Returns the splitted shapes of `a` along the `axis` with each size from `sizes`.
	 */
	public static function inferShapesSplit(a:Vector<Int>, axis:Int, sizes:Array<Int>):Array<Array<Int>> {
		if (axis < 0 || axis >= a.length) throw "invalid axis";
		var res:Array<Array<Int>> = [];
		var sizeSum:Int = 0;
		for (size in sizes) {
			var shape:Array<Int> = a.toArray();
			shape[axis] = size;
			res.push(shape);
			sizeSum += size;
		}
		if (sizeSum != a[axis]) throw "invalid sizes: sum of the split sizes " + sizeSum + " should equal to the dimension size along the splitting axis " + a[axis];
		return res;
	}

	/**
	 * Returns the merged shape of `as` along the `axis`.
	 */
	public static function inferShapeMerge(as:Array<Vector<Int>>, axis:Int):Array<Int> {
		if (axis < 0 || axis >= as[0].length) throw "invalid axis";
		var res:Array<Int> = as[0].toArray();
		var totalSize:Int = 0;
		for (a in as) {
			if (a[axis] == -1) {
				totalSize = -1;
				break;
			}
			totalSize += a[axis];
		}
		res[axis] = totalSize;
		return res;
	}

	/**
	 * Returns the broadcasted shape between `a` and `b`, with actual shapes `aActual` and `bActual`. Actual shapes are
	 * required separately as broadcasting is disabled for a dimension of size `-1`, even if actual dimension size is `1`.
	 */
	public static function broadcast(a:Vector<Int>, b:Vector<Int>, aActual:Vector<Int> = null, bActual:Vector<Int> = null):Array<Int> {
		if ((aActual == null) != (bActual == null)) throw "invalid argument";
		if (aActual != null && bActual != null) {
			if (aActual.length == 0) return b.toArray();
			if (bActual.length == 0) return a.toArray();
			if (aActual.length != bActual.length) throw "ranks mismatch";
			var res:Array<Int> = [];
			for (i in 0...a.length) {
				var ai:Int = a[i];
				var bi:Int = b[i];
				var aiActual:Int = aActual[i];
				var biActual:Int = bActual[i];
				switch ([ai, bi]) {
				case [1, 1]:
					res.push(1);
				case [1, _]:
					res.push(biActual);
				case [_, 1]:
					res.push(aiActual);
				case _:
					if (aiActual == biActual) res.push(aiActual);
					else {
						if (aiActual == 1 || biActual == 1) throw "shapes mismatch; cannot broadcast along an unspecified dimension";
						else throw "shapes mismatch";
					}
				}
			}
			return res;
		} else {
			if (a.length == 0) return b.toArray();
			if (b.length == 0) return a.toArray();
			if (a.length != b.length) throw "ranks mismatch";
			var res:Array<Int> = [];
			for (i in 0...a.length) {
				var ai:Int = a[i];
				var bi:Int = b[i];
				switch ([ai, bi]) {
				case [1, -1] | [-1, 1]:
					res.push(-1);
				case [1, n] | [n, 1] | [-1, n] | [n, -1]:
					res.push(n);
				case _:
					if (ai == bi) res.push(ai);
					else throw "shapes mismatch";
				}
			}
			return res;
		}
	}

	/**
	 * Returns the broadcasted axes assuming the broadcasted shape is `dst` and the original shape is `a`.
	 */
	public static function getBroadcastedAxes(dst:Vector<Int>, a:Vector<Int>):Array<Int> {
		if (a.length == 0) {
			// scalar broadcasting
			return [for (i in 0...dst.length) if (dst[i] != 1) i];
		}
		var res:Array<Int> = [];
		for (i in 0...a.length) {
			if (dst[i] != 1 && a[i] == 1) res.push(i);
		}
		return res;
	}
}
