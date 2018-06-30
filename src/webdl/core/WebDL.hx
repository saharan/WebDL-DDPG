package webdl.core;
import haxe.ds.Vector;
import webdl.core.backend.Backend;
import webdl.core.graph.Graph;
import webdl.core.nn.Activation;

/**
 * This class is used to create tensors, construct a graph, and run any operations on the graph.
 */
class WebDL {
	static var backend:Backend = null;

	/**
	 * Sets computation backend to the specified implementation.
	 * - `cpu`: use CPU for operations
	 * - `gpu`: use GPU through WebGL(js) or CUDA(python) for operations
	 */
	public static function setBackend(backend:String):Void {
		WebDL.backend = switch (backend.toLowerCase()) {
		case "cpu":
			null; // TODO: CPU is not supported yet lol
		case "gpu":
			#if js
				new webdl.core.backend.gpu.GpuBackend();
			#elseif python
				new webdl.core.backend.cuda.CudaBackend();
			#else
				throw "GPU backend is not supported on this platform";
			#end
		case _:
			throw "invalid backend: " + backend;
		}
	}

	/**
	 * Returns a tensor of the shape. `-1` in `shape` represents
	 * unspecified dimension size.
	 */
	public static function tensorOfShape(shape:Array<Int>):Tensor {
		for (dimSize in shape) if (dimSize != -1 && dimSize <= 0) throw "all the dimension size must be positive";
		return new Tensor(backend, shape);
	}

	/**
	 * Returns a tensor of the rank. All the dimension size of
	 * the tensor are set to `-1` (unspecified).
	 */
	public static function tensorOfRank(rank:Int):Tensor {
		if (rank < 0) throw "invalid rank";
		return new Tensor(backend, [for (i in 0...rank) -1]);
	}

	/**
	 * Returns the tensor with `value`. All the dimension size of
	 * the tensor are fixed.
	 */
	public static function tensorOfValue(value:Any):Tensor {
		if (Std.is(value, Float)) {
			var res:Tensor = tensorOfRank(0);
			res.set0D(cast value);
			return res;
		} else if (Std.is(value, Array)) {
			var value1D:Array<Float> = cast value;
			if (Std.is(value1D[0], Array)) {
				var value2D:Array<Array<Float>> = cast value;
				if (Std.is(value2D[0][0], Array)) {
					var value3D:Array<Array<Array<Float>>> = cast value;
					if (Std.is(value3D[0][0][0], Array)) {
						var value4D:Array<Array<Array<Array<Float>>>> = cast value;
						var res:Tensor = tensorOfShape([value4D.length, value4D[0].length, value4D[0][0].length, value4D[0][0][0].length]);
						res.set4D(value4D);
						return res;
					}
					var res:Tensor = tensorOfShape([value3D.length, value3D[0].length, value3D[0][0].length]);
					res.set3D(value3D);
					return res;
				}
				var res:Tensor = tensorOfShape([value2D.length, value2D[0].length]);
				res.set2D(value2D);
				return res;
			}
			var res:Tensor = tensorOfShape([value1D.length]);
			res.set1D(value1D);
			return res;
		}
		throw "invalid value";
	}

	/**
	 * Returns the tensor of the same shape of `t`.
	 */
	public static function tensorLike(t:Tensor):Tensor {
		return tensorOfShape(t.shape.toArray());
	}

	// --- <graph constructions> ---

	/**
	 * Returns element-wise `a + b`.
	 */
	public static function add(a:Tensor, b:Tensor):Tensor {
		var res:Tensor = new Tensor(backend, ShapeInference.inferShapeBinOp(a.shape, b.shape));
		backend.add(a, b, res);
		return res;
	}

	/**
	 * Returns `a + b`.
	 */
	public static function addConst(a:Tensor, b:Float):Tensor {
		var res:Tensor = new Tensor(backend, a.shape.toArray());
		backend.addConst(a, b, res);
		return res;
	}

	/**
	 * Returns element-wise `a - b`.
	 */
	public static function sub(a:Tensor, b:Tensor):Tensor {
		var res:Tensor = new Tensor(backend, ShapeInference.inferShapeBinOp(a.shape, b.shape));
		backend.sub(a, b, res);
		return res;
	}

	/**
	 * Returns `a - b`.
	 */
	public static function subConst(a:Tensor, b:Float):Tensor {
		var res:Tensor = new Tensor(backend, a.shape.toArray());
		backend.addConst(a, -b, res);
		return res;
	}

	/**
	 * Returns element-wise `aScale * a + bScale * b`.
	 */
	public static function linComb(a:Tensor, b:Tensor, aScale:Float, bScale:Float):Tensor {
		var res:Tensor = new Tensor(backend, ShapeInference.inferShapeBinOp(a.shape, b.shape));
		backend.linComb(a, b, res, aScale, bScale);
		return res;
	}

	/**
	 * Returns element-wise `a * b`.
	 */
	public static function mul(a:Tensor, b:Tensor):Tensor {
		var res:Tensor = new Tensor(backend, ShapeInference.inferShapeBinOp(a.shape, b.shape));
		backend.mul(a, b, res);
		return res;
	}

	/**
	 * Returns `a * b`.
	 */
	public static function mulConst(a:Tensor, b:Float):Tensor {
		var res:Tensor = new Tensor(backend, a.shape.toArray());
		backend.mulConst(a, b, res);
		return res;
	}

	/**
	 * Returns element-wise `a / b`.
	 */
	public static function div(a:Tensor, b:Tensor):Tensor {
		var res:Tensor = new Tensor(backend, ShapeInference.inferShapeBinOp(a.shape, b.shape));
		backend.div(a, b, res);
		return res;
	}

	/**
	 * Returns `a / b`.
	 */
	public static function divConst(a:Tensor, b:Float):Tensor {
		var res:Tensor = new Tensor(backend, a.shape.toArray());
		backend.mulConst(a, 1 / b, res);
		return res;
	}

	/**
	 * Returns element-wise `pow(a, b)`.
	 */
	public static function pow(a:Tensor, b:Tensor):Tensor {
		var res:Tensor = new Tensor(backend, ShapeInference.inferShapeBinOp(a.shape, b.shape));
		backend.pow(a, b, res);
		return res;
	}

	/**
	 * Returns `pow(a, b)`.
	 */
	public static function powConst(a:Tensor, b:Float):Tensor {
		var res:Tensor = new Tensor(backend, a.shape.toArray());
		backend.powConst(a, b, res);
		return res;
	}

	/**
	 * Returns the matrix multiplication of `a` and `b`. `a` and `b` must have
	 * ranks higher than or equal to `2`.
	 */
	public static function matMul(a:Tensor, b:Tensor):Tensor {
		var res:Tensor = new Tensor(backend, ShapeInference.inferShapeMatMul(a.shape, b.shape));
		backend.matMul(a, b, res);
		return res;
	}

	/**
	 * Returns the tensor contraction of `a` and `b`. You must specify
	 * either `count` or `axes`.
	 *
	 * If `count` is specified, this contracts the last `count` axes
	 * of `a` and the first `count` axes of `b`.
	 *
	 * If `axes` is specified, this contracts axes in `axes[0]` of `a`
	 * and axes in `axes[1]` of `b`.
	 */
	public static function tensorDot(a:Tensor, b:Tensor, ?count:Int = -1, ?axes:Array<Array<Int>> = null):Tensor {
		var res:Tensor = new Tensor(backend, ShapeInference.inferShapeTensorDot(a.shape, b.shape, count, axes));
		backend.tensorDot(a, b, res, count, axes);
		return res;
	}

	/**
	 * Returns element-wise `|a|`.
	 */
	public static function abs(a:Tensor):Tensor {
		var res:Tensor = new Tensor(backend, a.shape.toArray());
		backend.abs(a, res);
		return res;
	}

	/**
	 * Returns element-wise `log(a)`.
	 */
	public static function log(a:Tensor):Tensor {
		var res:Tensor = new Tensor(backend, a.shape.toArray());
		backend.log(a, res);
		return res;
	}

	/**
	 * Returns element-wise `a * a`.
	 */
	public static function square(a:Tensor):Tensor {
		var res:Tensor = new Tensor(backend, a.shape.toArray());
		backend.mul(a, a, res);
		return res;
	}

	/**
	 * Returns element-wise `exp(a)`.
	 */
	public static function exp(a:Tensor):Tensor {
		var res:Tensor = new Tensor(backend, a.shape.toArray());
		backend.exp(a, res);
		return res;
	}

	// ---

	/**
	 * Returns element-wise `activ(a)`. Where `activ` is an activation
	 * function corresponded to `activation`.
	 */
	public static function activation(a:Tensor, activation:Activation):Tensor {
		var res:Tensor = new Tensor(backend, a.shape.toArray());
		backend.activation(a, res, activation);
		return res;
	}

	/**
	 * Adds 1-D bias `b` to each lowest dimension of `a` element-wise,
	 * and returns the result.
	 */
	public static function biasAdd(a:Tensor, b:Tensor):Tensor {
		var res:Tensor = new Tensor(backend, a.shape.toArray());
		backend.biasAdd(a, b, res);
		return res;
	}

	// ---

	/**
	 * Returns the sum of the elements of `a` along the `axis`, reducing one dimension.
	 */
	public static function reduceSum(a:Tensor, axis:Int, keepDim:Bool = false):Tensor {
		var res:Tensor = new Tensor(backend, ShapeInference.inferShapeReduce(a.shape, axis, keepDim));
		backend.reduceSum(a, res, axis, keepDim);
		return res;
	}

	/**
	 * Returns the mean of the elements of `a` along the `axis`, reducing one dimension.
	 */
	public static function reduceMean(a:Tensor, axis:Int, keepDim:Bool = false):Tensor {
		var res:Tensor = new Tensor(backend, ShapeInference.inferShapeReduce(a.shape, axis, keepDim));
		backend.reduceMean(a, res, axis, keepDim);
		return res;
	}

	/**
	 * Splits `a` into tensors along the `axis` with each size `sizes`, and returns them.
	 */
	public static function split(a:Tensor, axis:Int, sizes:Array<Int>):Array<Tensor> {
		var shapes:Array<Array<Int>> = ShapeInference.inferShapesSplit(a.shape, axis, sizes);
		var res:Array<Tensor> = [];
		for (shape in shapes) {
			res.push(new Tensor(backend, shape));
		}
		backend.split(a, res, axis, sizes);
		return res;
	}

	/**
	 * Merges tensors `as` into a tensor along the `axis` and returns it.
	 */
	public static function merge(as:Array<Tensor>, axis:Int):Tensor {
		var res:Tensor = new Tensor(backend, ShapeInference.inferShapeMerge(as.map((t) -> t.shape), axis));
		backend.merge(as, res, axis);
		return res;
	}

	// ---

	/**
	 * Computes the sum of derivatives of `y` w.r.t. `xs`.
	 * Note that this operation is **not differentiable**, so you cannot
	 * compute gradients of gradients. If `gradY` is specified, the initial
	 * gradient of `y` will be set to `gradY` insted of filled with `1.0`s.
	 */
	public static function gradients(y:Tensor, xs:Array<Tensor>, gradY:Tensor = null):Array<Tensor> {
		var res:Array<Tensor> = [for (x in xs) new Tensor(backend, x.shape.toArray())];
		backend.gradients(y, xs, res, gradY);
		return res;
	}

	/**
	 * Copies the value from `src` to `dst`, and returns the updated value of `dst`.
	 */
	public static function assign(dst:Tensor, src:Tensor):Tensor {
		var res:Tensor = new Tensor(backend, src.shape.toArray());
		backend.assign(src, res, dst);
		return res;
	}

	/**
	 * Returns `cond > 0.5 ? a : b` element-wise. Note that `a` and `b` will be
	 * evaluated regardless of `cond`.
	 */
	public static function where(cond:Tensor, a:Tensor, b:Tensor):Tensor {
		var res:Tensor = new Tensor(backend, a.shape.toArray());
		backend.where(cond, a, b, res);
		return res;
	}

	@:allow(webdl.core.optimizer.AdamOptimizer)
	static function adamUpdate(count:Tensor, t:Tensor, g:Tensor, m:Tensor, v:Tensor, alpha:Tensor, beta1:Tensor, beta2:Tensor, epsilon:Tensor, l2Decay:Tensor):Tensor {
		var res:Tensor = new Tensor(backend, []);
		backend.adamUpdate(count, t, g, m, v, alpha, beta1, beta2, epsilon, l2Decay, res);
		return res;
	}

	// --- </graph constructions> ---

	/**
	 * Returns the trainable variables w.r.t. `references`.
	 */
	public static function getTrainableVariables(references:Array<Tensor>):Array<Tensor> {
		return Graph.collectNodesConcerned(references.map((t) -> t.node)).filter((n) -> n.tensor.trainable).map((n) -> n.tensor);
	}

	/**
	 * Returns the tensors to save w.r.t. `references`.
	 */
	public static function getTensorsToSave(references:Array<Tensor>):Array<Tensor> {
		return Graph.collectNodesConcerned(references.map((t) -> t.node)).filter((n) -> n.tensor.shouldBeSaved).map((n) -> n.tensor);
	}

	/**
	 * Evaluates `tensors`. This excutes operations related to (i.e. should be carried out
	 * to evaluate) `tensors`.
	 */
	public static function run(tensors:Array<Tensor>):Void {
		Graph.run(tensors.map((t) -> t.node));
	}

	/**
	 * Exports all elements of `tensors` to a 1-D array. Note that no information about
	 * shapes of `tensors` will not be saved, and that the return value depends on the order
	 * of the `tensors`.
	 */
	public static function exportElements(tensors:Array<Tensor>):Array<Float> {
		var res:Array<Float> = [];
		for (t in tensors) {
			res = res.concat(t.getArray());
		}
		return res;
	}

	/**
	 * Imports all elements of `tensors` from `data`. The size of `data` must equal to the
	 * number of all elements in `tensors`.
	 */
	public static function importElements(tensors:Array<Tensor>, data:Array<Float>):Void {
		var copy:Array<Float> = data.copy();
		for (t in tensors) {
			if (copy.length < t.actualSize) {
				throw "data size too small";
			}
			if (t.actualSize == -1) throw "tensor size unspecified";
			t.setArray(copy.splice(0, t.actualSize));
		}
		if (copy.length > 0) throw "data size too large";
	}

}
