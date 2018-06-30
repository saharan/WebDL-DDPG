package webdl.core.layer;
import haxe.ds.Vector;
import webdl.core.Tensor;

/**
 * A base class of layers.
 */
class Layer {
	/**
	 * The input tensor.
	 */
	public var input(default, null):Tensor;

	/**
	 * The output tensor.
	 */
	public var output(default, null):Tensor;

	/**
	 * The update operations of the layer.
	 */
	public var updates:Array<Tensor>;

	function new() {
		updates = [];
	}

	/**
	 * Initializes the layer.
	 */
	public function init():Void {
	}

}
