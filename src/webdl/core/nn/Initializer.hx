package webdl.core.nn;

/**
 * The interface of a weight initialization method.
 */
interface Initializer {
	/**
	 * The number of input weights.
	 */
	public var numInputs:Int;

	/**
	 * Generates a weight.
	 */
	public function next():Float;
}
