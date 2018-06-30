package webdl.core.nn;

/**
 * This initializes weights with specified uniform distribution.
 */
class UniformInitializer implements Initializer {
	public var numInputs:Int;
	var range:Float;

	public function new(range:Float) {
		this.range = range;
	}

	public function next():Float {
		return RandUtil.uniform(-range, range);
	}

}
