package webdl.core.nn;

/**
 * An implementation of Xavier's initializer.
 */
class XavierInitializer implements Initializer {
	public var numInputs:Int;

	public function new() {
	}

	public function next():Float {
		return RandUtil.normal(0, 1 / numInputs);
	}
}
