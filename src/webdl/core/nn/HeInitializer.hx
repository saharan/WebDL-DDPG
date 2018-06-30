package webdl.core.nn;

/**
 * An implementation of He's initializer.
 */
class HeInitializer implements Initializer {
	public var numInputs:Int;

	public function new() {
	}

	public function next():Float {
		return RandUtil.normal(0, 2 / numInputs);
	}
}
