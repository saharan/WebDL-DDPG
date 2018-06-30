package webdl.test;

/**
 * A part of unit tests
 */
class Test {
	var name:String;
	var f:Void -> Void;

	public function new(name:String, f:Void -> Void) {
		this.name = name;
		this.f = f;
	}

	public function run():Bool {
		try {
			f();
			trace('test passed: $name');
			return true;
		} catch (e:Any) {
			trace('*** test failed: $name');
			trace('    error: $e');
			throw e;
			return false;
		}
	}
}
