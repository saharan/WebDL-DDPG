package pycuda;

/**
 * test
 */
@:pythonImport("pycuda.compiler", "SourceModule")
extern class SourceModule {
	public function new(source:String);
	@:native("get_function")
	public function getFunction(name:String):Function;
}
