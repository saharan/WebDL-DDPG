package pycuda;

/**
 * ...
 */
extern class NdArray {
	public var nbytes(default, never):Int;
	public var dtype(default, never):DType;
	@:native("astype")
	public function asType(dtype:DType):NdArray;
	@:native("tolist")
	public function toArray():Array<Any>;
	@:overload(function(val:Any):Void {})
	public function itemset(index:Int, val:Any):Void;
}
