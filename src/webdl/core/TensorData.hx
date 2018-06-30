package webdl.core;

/**
 * Internal 1-D-flattened data of a tensor.
 */
interface TensorData {
	public var maxSize(default, null):Int;

	public function isPreferableSize(size:Int):Bool;

	public function getValue(size:Int):Array<Float>;

	public function setValue(value:Array<Float>):Void;

	public function clearValue(value:Float):Void;

	public function getDiff(size:Int):Array<Float>;

	public function setDiff(diff:Array<Float>):Void;

	public function clearDiff(diff:Float):Void;
}
