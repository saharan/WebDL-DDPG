package webdl.core.backend.cuda;
import webdl.core.TensorData;

class CudaTensorData implements TensorData {
	public var maxSize(default, null):Int;
	public var val(default, null):CudaArray;   // value
	public var dif(default, null):CudaArray;   // diff
	public var shape(default, null):CudaArray; // shape

	public function new(requestedSize:Int) {
		maxSize = 1;
		while (maxSize < requestedSize) {
			maxSize *= 2;
		}
		val = new CudaArray(maxSize);
		dif = new CudaArray(maxSize);
		shape = new CudaArray(4, true); // 0 = minor, ..., 3 = major
	}

	public function isPreferableSize(size:Int):Bool {
		return size <= maxSize && size * 2 > maxSize;
	}

	public function getValue(size:Int):Array<Float> {
		if (size > maxSize) throw "max size exceeded";
		val.downloadData();
		var array:Array<Float> = cast val.host.toArray();
		return array.slice(0, size); // extract from `0` until `size`
	}

	public function setValue(value:Array<Float>):Void {
		if (value.length > maxSize) throw "max size exceeded";
		val.uploadData(value);
	}

	public function clearValue(value:Float):Void {
		val.clear(value);
	}

	public function getDiff(size:Int):Array<Float> {
		if (size > maxSize) throw "max size exceeded";
		dif.downloadData();
		var array:Array<Float> = cast dif.host.toArray();
		return array.slice(0, size); // extract from `0` until `size`
	}

	public function setDiff(diff:Array<Float>):Void {
		if (diff.length > maxSize) throw "max size exceeded";
		dif.uploadData(diff);
	}

	public function clearDiff(diff:Float):Void {
		dif.clear(diff);
	}

	public function dispose():Void {
		val.dispose();
		dif.dispose();
		shape.dispose();
	}

}
