package webdl.core.backend.cuda;
import haxe.io.Float32Array;
import pycuda.DevicePointer;
import pycuda.NdArray;
import pycuda.NumPy;
import pycuda.PyCuda;

/**
 * 1-D array, stored on CUDA device.
 */
class CudaArray {
	public var host(default, null):NdArray;
	public var device(default, null):DevicePointer;
	public var maxSize(default, null):Int;
	public var integer:Bool;

	public function new(maxSize:Int, integer:Bool = false) {
		this.maxSize = maxSize;
		this.integer = integer;
		host = NumPy.array([for (i in 0...maxSize) 0]).asType(integer ? NumPy.INT32 : NumPy.FLOAT32);
		device = PyCuda.memAlloc(host.nbytes);
		PyCuda.memcpyHost2Device(device, host);
	}

	public function downloadData():Void {
		PyCuda.memcpyDevice2Host(host, device);
	}

	@:extern
	inline function float32ToInt32(a:Float):Int {
		var ary:Float32Array = new Float32Array(1);
		ary.set(0, a);
		return ary.view.buffer.getInt32(0);
	}

	public function clear(value:Float):Void {
		if (integer) throw "!?";
		PyCuda.memsetD32(device, float32ToInt32(value), maxSize);
	}

	public function uploadData(array:Array<Float>):Void {
		if (integer) throw "!?";
		if (array.length > maxSize) throw "!?"; // internal error
		var num:Int = array.length;
		for (i in 0...num) {
			host.itemset(i, array[i]);
		}
		PyCuda.memcpyHost2Device(device, host);
	}

	public function uploadDataInt(array:Array<Int>):Void {
		if (!integer) throw "!?";
		if (array.length > maxSize) throw "!?"; // internal error
		var num:Int = array.length;
		for (i in 0...num) {
			host.itemset(i, array[i]);
		}
		PyCuda.memcpyHost2Device(device, host);
	}

	public function dispose():Void {
		device.free();
		host = null;
		device = null;
		maxSize = 0;
	}

}
