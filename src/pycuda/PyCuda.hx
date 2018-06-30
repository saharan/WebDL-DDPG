package pycuda;

/**
 * test
 */
@:pythonImport("pycuda.driver")
extern class PyCuda {
	@:native("mem_alloc")
	public static function memAlloc(num:Int):DevicePointer;
	@:native("memcpy_htod")
	public static function memcpyHost2Device(dest:DevicePointer, src:NdArray):Void;
	@:native("memcpy_dtoh")
	public static function memcpyDevice2Host(dest:NdArray, src:DevicePointer):Void;
	@:native("memset_d32")
	public static function memsetD32(dest:DevicePointer, data:Int, numElements:Int):Void;
}
