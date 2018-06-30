package pycuda;

/**
 * ...
 */
@:pythonImport("numpy")
extern class NumPy {
	@:native("bool")     public static var BOOL    (default, never):DType;
	@:native("int8")     public static var INT8    (default, never):DType;
	@:native("int16")    public static var INT16   (default, never):DType;
	@:native("int32")    public static var INT32   (default, never):DType;
	@:native("int64")    public static var INT64   (default, never):DType;
	@:native("uint8")    public static var UINT8   (default, never):DType;
	@:native("uint16")   public static var UINT16  (default, never):DType;
	@:native("uint32")   public static var UINT32  (default, never):DType;
	@:native("uint64")   public static var UINT64  (default, never):DType;
	@:native("float16")  public static var FLOAT16 (default, never):DType;
	@:native("float32")  public static var FLOAT32 (default, never):DType;
	@:native("float64")  public static var FLOAT64 (default, never):DType;
	@:native("float128") public static var FLOAT128(default, never):DType;

	public static function array(list:Array<Any>):NdArray;
	@:native("empty_like")
	public static function emptyLike(a:NdArray, ?dtype:DType):NdArray;
}
