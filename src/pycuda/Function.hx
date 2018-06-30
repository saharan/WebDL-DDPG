package pycuda;
import python.KwArgs;
import python.Tuple;
import python.Tuple.Tuple3;
import python.VarArgs;

/**
 * ...
 */
extern class Function {
	public function prepare(argTypes:String, shared:Any = null, ?texrefs:Array<Any> = []):Void;
	@:native("prepared_call")
	public function preparedCall(grid:Tuple<Int>, block:Tuple<Int>, args:VarArgs<Dynamic>):Void;
}
