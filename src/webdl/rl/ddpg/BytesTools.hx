package webdl.rl.ddpg;
import haxe.io.BytesInput;
import haxe.io.BytesOutput;

/**
 * ...
 */
class BytesTools {
	public static function writeFloatArray(bo:BytesOutput, array:Array<Float>):Void {
		bo.writeInt32(array.length);
		for (a in array) {
			if (!Math.isFinite(a)) throw "float must be finite";
			bo.writeFloat(a);
		}
	}

	public static function readFloatArray(bi:BytesInput):Array<Float> {
		var len:Int = bi.readInt32();
		var res:Array<Float> = [];
		for (i in 0...len) {
			var a:Float = bi.readFloat();
			if (!Math.isFinite(a)) throw "float must be finite";
			if (Math.abs(a) > 1e+37) a = 0; // due to haxe's bug (fixed in 4.0.0-preview.4)
			res.push(a);
		}
		return res;
	}
}
