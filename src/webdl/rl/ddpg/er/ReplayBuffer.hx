package webdl.rl.ddpg.er;
import haxe.ds.Vector;
import haxe.io.BytesInput;
import haxe.io.BytesOutput;
import webdl.rl.ddpg.er.Experience;
using webdl.rl.ddpg.BytesTools;

/**
 * Storage of experiences to replay.
 */
class ReplayBuffer {
	var maxBufferSize:Int;
	var buffer:Vector<Experience>;
	public var size(default, null):Int;
	public var index(default, null):Int;

	public function new(maxBufferSize:Int) {
		this.maxBufferSize = maxBufferSize;
		buffer = new Vector(maxBufferSize);
		size = 0;
		index = 0;
	}

	public function clear():Void {
		while (size > 0) buffer[--size] = null;
		index = 0;
	}

	public function push(experience:Experience):Void {
		buffer[index] = experience;
		index = (index + 1) % maxBufferSize;
		if (size < maxBufferSize) {
			size++;
		}
	}

	public function sample(num:Int):Array<Experience> {
		var res:Array<Experience> = [];
		for (i in 0...num) {
			res.push(buffer[Std.random(size)]);
		}
		return res;
	}

	public function writeBytes(bo:BytesOutput):Void {
		bo.writeInt32(maxBufferSize);
		bo.writeInt32(size);
		bo.writeInt32(index);
		for (i in 0...size) {
			var e:Experience = buffer[i];
			bo.writeFloatArray(e.state);
			bo.writeFloatArray(e.action);
			bo.writeFloat(e.reward);
			bo.writeFloatArray(e.state2);
			bo.writeByte(e.done ? 1 : 0);
		}
	}

	public static function readBytes(bi:BytesInput):ReplayBuffer {
		var maxBufferSize:Int = bi.readInt32();
		if (maxBufferSize > 3000000) maxBufferSize = 3000000;

		var rb:ReplayBuffer = new ReplayBuffer(maxBufferSize);
		rb.size = bi.readInt32();
		rb.index = bi.readInt32();
		for (i in 0...rb.size) {
			rb.buffer[i] = new Experience(
				bi.readFloatArray(),
				bi.readFloatArray(),
				bi.readFloat(),
				bi.readFloatArray(),
				bi.readByte() != 0
			);
		}
		return rb;
	}
}
