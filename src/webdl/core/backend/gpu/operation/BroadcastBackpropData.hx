package webdl.core.backend.gpu.operation;
import webdl.core.Tensor;
import webdl.core.backend.gpu.GpuAtomicOperation;

/**
 * ...
 */
class BroadcastBackpropData {
	public var src:Tensor;
	public var dst:Tensor;
	public var axes:Array<Int>;
	public var intermediates:Array<Tensor>;
	public var ops:Array<GpuAtomicOperation>;
	public var scalar:Bool;

	public function new(src:Tensor, dst:Tensor, axes:Array<Int>, intermediates:Array<Tensor>, ops:Array<GpuAtomicOperation>, scalar:Bool) {
		this.src = src;
		this.dst = dst;
		this.axes = axes;
		this.intermediates = intermediates;
		this.ops = ops;
		this.scalar = scalar;
	}

}
