package webdl.core.backend.cuda.operation;
import webdl.core.Tensor;

/**
 * ...
 */
class BroadcastBackpropData {
	public var src:Tensor;
	public var dst:Tensor;
	public var axes:Array<Int>;
	public var intermediates:Array<Tensor>;
	public var ops:Array<CudaAtomicOperation>;
	public var scalar:Bool;

	public function new(src:Tensor, dst:Tensor, axes:Array<Int>, intermediates:Array<Tensor>, ops:Array<CudaAtomicOperation>, scalar:Bool) {
		this.src = src;
		this.dst = dst;
		this.axes = axes;
		this.intermediates = intermediates;
		this.ops = ops;
		this.scalar = scalar;
	}

}
