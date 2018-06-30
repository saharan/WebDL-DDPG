package webdl.core.backend.cuda.operation;
import webdl.core.Tensor;
import webdl.core.backend.cuda.CudaAtomicOperation;

/**
 * ...
 */
class CudaAssignOperation extends CudaOperation {
	var a:Tensor;
	var dummyDst:Tensor;
	var dst:Tensor;
	var assignOp:CudaAtomicOperation;

	public function new(a:Tensor, dummyDst:Tensor, dst:Tensor) {
		super([a], [dummyDst]);
		this.a = a;
		this.dummyDst = dummyDst;
		this.dst = dst;
		this.forwardOps = [
			new CudaAtomicOperation([dummyDst, a], "assign_forward_dummy", '
				val(0, idx4) = val(1, idx4);
			')
		];
		this.backwardOps = [
			new CudaAtomicOperation([a, dummyDst], "assign_backward", '
				float d = dif(1, idx4);
				dif(0, idx4) += d;
			')
		];
		assignOp = new CudaAtomicOperation([dst, a], "assign_forward", '
			val(0, idx4) = val(1, idx4);
		');
	}

	override function shapeCheck():Void {
		dst.assignShape(a.actualShape);
		dummyDst.assignShape(a.actualShape);
	}

	override function onAfterRun():Void {
		assignOp.run();
	}

}
