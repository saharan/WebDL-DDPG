package webdl.core.backend.cuda.operation;
import haxe.ds.Vector;
import webdl.core.Tensor;
import webdl.core.backend.cuda.CudaAtomicOperation;

/**
 * ...
 */
class CudaAdamUpdateOperation extends CudaOperation {
	var count:Tensor;
	var t:Tensor;
	var g:Tensor;
	var m:Tensor;
	var v:Tensor;
	var alpha:Tensor;
	var beta1:Tensor;
	var beta2:Tensor;
	var epsilon:Tensor;
	var l2Decay:Tensor;

	public function new(count:Tensor, t:Tensor, g:Tensor, m:Tensor, v:Tensor, alpha:Tensor, beta1:Tensor, beta2:Tensor, epsilon:Tensor, l2Decay:Tensor, dummyDst:Tensor) {
		super([count, t, g, m, v, alpha, beta1, beta2, epsilon, l2Decay], [dummyDst]);
		this.count = count;
		this.t = t;
		this.g = g;
		this.m = m;
		this.v = v;
		this.alpha = alpha;
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.epsilon = epsilon;
		this.l2Decay = l2Decay;

		this.forwardOps = [
			new CudaAtomicOperation([t, count, m, v, g, alpha, beta1, beta2, epsilon, l2Decay], "adam_update", '
				float t       = val(0, idx4);
				float count   = val(1, idx4);
				float m       = val(2, idx4);
				float v       = val(3, idx4);
				float g       = val(4, idx4);
				float alpha   = val(5, idx4);
				float beta1   = val(6, idx4);
				float beta2   = val(7, idx4);
				float epsilon = val(8, idx4);
				float l2Decay = val(9, idx4);

				m = beta1 * m + (1 - beta1) * g;     // update m
				v = beta2 * v + (1 - beta2) * g * g; // update v
				val(2, idx4) = m;
				val(3, idx4) = v;

				float mHat = m / (1 - powf(beta1, count));
				float vHat = v / (1 - powf(beta2, count));
				t -= alpha * (mHat / (sqrtf(vHat) + epsilon) + l2Decay * t); // update t
				val(0, idx4) = t;
			')
		];
		this.backwardOps = [];
	}

	override function shapeCheck():Void {
		shapeEq(t.actualShape, g.actualShape);
		m.assignShape(t.actualShape);
		v.assignShape(t.actualShape);
		var scalarShape:Vector<Int> = new Vector(0);
		shapeEq(alpha.actualShape, scalarShape);
		shapeEq(beta1.actualShape, scalarShape);
		shapeEq(beta2.actualShape, scalarShape);
		shapeEq(epsilon.actualShape, scalarShape);
		shapeEq(l2Decay.actualShape, scalarShape);
	}

	override public function backwardRun():Void {
		throw "differentiation of adam update is not supported";
	}

}
