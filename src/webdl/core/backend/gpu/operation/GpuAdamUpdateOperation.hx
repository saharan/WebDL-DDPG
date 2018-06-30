package webdl.core.backend.gpu.operation;
import haxe.ds.Vector;
import webdl.core.Tensor;

/**
 * ...
 */
class GpuAdamUpdateOperation extends GpuOperation {
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

	public function new(backend:GpuBackend, count:Tensor, t:Tensor, g:Tensor, m:Tensor, v:Tensor, alpha:Tensor, beta1:Tensor, beta2:Tensor, epsilon:Tensor, l2Decay:Tensor, dummyDst:Tensor) {
		super(backend, [count, t, g, m, v, alpha, beta1, beta2, epsilon, l2Decay], [dummyDst]);
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
			fop([g, m, beta1], m, "adam_update_m", '
				float run(ivec4 idx4) {
					float g = src1(idx4).value;
					float m = src2(idx4).value;
					float beta1 = src3(idx4).value;
					return beta1 * m + (1.0 - beta1) * g;
				}
			'),
			fop([g, v, beta2], v, "adam_update_v", '
				float run(ivec4 idx4) {
					float g = src1(idx4).value;
					float v = src2(idx4).value;
					float beta2 = src3(idx4).value;
					return beta2 * v + (1.0 - beta2) * g * g;
				}
			'),
			fop([t, count, m, v, alpha, beta1, beta2, epsilon, l2Decay], t, "adam_update_t", '
				float run(ivec4 idx4) {
					float t       = src1(idx4).value;
					float count   = src2(idx4).value;
					float m       = src3(idx4).value;
					float v       = src4(idx4).value;
					float alpha   = src5(idx4).value;
					float beta1   = src6(idx4).value;
					float beta2   = src7(idx4).value;
					float epsilon = src8(idx4).value;
					float l2Decay = src9(idx4).value;
					float mHat = m / (1.0 - pow(beta1, count));
					float vHat = v / (1.0 - pow(beta2, count));
					return t - alpha * (mHat / (sqrt(vHat) + epsilon) + l2Decay * t);
				}
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
