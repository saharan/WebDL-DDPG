package webdl.core.backend.gpu.operation;
import haxe.ds.Vector;
import webdl.core.backend.gpu.ShaderConsts.*;

/**
 * ...
 */
class GpuReduceSumOperation extends GpuOperation {
	var a:Tensor;
	var dst:Tensor;
	var axis:Int;
	var keepDim:Bool;

	public function new(backend:GpuBackend, a:Tensor, dst:Tensor, axis:Int, keepDim:Bool) {
		super(backend, [a], [dst]);
		this.a = a;
		this.dst = dst;
		this.keepDim = keepDim;

		this.axis = axis;
		if (axis < 0 || axis >= a.rank) throw "invalid axis";
		var axisComp:String = "xyzw".charAt(a.rank - 1 - axis);
		var axisCompUpper:String = axisComp.toUpperCase();

		if (keepDim) {
			this.forwardOps = [
				fop([a], dst, "reducesum_forward_kd", '
					float run(ivec4 idx4) {
						ivec4 aIdx4 = replace$axisCompUpper(idx4, 0);
						int idx1Offset = index4To1(aIdx4, $U_SRC_SHAPE1);
						float sum = 0.0;
					' +
						GpuShader.loopOverDimensions("idx1", "idx1Offset", [1], [[axisComp]], '
							elem a = src1(idx1);
							sum += a.value;
						')
					+ '
						return sum;
					}
				')
			];
			this.backwardOps = [
				bop([dst], a, "reducesum_backward_kd", '
					float run(ivec4 idx4) {
						elem dst = src1(replace$axisCompUpper(idx4, 0));
						return dst.diff;
					}
				')
			];
		} else {
			this.forwardOps = [
				fop([a], dst, "reducesum_forward", '
					float run(ivec4 idx4) {
						ivec4 aIdx4 = insert$axisCompUpper(idx4, 0);
						int idx1Offset = index4To1(aIdx4, $U_SRC_SHAPE1);
						float sum = 0.0;
					' +
						GpuShader.loopOverDimensions("idx1", "idx1Offset", [1], [[axisComp]], '
							elem a = src1(idx1);
							sum += a.value;
						')
					+ '
						return sum;
					}
				')
			];
			this.backwardOps = [
				bop([dst], a, "reducesum_backward", '
					float run(ivec4 idx4) {
						elem dst = src1(delete$axisCompUpper(idx4));
						return dst.diff;
					}
				')
			];
		}
	}

	override function shapeCheck():Void {
		var dstShape:Array<Int> = [];

		if (keepDim) {
			dstShape = a.actualShape.toArray();
			dstShape[axis] = 1;
			dst.assignShape(dstShape);
		} else {
			for (i in 0...a.rank) {
				if (axis != i) {
					dstShape.push(a.actualShape[i]);
				}
			}
			dst.assignShape(dstShape);
		}
	}

}
