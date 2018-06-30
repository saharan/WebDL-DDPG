package webdl.core.backend.gpu.operation;
import haxe.ds.Vector;
import webdl.core.backend.gpu.ShaderConsts.*;

/**
 * ...
 */
class GpuMergeOperation extends GpuOperation {
	var as:Array<Tensor>;
	var axis:Int;
	var dst:Tensor;

	public function new(backend:GpuBackend, as:Array<Tensor>, dst:Tensor, axis:Int) {
		super(backend, as, [dst]);
		this.as = as;
		this.dst = dst;

		this.axis = axis;
		if (axis < 0 || axis >= dst.rank) throw "invalid axis";
		var axisComp:String = "xyzw".charAt(dst.rank - 1 - axis);
		var axisCompUpper:String = axisComp.toUpperCase();

		var offset:String = "0";
		var offsets:Array<String> = [];
		for (i in 0...as.length) {
			offsets.push(offset);
			offset += " + " + U_SRC_SHAPE + (i + 1) + "." + axisComp;
		}
		offsets.push(offset); // guard
		offsets = offsets.map((s) -> '($s)');

		this.forwardOps = [
			fop(as, dst, "merge_forward", '
				float run(ivec4 idx4) {
					int idx = idx4.$axisComp;
				' +
					[for (i in 0...as.length) '
						if (idx < ${offsets[i + 1]}) {
							elem a = src${i + 1}(add$axisCompUpper(idx4, -${offsets[i]}));
							return a.value;
						}
					'].join("\n")
				+ '
					return 0.0;
				}
			')
		];
		this.backwardOps = [
			for (i in 0...as.length) {
				bop(as.concat([dst]), as[i], "merge_backward_" + i, '
					float run(ivec4 idx4) {
						elem dst = src${as.length + 1}(add$axisCompUpper(idx4, ${offsets[i]}));
						return dst.diff;
					}
				');
			}
		];
	}

	override function shapeCheck():Void {
		var totalSize:Int = 0;
		var dstShape:Vector<Int> = as[0].actualShape.copy();
		dstShape[axis] = 0;
		for (i in 0...as.length) {
			var aShape:Vector<Int> = as[i].actualShape;
			totalSize += aShape[axis];

			// check shape size except for the axis dimension
			dstShape[axis] = aShape[axis];
			shapeEq(dstShape, aShape);
		}
		dstShape[axis] = totalSize;
		dst.assignShape(dstShape);
	}

}
