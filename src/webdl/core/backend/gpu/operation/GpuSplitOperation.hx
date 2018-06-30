package webdl.core.backend.gpu.operation;
import haxe.ds.Vector;
import webdl.core.backend.gpu.ShaderConsts.*;

/**
 * ...
 */
class GpuSplitOperation extends GpuOperation {
	var a:Tensor;
	var axis:Int;
	var dsts:Array<Tensor>;
	var sizes:Array<Int>;

	public function new(backend:GpuBackend, a:Tensor, dsts:Array<Tensor>, axis:Int, sizes:Array<Int>) {
		super(backend, [a], dsts);
		this.a = a;
		this.dsts = dsts;
		this.sizes = sizes;

		this.axis = axis;
		if (axis < 0 || axis >= a.rank) throw "invalid axis";
		var axisComp:String = "xyzw".charAt(a.rank - 1 - axis);
		var axisCompUpper:String = axisComp.toUpperCase();

		if (dsts.length != sizes.length) throw "invalid argument";

		var offset:Int = 0;
		var offsets:Array<Int> = [];
		for (size in sizes) {
			offsets.push(offset);
			offset += size;
		}
		offsets.push(offset); // guard

		this.forwardOps = [
			for (i in 0...dsts.length) {
				fop([a], dsts[i], "split_forward", '
					float run(ivec4 idx4) {
						elem a = src1(add$axisCompUpper(idx4, ${offsets[i]}));
						return a.value;
					}
				');
			}
		];
		this.backwardOps = [
			bop(dsts, a, "split_backward", '
				float run(ivec4 idx4) {
					int idx = idx4.$axisComp;
				' +
					[for (i in 0...dsts.length) '
						if (idx < ${offsets[i + 1]}) {
							elem dst = src${i + 1}(add$axisCompUpper(idx4, -${offsets[i]}));
							return dst.diff;
						}
					'].join("\n")
				+ '
					return 0.0;
				}
			')
		];
	}

	override function shapeCheck():Void {
		var totalSize:Int = 0;
		for (i in 0...dsts.length) {
			var dstShape:Vector<Int> = a.actualShape.copy();
			dstShape[axis] = sizes[i];
			dsts[i].assignShape(dstShape);
			totalSize += sizes[i];
		}
		if (totalSize != a.actualShape[axis]) throw "split sizes mismatch";
	}

}
