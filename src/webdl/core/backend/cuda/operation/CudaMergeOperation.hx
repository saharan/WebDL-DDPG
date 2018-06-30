package webdl.core.backend.cuda.operation;
import haxe.ds.Vector;
import webdl.core.backend.cuda.CudaAtomicOperation;

/**
 * ...
 */
class CudaMergeOperation extends CudaOperation {
	var as:Array<Tensor>;
	var axis:Int;
	var dst:Tensor;

	public function new(as:Array<Tensor>, dst:Tensor, axis:Int) {
		super(as, [dst]);
		this.as = as;
		this.dst = dst;

		this.axis = axis;
		if (axis < 0 || axis >= dst.rank) throw "invalid axis";
		var axisIndex:Int = dst.rank - 1 - axis;

		var offset:String = "0";
		var offsets:Array<String> = [];
		for (i in 0...as.length) {
			offsets.push('($offset)');
			offset += ' + shape${i + 1}[$axisIndex]';
		}
		offsets.push('($offset)'); // guard

		this.forwardOps = [
			new CudaAtomicOperation([dst].concat(as), "merge_forward", '
				def_idx4(aIdx4);
				int idx = idx4[$axisIndex];
				' + [
					for (i in 0...as.length) '
						if (idx < ${offsets[i + 1]}) {
							aIdx4[$axisIndex] -= ${offsets[i]};
							float a = val(${i + 1}, aIdx4);
							val(0, idx4) = a;
							return;
						}
					'
				].join("\n")
			)
		];
		this.backwardOps = [
			for (i in 0...as.length) {
				new CudaAtomicOperation([as[i]].concat(as).concat([dst]), "merge_backward_" + i, '
					def_idx4(dstIdx4);
					dstIdx4[$axisIndex] += ${offsets[i]};
					float d = dif(${as.length + 1}, dstIdx4);
					dif(0, idx4) += d;
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
