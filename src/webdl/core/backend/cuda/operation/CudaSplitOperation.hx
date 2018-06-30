package webdl.core.backend.cuda.operation;
import haxe.ds.Vector;
import webdl.core.backend.cuda.CudaAtomicOperation;

/**
 * ...
 */
class CudaSplitOperation extends CudaOperation {
	var a:Tensor;
	var axis:Int;
	var dsts:Array<Tensor>;
	var sizes:Array<Int>;

	public function new(a:Tensor, dsts:Array<Tensor>, axis:Int, sizes:Array<Int>) {
		super([a], dsts);
		this.a = a;
		this.dsts = dsts;
		this.sizes = sizes;

		this.axis = axis;
		if (axis < 0 || axis >= a.rank) throw "invalid axis";
		var axisIndex:Int = a.rank - 1 - axis;

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
				new CudaAtomicOperation([dsts[i], a], 'split_forward_$i', '
					def_idx4(aIdx4);
					aIdx4[$axisIndex] += ${offsets[i]};
					float a = val(1, aIdx4);
					val(0, idx4) = a;
				');
			}
		];
		this.backwardOps = [
			new CudaAtomicOperation([a].concat(dsts), "split_backward", '
				def_idx4(dstIdx4);
				int idx = dstIdx4[$axisIndex];
				' + [
					for (i in 0...dsts.length) '
						if (idx < ${offsets[i + 1]}) {
							dstIdx4[$axisIndex] -= ${offsets[i]};
							float d = dif(${i + 1}, dstIdx4);
							dif(0, idx4) += d;
							return;
						}
					'
				].join("\n")
			)
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
