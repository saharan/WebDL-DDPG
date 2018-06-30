package webdl.core.backend.cuda.operation;
import haxe.ds.Vector;
import webdl.core.Tensor;
import webdl.core.backend.cuda.CudaAtomicOperation;

/**
 * ...
 */
class CudaTensorDotOperation extends CudaOperation {
	var a:Tensor;
	var b:Tensor;
	var dst:Tensor;
	var adummyAxes:Array<Int>;
	var bdummyAxes:Array<Int>;
	var afreeAxes:Array<Int>;
	var bfreeAxes:Array<Int>;

	public function new(a:Tensor, b:Tensor, dst:Tensor, count:Int, axes:Array<Array<Int>>) {
		super([a, b], [dst]);
		this.a = a;
		this.b = b;
		this.dst = dst;

		if (count == -1 && axes == null || count != -1 && axes != null) throw "specify either count or axes";

		if (count != -1) {
			adummyAxes = [for (i in a.rank - count...a.rank) i];
			bdummyAxes = [for (i in 0...count) i];
		} else {
			if (axes.length != 2 || axes[0].length != axes[1].length) throw "invalid argument";
			adummyAxes = axes[0];
			bdummyAxes = axes[1];
		}
		afreeAxes = [for (i in 0...a.rank) i].filter((i) -> adummyAxes.indexOf(i) == -1);
		bfreeAxes = [for (i in 0...b.rank) i].filter((i) -> bdummyAxes.indexOf(i) == -1);
		var afreeInDstAxes:Array<Int> = [for (i in 0...afreeAxes.length) i];
		var bfreeInDstAxes:Array<Int> = [for (i in 0...bfreeAxes.length) afreeAxes.length + i];

		if (adummyAxes.length != bdummyAxes.length) throw "axis sizes mismatch";
		axesCheck(adummyAxes, a.rank);
		axesCheck(bdummyAxes, b.rank);

		var adummy:Array<Int> = adummyAxes.map((i) -> a.rank - 1 - i);
		var bdummy:Array<Int> = bdummyAxes.map((i) -> b.rank - 1 - i);
		var afree:Array<Int> = afreeAxes.map((i) -> a.rank - 1 - i);
		var bfree:Array<Int> = bfreeAxes.map((i) -> b.rank - 1 - i);
		var afreeInDst:Array<Int> = afreeInDstAxes.map((i) -> dst.rank - 1 - i);
		var bfreeInDst:Array<Int> = bfreeInDstAxes.map((i) -> dst.rank - 1 - i);

		var numAfrees:Int = afree.length;
		var numBfrees:Int = bfree.length;
		var numDummies:Int = adummy.length;

		// c := dst
		//
		// c_{afreeInDst|bfreeInDst} = a_{afree|adummy} * b_{bfree|bdummy}
		//
		// ↓ diff
		//
		// ∂c_{afreeInDst|bfreeInDst}/∂a_{afree|adummy} = b_{bfree|bdummy}
		// ∂c_{afreeInDst|bfreeInDst}/∂b_{bfree|bdummy} = a_{afree|adummy}
		//
		// ↓ backprop
		//
		// ∂a_{afree|adummy} <- ∂c_{afreeInDst|bfreeInDst} * b_{bfree|bdummy}
		// ∂b_{bfree|bdummy} <- ∂c_{afreeInDst|bfreeInDst} * a_{afree|adummy}

		this.forwardOps = [
			// c_(aFree,bFree) = a_{aFree,aDummy} b_{bFree,bDummy}
			new CudaAtomicOperation([dst, a, b], "tensordot_forward", '
				def_idx4(aIdx4);
				def_idx4(bIdx4);

				// assign free indices
				' +
					[for (i in 0...numAfrees) '
						aIdx4[${afree[i]}] = idx4[${afreeInDst[i]}];
					'].join("\n") +
					[for (i in 0...numBfrees) '
						bIdx4[${bfree[i]}] = idx4[${bfreeInDst[i]}];
					'].join("\n")
				+ '

				// loop over dummy indices
				float sum = 0;
				' +
					({
						var indices:Array<String> = [for (i in 0...numDummies) 'i$i'];
						var nums:Array<String> = [for (i in 0...numDummies) 'shape1[${adummy[i]}]'];
						var inner:String = [for (i in 0...numDummies) '
							aIdx4[${adummy[i]}] = ${indices[i]};
							bIdx4[${bdummy[i]}] = ${indices[i]};
						'].join("\n");
						inner += '
							float a = val(1, aIdx4);
							float b = val(2, bIdx4);
							sum += a * b;
						';
						loops(indices, nums, inner);
					})
				+ '
				val(0, idx4) = sum;
			')
		];

		this.backwardOps = [
			// ∂a_{afree|adummy} <- ∂c_{afreeInDst|bfreeInDst} * b_{bfree|bdummy}
			new CudaAtomicOperation([a, dst, b], "tensordot_backward_a", '
				def_idx4(dstIdx4);
				def_idx4(bIdx4);

				// assign afree and dummy indices
				' +
					[for (i in 0...numAfrees) '
						dstIdx4[${afreeInDst[i]}] = idx4[${afree[i]}];
					'].join("\n") +
					[for (i in 0...numDummies) '
						bIdx4[${bdummy[i]}] = idx4[${adummy[i]}];
					'].join("\n")
				+ '

				// loop over bfree indices
				float sum = 0;
				' +
					({
						var indices:Array<String> = [for (i in 0...numBfrees) 'i$i'];
						var nums:Array<String> = [for (i in 0...numBfrees) 'shape2[${bfree[i]}]'];
						var inner:String = [for (i in 0...numBfrees) '
							dstIdx4[${bfreeInDst[i]}] = ${indices[i]};
							bIdx4[${bfree[i]}] = ${indices[i]};
						'].join("\n");
						inner += '
							float d = dif(1, dstIdx4);
							float b = val(2, bIdx4);
							sum += d * b;
						';
						loops(indices, nums, inner);
					})
				+ '
				dif(0, idx4) += sum;
			'),

			// ∂b_{bfree|bdummy} <- ∂c_{afreeInDst|bfreeInDst} * a_{afree|adummy}
			new CudaAtomicOperation([b, dst, a], "tensordot_backward_b", '
				def_idx4(dstIdx4);
				def_idx4(aIdx4);

				// assign bfree and dummy indices
				' +
					[for (i in 0...numBfrees) '
						dstIdx4[${bfreeInDst[i]}] = idx4[${bfree[i]}];
					'].join("\n") +
					[for (i in 0...numDummies) '
						aIdx4[${adummy[i]}] = idx4[${bdummy[i]}];
					'].join("\n")
				+ '

				// loop over afree indices
				float sum = 0;
				' +
					({
						var indices:Array<String> = [for (i in 0...numAfrees) 'i$i'];
						var nums:Array<String> = [for (i in 0...numAfrees) 'shape2[${afree[i]}]'];
						var inner:String = [for (i in 0...numAfrees) '
							dstIdx4[${afreeInDst[i]}] = ${indices[i]};
							aIdx4[${afree[i]}] = ${indices[i]};
						'].join("\n");
						inner += '
							float d = dif(1, dstIdx4);
							float a = val(2, aIdx4);
							sum += d * a;
						';
						loops(indices, nums, inner);
					})
				+ '
				dif(0, idx4) += sum;
			')
		];
	}

	function loops(indices:Array<String>, nums:Array<String>, inner:String):String {
		if (indices.length != nums.length) throw "!?";
		var n:Int = indices.length;
		return
			[for (i in 0...n) 'for (int ${indices[i]} = 0; ${indices[i]} < ${nums[i]}; ${indices[i]}++) {'].join("\n") +
			inner +
			indices.map((_) -> "}").join("\n")
		;
	}

	function axesCheck(axes:Array<Int>, rank:Int):Void {
		if (axes.length > rank) throw "too many axes";
		for (i in 0...axes.length) {
			if (axes[i] < 0 || axes[i] >= rank) throw "invalid axis";
			for (j in 0...i) {
				if (axes[i] == axes[j]) throw "duplicate axes";
			}
		}
	}

	override function shapeCheck():Void {
		// check for dummy axes
		if (adummyAxes.length != bdummyAxes.length) throw "!?";
		for (i in 0...adummyAxes.length) {
			var adummySize:Int = a.actualShape[adummyAxes[i]];
			var bdummySize:Int = b.actualShape[bdummyAxes[i]];
			if (adummySize == -1 || bdummySize == -1) throw "no data assigned";
			if (adummySize != bdummySize) throw "cannot multiply tensors";
		}

		// collect free axes
		var dstShape:Array<Int> = [];
		for (i in afreeAxes) {
			dstShape.push(a.actualShape[i]);
		}
		for (i in bfreeAxes) {
			dstShape.push(b.actualShape[i]);
		}

		dst.assignShape(dstShape);
	}
}
