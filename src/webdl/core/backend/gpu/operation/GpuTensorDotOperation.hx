package webdl.core.backend.gpu.operation;
import haxe.ds.Vector;
import webdl.core.Tensor;
import webdl.core.backend.gpu.ShaderConsts.*;

/**
 * ...
 */
class GpuTensorDotOperation extends GpuOperation {
	var a:Tensor;
	var b:Tensor;
	var dst:Tensor;
	var adummyAxes:Array<Int>;
	var bdummyAxes:Array<Int>;
	var afreeAxes:Array<Int>;
	var bfreeAxes:Array<Int>;

	public function new(backend:GpuBackend, a:Tensor, b:Tensor, dst:Tensor, count:Int, axes:Array<Array<Int>>) {
		super(backend, [a, b], [dst]);
		this.a = a;
		this.b = b;
		this.dst = dst;

		if (count == -1 && axes == null || count != -1 && axes != null) throw "specify either count or axes";

		if (count != -1) {
			adummyAxes = [for (i in a.rank - count...a.rank) i];
			bdummyAxes = [for (i in 0...count) i];
		} else {
			if (axes.length != 2) throw "invalid argument";
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

		var adummy:Array<String> = adummyAxes.map((i) -> "xyzw".charAt(a.rank - 1 - i));
		var bdummy:Array<String> = bdummyAxes.map((i) -> "xyzw".charAt(b.rank - 1 - i));
		var afree:Array<String> = afreeAxes.map((i) -> "xyzw".charAt(a.rank - 1 - i));
		var bfree:Array<String> = bfreeAxes.map((i) -> "xyzw".charAt(b.rank - 1 - i));
		var afreeInDst:Array<String> = afreeInDstAxes.map((i) -> "xyzw".charAt(dst.rank - 1 - i));
		var bfreeInDst:Array<String> = bfreeInDstAxes.map((i) -> "xyzw".charAt(dst.rank - 1 - i));

		var adummyAll:String = adummy.join("");
		var bdummyAll:String = bdummy.join("");
		var afreeAll:String = afree.join("");
		var bfreeAll:String = bfree.join("");
		var afreeInDstAll:String = afreeInDst.join("");
		var bfreeInDstAll:String = bfreeInDst.join("");

		var hasAfree:Bool = afree.length != 0;
		var hasBfree:Bool = bfree.length != 0;
		var hasDummy:Bool = adummy.length != 0;

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
			fop([a, b], dst, "tensordot_forward", '
				float run(ivec4 idx4) {
					ivec4 aIdx4 = ivec4(0);
					ivec4 bIdx4 = ivec4(0);

					// assign free indices
				' +
					(hasAfree ? 'aIdx4.$afreeAll = idx4.$afreeInDstAll;\n' : '') +
					(hasBfree ? 'bIdx4.$bfreeAll = idx4.$bfreeInDstAll;\n' : '')
				+ '

					int aIdx1Offset = index4To1(aIdx4, $U_SRC_SHAPE1);
					int bIdx1Offset = index4To1(bIdx4, $U_SRC_SHAPE2);

					ivec2 idx1Offsets = ivec2(aIdx1Offset, bIdx1Offset);

					float sum = 0.0;
				' +
					GpuShader.loopOverDimensions("idx1s", "idx1Offsets", [1, 2], [adummy, bdummy], '
						elem a = src1(idx1s.x);
						elem b = src2(idx1s.y);
						sum += a.value * b.value;
					')
				+ '
					return sum;
				}
			')
		];

		this.backwardOps = [
			// ∂a_{afree|adummy} <- ∂c_{afreeInDst|bfreeInDst} * b_{bfree|bdummy}
			bop([dst, b], a, "tensordot_backward_a", '
				float run(ivec4 idx4) {
					ivec4 dstIdx4 = ivec4(0);
					ivec4 bIdx4 = ivec4(0);

					// assign free and dummy indices
				' +
					(hasAfree ? 'dstIdx4.$afreeInDstAll = idx4.$afreeAll;\n' : '') +
					(hasDummy ? 'bIdx4.$bdummyAll = idx4.$adummyAll;\n' : '')
				+ '

					int dstIdx1Offset = index4To1(dstIdx4, $U_SRC_SHAPE1);
					int bIdx1Offset = index4To1(bIdx4, $U_SRC_SHAPE2);

					ivec2 idx1Offsets = ivec2(dstIdx1Offset, bIdx1Offset);

					float sum = 0.0;
				' +
					GpuShader.loopOverDimensions("idx1s", "idx1Offsets", [1, 2], [bfreeInDst, bfree], '
						elem dst = src1(idx1s.x);
						elem b = src2(idx1s.y);
						sum += dst.diff * b.value;
					')
				+ '
					return sum;
				}
			'),

			// ∂b_{bfree|bdummy} <- ∂c_{afreeInDst|bfreeInDst} * a_{afree|adummy}
			bop([dst, a], b, "tensordot_backward_b", '
				float run(ivec4 idx4) {
					ivec4 dstIdx4 = ivec4(0);
					ivec4 aIdx4 = ivec4(0);

					// assign free and dummy indices
				' +
					(hasBfree ? 'dstIdx4.$bfreeInDstAll = idx4.$bfreeAll;\n' : '') +
					(hasDummy ? 'aIdx4.$adummyAll = idx4.$bdummyAll;\n' : '')
				+ '

					int dstIdx1Offset = index4To1(dstIdx4, $U_SRC_SHAPE1);
					int aIdx1Offset = index4To1(aIdx4, $U_SRC_SHAPE2);

					ivec2 idx1Offsets = ivec2(dstIdx1Offset, aIdx1Offset);

					float sum = 0.0;
				' +
					GpuShader.loopOverDimensions("idx1s", "idx1Offsets", [1, 2], [afreeInDst, afree], '
						elem dst = src1(idx1s.x);
						elem a = src2(idx1s.y);
						sum += dst.diff * a.value;
					')
				+ '
					return sum;
				}
			')
		];
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
