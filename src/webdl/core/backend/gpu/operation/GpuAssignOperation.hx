package webdl.core.backend.gpu.operation;
import js.html.webgl.GL;
import webdl.core.Tensor;
import webdl.core.backend.gpu.GpuAtomicOperation;
import webdl.core.backend.gpu.GpuTensorData;
import webdl.core.backend.gpu.ShaderConsts.*;

/**
 * ...
 */
class GpuAssignOperation extends GpuOperation {
	var a:Tensor;
	var dummyDst:Tensor;
	var dst:Tensor;
	var assignOp:GpuAtomicOperation;

	public function new(backend:GpuBackend, a:Tensor, dummyDst:Tensor, dst:Tensor) {
		super(backend, [a], [dummyDst]);
		this.a = a;
		this.dummyDst = dummyDst;
		this.dst = dst;
		this.forwardOps = [
			fop([a], dummyDst, "assign_forward_dummy", '
				float run(ivec4 idx4) {
					elem a = src1(idx4);
					return a.value;
				}
			')
		];
		this.backwardOps = [
			bop([dummyDst], a, "assign_backward", '
				float run(ivec4 idx4) {
					elem dst = src1(idx4);
					return dst.diff;
				}
			')
		];
		assignOp = fop([a], dst, "assign_forward", '
			float run(ivec4 idx4) {
				elem a = src1(idx4);
				return a.value;
			}
		');
	}

	override public function run():Void {
		shapeCheck();
		if (!assignFast(a, dummyDst)) {
			//trace("assign forward");
			for (forwardOp in forwardOps) {
				backend.runAtomicOperation(forwardOp);
			}
		}
	}

	override function backwardRun():Void {
		if (!assignFast(dummyDst, a)) {
			//trace("assign backward");
			super.backwardRun();
		}
	}

	override function shapeCheck():Void {
		dst.assignShape(a.actualShape);
		dummyDst.assignShape(a.actualShape);
	}

	override function onAfterRun():Void {
		if (!assignFast(a, dst)) {
			//trace("assign");
			backend.runAtomicOperation(assignOp);
		}
	}

	function assignFast(src:Tensor, dst:Tensor):Bool {
		var srcData:GpuTensorData = cast src.data;
		var dstData:GpuTensorData = cast dst.data;
		if (srcData.src.texSize != dstData.src.texSize) {
			return false;
		}
		var size:Int = srcData.src.texSize;
		var gl:GL = backend.gl;
		gl.bindFramebuffer(GL.FRAMEBUFFER, srcData.src.fbuf);
		gl.bindTexture(GL.TEXTURE_2D, dstData.src.tex);
		gl.copyTexSubImage2D(GL.TEXTURE_2D, 0, 0, 0, 0, 0, size, size);
		gl.bindTexture(GL.TEXTURE_2D, null);
		gl.bindFramebuffer(GL.FRAMEBUFFER, null);
		return true;
	}

}
