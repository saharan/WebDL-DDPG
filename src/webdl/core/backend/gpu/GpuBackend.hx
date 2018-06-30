package webdl.core.backend.gpu;
import js.Browser;
import js.html.CanvasElement;
import js.html.Float32Array;
import js.html.Int16Array;
import js.html.webgl.Buffer;
import js.html.webgl.GL;
import js.html.webgl.Program;
import webdl.core.Operation;
import webdl.core.Tensor;
import webdl.core.TensorData;
import webdl.core.backend.Backend;
import webdl.core.backend.gpu.operation.GpuAbsOperation;
import webdl.core.backend.gpu.operation.GpuActivationOperation;
import webdl.core.backend.gpu.operation.GpuAdamUpdateOperation;
import webdl.core.backend.gpu.operation.GpuAddConstOperation;
import webdl.core.backend.gpu.operation.GpuAddOperation;
import webdl.core.backend.gpu.operation.GpuAssignOperation;
import webdl.core.backend.gpu.operation.GpuBiasAddOperation;
import webdl.core.backend.gpu.operation.GpuDivOperation;
import webdl.core.backend.gpu.operation.GpuExpOperation;
import webdl.core.backend.gpu.operation.GpuGradientsOperation;
import webdl.core.backend.gpu.operation.GpuLinCombOperation;
import webdl.core.backend.gpu.operation.GpuLogOperation;
import webdl.core.backend.gpu.operation.GpuMatMulOperation;
import webdl.core.backend.gpu.operation.GpuMergeOperation;
import webdl.core.backend.gpu.operation.GpuMulConstOperation;
import webdl.core.backend.gpu.operation.GpuMulOperation;
import webdl.core.backend.gpu.operation.GpuPowConstOperation;
import webdl.core.backend.gpu.operation.GpuPowOperation;
import webdl.core.backend.gpu.operation.GpuReduceMeanOperation;
import webdl.core.backend.gpu.operation.GpuReduceSumOperation;
import webdl.core.backend.gpu.operation.GpuSplitOperation;
import webdl.core.backend.gpu.operation.GpuSubOperation;
import webdl.core.backend.gpu.operation.GpuTensorDotOperation;
import webdl.core.backend.gpu.operation.GpuWhereOperation;
import webdl.core.nn.Activation;

/**
 * ...
 */
class GpuBackend implements Backend {
	public var gl:GL;
	var vbuf:Buffer;
	var ibuf:Buffer;
	var disposedData:Array<GpuTensorData>;

	public function new() {
		var canvas:CanvasElement = Browser.document.createCanvasElement();
		gl = canvas.getContextWebGL();
		gl.getExtension("OES_texture_float");
		gl.getExtension("OES_texture_float_linear");
		gl.getExtension("WEBGL_color_buffer_float");

		vbuf = gl.createBuffer();
		ibuf = gl.createBuffer();

		gl.bindBuffer(GL.ARRAY_BUFFER, vbuf);
		gl.bufferData(GL.ARRAY_BUFFER, new Float32Array([
			-1, -1, 0, 0, // x, y, u, v, ...
			1, -1, 1, 0,
			1, 1, 1, 1,
			-1, 1, 0, 1
		]), GL.STATIC_DRAW);
		gl.bindBuffer(GL.ARRAY_BUFFER, null);

		gl.bindBuffer(GL.ELEMENT_ARRAY_BUFFER, ibuf);
		gl.bufferData(GL.ELEMENT_ARRAY_BUFFER, new Int16Array([
			0, 1, 2,
			0, 2, 3
		]), GL.STATIC_DRAW);
		gl.bindBuffer(GL.ELEMENT_ARRAY_BUFFER, null);

		disposedData = [];
	}

	@:allow(webdl.core.backend.gpu)
	function runAtomicOperation(op:GpuAtomicOperation):Void {
		//trace("running an operation...");

		// preprocess
		op.preDraw();

		var program:Program = op.shader.program;
		gl.useProgram(program);

		// get attribute indices
		var posAttrib:Int = gl.getAttribLocation(program, ShaderConsts.A_POS);
		var uvAttrib:Int = gl.getAttribLocation(program, ShaderConsts.A_UV);

		// enable & setup attributes
		gl.bindBuffer(GL.ARRAY_BUFFER, vbuf);
		gl.enableVertexAttribArray(posAttrib);
		gl.vertexAttribPointer(posAttrib, 2, GL.FLOAT, false, 16, 0);
		gl.enableVertexAttribArray(uvAttrib);
		gl.vertexAttribPointer(uvAttrib, 2, GL.FLOAT, false, 16, 8);
		gl.bindBuffer(GL.ARRAY_BUFFER, null);

		// bind uniforms
		op.bindUniforms();

		// draw
		gl.bindBuffer(GL.ELEMENT_ARRAY_BUFFER, ibuf);
		gl.drawElements(GL.TRIANGLES, 6, GL.UNSIGNED_SHORT, 0);
		gl.bindBuffer(GL.ELEMENT_ARRAY_BUFFER, null);

		// disable attributes
		gl.disableVertexAttribArray(posAttrib);
		gl.disableVertexAttribArray(uvAttrib);

		// postprocess
		op.postDraw();
	}

	public function requestTensorData(size:Int):TensorData {
		for (d in disposedData) {
			if (d.isPreferableSize(size)) {
				disposedData.remove(d);
				//trace("data reused: size = " + d.maxSize);
				return d;
			}
		}
		//trace("data created: size = " + size);
		return new GpuTensorData(gl, size);
	}

	public function disposeTensorData(data:TensorData):Void {
		if (!Std.is(data, GpuTensorData)) throw "backends mismatch";
		disposedData.push(cast data);
	}

	public function add(a:Tensor, b:Tensor, dst:Tensor):Operation {
		return new GpuAddOperation(this, a, b, dst);
	}

	public function addConst(a:Tensor, b:Float, dst:Tensor):Operation {
		return new GpuAddConstOperation(this, a, b, dst);
	}

	public function sub(a:Tensor, b:Tensor, dst:Tensor):Operation {
		return new GpuSubOperation(this, a, b, dst);
	}

	public function linComb(a:Tensor, b:Tensor, dst:Tensor, aScale:Float, bScale:Float):Operation {
		return new GpuLinCombOperation(this, a, b, dst, aScale, bScale);
	}

	public function mul(a:Tensor, b:Tensor, dst:Tensor):Operation {
		return new GpuMulOperation(this, a, b, dst);
	}

	public function mulConst(a:Tensor, b:Float, dst:Tensor):Operation {
		return new GpuMulConstOperation(this, a, b, dst);
	}

	public function div(a:Tensor, b:Tensor, dst:Tensor):Operation {
		return new GpuDivOperation(this, a, b, dst);
	}

	public function pow(a:Tensor, b:Tensor, dst:Tensor):Operation {
		return new GpuPowOperation(this, a, b, dst);
	}

	public function powConst(a:Tensor, b:Float, dst:Tensor):Operation {
		return new GpuPowConstOperation(this, a, b, dst);
	}

	public function matMul(a:Tensor, b:Tensor, dst:Tensor):Operation {
		return new GpuMatMulOperation(this, a, b, dst);
	}

	public function tensorDot(a:Tensor, b:Tensor, dst:Tensor, count:Int, axes:Array<Array<Int>>):Operation {
		return new GpuTensorDotOperation(this, a, b, dst, count, axes);
	}

	public function abs(a:Tensor, dst:Tensor):Operation {
		return new GpuAbsOperation(this, a, dst);
	}

	public function log(a:Tensor, dst:Tensor):Operation {
		return new GpuLogOperation(this, a, dst);
	}

	public function exp(a:Tensor, dst:Tensor):Operation {
		return new GpuExpOperation(this, a, dst);
	}

	public function biasAdd(a:Tensor, b:Tensor, dst:Tensor):Operation {
		return new GpuBiasAddOperation(this, a, b, dst);
	}

	public function activation(a:Tensor, dst:Tensor, activation:Activation):Operation {
		return new GpuActivationOperation(this, a, dst, activation);
	}

	public function reduceSum(a:Tensor, dst:Tensor, axis:Int, keepDim:Bool):Operation {
		return new GpuReduceSumOperation(this, a, dst, axis, keepDim);
	}

	public function reduceMean(a:Tensor, dst:Tensor, axis:Int, keepDim:Bool):Operation {
		return new GpuReduceMeanOperation(this, a, dst, axis, keepDim);
	}

	public function split(a:Tensor, dsts:Array<Tensor>, axis:Int, sizes:Array<Int>):Operation {
		return new GpuSplitOperation(this, a, dsts, axis, sizes);
	}

	public function merge(as:Array<Tensor>, dst:Tensor, axis:Int):Operation {
		return new GpuMergeOperation(this, as, dst, axis);
	}

	public function gradients(y:Tensor, xs:Array<Tensor>, dsts:Array<Tensor>, gradY:Tensor):Operation {
		return new GpuGradientsOperation(this, y, xs, dsts, gradY);
	}

	public function assign(a:Tensor, dummyDst:Tensor, dst:Tensor):Operation {
		return new GpuAssignOperation(this, a, dummyDst, dst);
	}

	public function where(cond:Tensor, a:Tensor, b:Tensor, dst:Tensor):Operation {
		return new GpuWhereOperation(this, cond, a, b, dst);
	}

	public function adamUpdate(count:Tensor, t:Tensor, g:Tensor, m:Tensor, v:Tensor, alpha:Tensor, beta1:Tensor, beta2:Tensor, epsilon:Tensor, l2Decay:Tensor, dummyDst:Tensor):Operation {
		return new GpuAdamUpdateOperation(this, count, t, g, m, v, alpha, beta1, beta2, epsilon, l2Decay, dummyDst);
	}

}
