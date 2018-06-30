package webdl.core.backend.gpu;
import js.html.webgl.GL;
import webdl.core.TensorData;

class GpuTensorData implements TensorData {
	public var maxSize(default, null):Int;
	public var src(default, null):GpuArray;
	public var dst(default, null):GpuArray;
	var gl:GL;

	public function new(gl:GL, requestedSize:Int) {
		this.gl = gl;
		var texSize:Int = 1;
		while (texSize * texSize < requestedSize) {
			texSize *= 2;
		}
		maxSize = texSize * texSize;
		src = new GpuArray(gl, texSize);
		dst = new GpuArray(gl, texSize);
	}

	public function isPreferableSize(size:Int):Bool {
		return size <= maxSize && size * 4 > maxSize;
	}

	public function getValue(size:Int):Array<Float> {
		if (size > maxSize) throw "max size exceeded";
		src.downloadPixels();
		var value:Array<Float> = [];
		for (i in 0...size) {
			value.push(src.data[i << 2]);
		}
		return value;
	}

	public function setValue(value:Array<Float>):Void {
		if (value.length > maxSize) throw "max size exceeded";
		for (i in 0...value.length) {
			src.data[i << 2] = value[i];
		}
		src.uploadPixels();
	}

	public function clearValue(value:Float):Void {
		gl.bindFramebuffer(GL.FRAMEBUFFER, src.fbuf);
		gl.colorMask(true, false, false, false);
		gl.clearColor(value, 0, 0, 0);
		gl.clear(GL.COLOR_BUFFER_BIT);
		gl.colorMask(true, true, true, true);
		gl.bindFramebuffer(GL.FRAMEBUFFER, null);
	}

	public function getDiff(size:Int):Array<Float> {
		if (size > maxSize) throw "max size exceeded";
		src.downloadPixels();
		var diff:Array<Float> = [];
		for (i in 0...size) {
			diff.push(src.data[i << 2 | 1]);
		}
		return diff;
	}

	public function setDiff(diff:Array<Float>):Void {
		if (diff.length > maxSize) throw "max size exceeded";
		for (i in 0...diff.length) {
			src.data[i << 2 | 1] = diff[i];
		}
		src.uploadPixels();
	}

	public function clearDiff(diff:Float):Void {
		gl.bindFramebuffer(GL.FRAMEBUFFER, src.fbuf);
		gl.colorMask(false, true, false, false);
		gl.clearColor(0, diff, 0, 0);
		gl.clear(GL.COLOR_BUFFER_BIT);
		gl.colorMask(true, true, true, true);
		gl.bindFramebuffer(GL.FRAMEBUFFER, null);
	}

	public function flip():Void {
		var tmp = src;
		src = dst;
		dst = tmp;
	}

}
