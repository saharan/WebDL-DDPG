package webdl.core.backend.gpu;
import js.html.Float32Array;
import js.html.webgl.Framebuffer;
import js.html.webgl.GL;
import js.html.webgl.Renderbuffer;
import js.html.webgl.Texture;
import js.html.webgl.UniformLocation;

/**
 * 1-D array, stored on GPU as texture.
 */
class GpuArray {
	public var tex(default, null):Texture;
	public var texSize(default, null):Int;
	public var data(default, null):Float32Array;
	public var dataLength(default, null):Int;
	public var maxSize(default, null):Int;

	public var fbuf(default, null):Framebuffer;
	var dbuf:Renderbuffer;
	var gl:GL;

	public function new(gl:GL, texSize:Int) {
		this.gl = gl;
		this.texSize = texSize;
		this.maxSize = texSize * texSize;

		// create data
		dataLength = texSize * texSize * 4;
		data = new Float32Array(dataLength);
		for (i in 0...dataLength) data[i] = 0;

		// create texture
		tex = gl.createTexture();

		// setup texture
		gl.bindTexture(GL.TEXTURE_2D, tex);
		// args: target, level, internalformat, width, height, border, format, type, pixels
		gl.texImage2D(GL.TEXTURE_2D, 0, GL.RGBA, texSize, texSize, 0, GL.RGBA, GL.FLOAT, data);
		gl.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MAG_FILTER, GL.NEAREST);
		gl.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MIN_FILTER, GL.NEAREST);
		gl.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_WRAP_S, GL.REPEAT);
		gl.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_WRAP_T, GL.REPEAT);
		gl.bindTexture(GL.TEXTURE_2D, null);

		// create depthbuffer
		dbuf = gl.createRenderbuffer();

		// setup depthbuffer
		gl.bindRenderbuffer(GL.RENDERBUFFER, dbuf);
		gl.renderbufferStorage(GL.RENDERBUFFER, GL.DEPTH_COMPONENT16, texSize, texSize);
		gl.bindRenderbuffer(GL.RENDERBUFFER, null);

		// create framebuffer
		fbuf = gl.createFramebuffer();

		// setup framebuffer
		gl.bindFramebuffer(GL.FRAMEBUFFER, fbuf);
		gl.framebufferRenderbuffer(GL.FRAMEBUFFER, GL.DEPTH_ATTACHMENT, GL.RENDERBUFFER, dbuf);
		gl.framebufferTexture2D(GL.FRAMEBUFFER, GL.COLOR_ATTACHMENT0, GL.TEXTURE_2D, tex, 0);
		gl.bindFramebuffer(GL.FRAMEBUFFER, null);

		//trace("created. size = " + texSize + "x" + texSize);
	}

	public function downloadPixels():Void {
		gl.bindFramebuffer(GL.FRAMEBUFFER, fbuf);
		gl.readPixels(0, 0, texSize, texSize, GL.RGBA, GL.FLOAT, data);
		gl.bindFramebuffer(GL.FRAMEBUFFER, null);
	}

	public function uploadPixels():Void {
		gl.bindTexture(GL.TEXTURE_2D, tex);
		// args: target, level, internalformat, width, height, border, format, type, pixels
		gl.texImage2D(GL.TEXTURE_2D, 0, GL.RGBA, texSize, texSize, 0, GL.RGBA, GL.FLOAT, data);
		gl.bindTexture(GL.TEXTURE_2D, null);
	}

	public function setAsDst():Void {
		gl.bindFramebuffer(GL.FRAMEBUFFER, fbuf);
		gl.viewport(0, 0, texSize, texSize);
	}

	public function setAsSrc(unif:UniformLocation, unit:Int):Void {
		gl.activeTexture(GL.TEXTURE0 + unit);
		gl.bindTexture(GL.TEXTURE_2D, tex);
		gl.activeTexture(GL.TEXTURE0);
		gl.uniform1i(unif, unit);
	}

}
