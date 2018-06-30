package pot.graphics;
import haxe.ds.Vector;
import js.Browser;
import js.html.CanvasRenderingContext2D;
import js.html.Float32Array;
import js.html.Image;
import js.html.Uint8Array;
import js.html.webgl.Framebuffer;
import js.html.webgl.GL;
import js.html.webgl.Renderbuffer;

/**
 * Bitmap image class
 */
@:allow(pot.graphics)
class Texture {
	// (width, height) is the texture's original size before resizing
	public var width(default, null):Int;
	public var height(default, null):Int;
	// (textureWidth, textureHeight) is the texture's actual size after resizing
	public var textureWidth(default, null):Int;
	public var textureHeight(default, null):Int;

	var imageToU:Float;
	var imageToV:Float;
	var normalToU:Float;
	var normalToV:Float;

	var gl:GL;
	var texture:js.html.webgl.Texture;

	var frameBuffer:Framebuffer;
	var depthBuffer:Renderbuffer;

	var texWrap:TextureWrap;
	var texFilter:TextureFilter;

	var floating:Bool;
	var pixelBufferFloat:Float32Array;
	var pixelBufferByte:Uint8Array;

	function new(gl:GL) {
		this.gl = gl;
		texture = null;
		texWrap = Clamp;
		texFilter = Linear;

		texture = gl.createTexture();
		frameBuffer = gl.createFramebuffer();
		depthBuffer = gl.createRenderbuffer();
	}

	function init(width:Int, height:Int, floating:Bool):Void {
		this.width = width;
		this.height = height;
		this.floating = floating;

		textureWidth = pow2(width);
		textureHeight = pow2(height);
		imageToU = 1 / textureWidth;
		imageToV = 1 / textureHeight;
		normalToU = width / textureWidth;
		normalToV = height / textureHeight;

		gl.bindTexture(GL.TEXTURE_2D, texture);
		gl.texImage2D(GL.TEXTURE_2D, 0, GL.RGBA, textureWidth, textureHeight, 0, GL.RGBA, floating ? GL.FLOAT : GL.UNSIGNED_BYTE, null);
		gl.bindTexture(GL.TEXTURE_2D, null);

		if (floating) {
			pixelBufferFloat = new Float32Array(textureWidth * textureHeight * 4);
		} else {
			pixelBufferByte = new Uint8Array(textureWidth * textureHeight * 4);
		}

		filter(texFilter);
		wrap(texWrap);

		initFBO();
	}

	function load(image:Image, scalingMode:ScalingMode, floating:Bool):Void {
		this.floating = floating;

		var imageWidth:Int = image.width;
		var imageHeight:Int = image.height;

		textureWidth = pow2(imageWidth);
		textureHeight = pow2(imageHeight);

		// expand the image
		var canvas = Browser.document.createCanvasElement();
		canvas.width = textureWidth;
		canvas.height = textureHeight;
		var g:CanvasRenderingContext2D = canvas.getContext2d();
		g.transform(1, 0, 0, -1, 0, textureHeight); // invert y-axis
		switch (scalingMode) {
		case Scale:
			g.drawImage(image, 0, 0, imageWidth, imageHeight, 0, 0, textureWidth, textureHeight);

			// set texture's size as using size
			width = textureWidth;
			height = textureHeight;

			// compute uv matrix
			imageToU = 1 / imageWidth;
			imageToV = 1 / imageHeight;
			normalToU = 1;
			normalToV = 1;

		case Original:
			g.drawImage(image, 0, 0, imageWidth, imageHeight, 0, 0, imageWidth, imageHeight);
			g.drawImage(image, imageWidth - 0.5, 0, 0.5, imageHeight, imageWidth, 0, textureWidth - imageWidth, imageHeight);
			g.drawImage(image, 0, imageHeight - 0.5, imageWidth, 0.5, 0, imageHeight, imageWidth, textureHeight - imageHeight);
			g.drawImage(image, imageWidth - 0.5, imageHeight - 0.5, 0.5, 0.5, imageWidth, imageHeight, textureWidth - imageWidth, textureHeight - imageHeight);

			// set original image's size as using size
			width = textureWidth;
			height = textureHeight;

			// compute uv matrix
			imageToU = 1 / textureWidth;
			imageToV = 1 / textureHeight;
			normalToU = imageWidth / textureWidth;
			normalToV = imageHeight / textureHeight;
		}

		gl.bindTexture(GL.TEXTURE_2D, texture);
		gl.texImage2D(GL.TEXTURE_2D, 0, GL.RGBA, GL.RGBA, floating ? GL.FLOAT : GL.UNSIGNED_BYTE, canvas);
		gl.bindTexture(GL.TEXTURE_2D, null);

		if (floating) {
			pixelBufferFloat = new Float32Array(textureWidth * textureHeight * 4);
		} else {
			pixelBufferByte = new Uint8Array(textureWidth * textureHeight * 4);
		}

		filter(texFilter);
		wrap(texWrap);

		initFBO();
	}

	public function uploadPixelsFloat(xOffset:Int, yOffset:Int, width:Int, height:Int, pixelsRGBA:Vector<Float>):Void {
		if (texture == null) throw "not initialized";
		if (!floating) throw "not floating point texture";
		if (pixelsRGBA.length != width * height * 4) throw "dimensions mismatch";

		// copy to ArrayBufferView, mirroring vertical order
		var srcIdx:Int = 0;
		var dstIdx:Int = width * (height - 1) * 4;
		for (i in 0...height) {
			for (j in 0...width) {
				pixelBufferFloat[dstIdx++] = pixelsRGBA[srcIdx++];
				pixelBufferFloat[dstIdx++] = pixelsRGBA[srcIdx++];
				pixelBufferFloat[dstIdx++] = pixelsRGBA[srcIdx++];
				pixelBufferFloat[dstIdx++] = pixelsRGBA[srcIdx++];
			}
			dstIdx -= width << 3;
		}

		gl.bindTexture(GL.TEXTURE_2D, texture);
		gl.texSubImage2D(GL.TEXTURE_2D, 0, xOffset, textureHeight - height - yOffset, width, height, GL.RGBA, GL.FLOAT, pixelBufferFloat);
		gl.bindTexture(GL.TEXTURE_2D, null);
	}

	public function uploadPixelsInt(xOffset:Int, yOffset:Int, width:Int, height:Int, pixelsInt32:Vector<Int>):Void {
		if (texture == null) throw "not initialized";
		if (floating) throw "not int texture";
		if (pixelsInt32.length != width * height) throw "dimensions mismatch";

		// copy to ArrayBufferView, mirroring vertical order
		var srcIdx:Int = 0;
		var dstIdx:Int = width * (height - 1) * 4;
		for (i in 0...height) {
			for (j in 0...width) {
				var pix:Int = pixelsInt32[srcIdx++];
				var pixA:Int = pix >>> 24;
				var pixR:Int = pix >>> 16 & 0xff;
				var pixG:Int = pix >>> 8 & 0xff;
				var pixB:Int = pix & 0xff;
				pixelBufferByte[dstIdx++] = pixR;
				pixelBufferByte[dstIdx++] = pixG;
				pixelBufferByte[dstIdx++] = pixB;
				pixelBufferByte[dstIdx++] = pixA;
			}
			dstIdx -= width << 3;
		}

		gl.bindTexture(GL.TEXTURE_2D, texture);
		gl.texSubImage2D(GL.TEXTURE_2D, 0, xOffset, textureHeight - height - yOffset, width, height, GL.RGBA, GL.UNSIGNED_BYTE, pixelBufferByte);
		gl.bindTexture(GL.TEXTURE_2D, null);
	}

	function initFBO():Void {
		// init depth buffer
		gl.bindRenderbuffer(GL.RENDERBUFFER, depthBuffer);
		gl.renderbufferStorage(GL.RENDERBUFFER, GL.DEPTH_COMPONENT16, textureWidth, textureHeight);
		gl.bindRenderbuffer(GL.RENDERBUFFER, null);

		// init frame buffer
		gl.bindFramebuffer(GL.FRAMEBUFFER, frameBuffer);
		// bind depth buffer
		gl.framebufferRenderbuffer(GL.FRAMEBUFFER, GL.DEPTH_ATTACHMENT, GL.RENDERBUFFER, depthBuffer);
		// bind texture
		gl.framebufferTexture2D(GL.FRAMEBUFFER, GL.COLOR_ATTACHMENT0, GL.TEXTURE_2D, texture, 0);
		gl.bindFramebuffer(GL.FRAMEBUFFER, null);
	}

	public inline function filter(filter:TextureFilter):Void {
		texFilter = filter;
		gl.bindTexture(GL.TEXTURE_2D, texture);
		switch (filter) {
		case Nearest:
			gl.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MAG_FILTER, GL.NEAREST);
			gl.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MIN_FILTER, GL.NEAREST);
		case Linear:
			gl.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MAG_FILTER, GL.LINEAR);
			gl.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MIN_FILTER, GL.LINEAR);
		}
		gl.bindTexture(GL.TEXTURE_2D, null);
	}

	public inline function wrap(wrap:TextureWrap):Void {
		texWrap = wrap;
		gl.bindTexture(GL.TEXTURE_2D, texture);
		gl.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_WRAP_S, wrap);
		gl.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_WRAP_T, wrap);
		gl.bindTexture(GL.TEXTURE_2D, null);
	}

	@:extern
	inline function pow2(x:Int):Int {
		x -= 1;
		x = x | x >> 1;
		x = x | x >> 2;
		x = x | x >> 4;
		x = x | x >> 8;
		x = x | x >> 16;
		return x + 1;
	}
}
