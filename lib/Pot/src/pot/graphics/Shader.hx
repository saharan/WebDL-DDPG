package pot.graphics;
import js.Browser;
import js.Error;
import js.html.Float32Array;
import js.html.webgl.GL;
import js.html.webgl.Program;
import js.html.webgl.UniformLocation;

/**
 * Shader class
 */
@:allow(pot.graphics)
class Shader {
	public var pg:Program;
	public var vs:js.html.webgl.Shader;
	public var fs:js.html.webgl.Shader;

	var vSource:String;
	var fSource:String;
	var compiled:Bool;
	var gl:GL;
	var map:Map<String, UniformLocation>;
	var uniforms:Map<String, Uniform>;

	var mat2:Float32Array;
	var mat3:Float32Array;
	var mat4:Float32Array;

	function new(gl:GL) {
		this.gl = gl;
		pg = gl.createProgram();
		vs = gl.createShader(GL.VERTEX_SHADER);
		fs = gl.createShader(GL.FRAGMENT_SHADER);
		map = null;
		vSource = null;
		fSource = null;
		compiled = false;
		uniforms = new Map();
	}

	public function setInt(name:String, x:Int, ?y:Int, ?z:Int, ?w:Int):Void {
		var uniform:Uniform = uniforms.get(name);
		if (uniform == null) {
			uniform = new Uniform();
		}

		uniform.type = Uniform.INT;
		uniform.i1 = x;
		if (y != null) {
			uniform.type = Uniform.INT2;
			uniform.i2 = y;
		}
		if (z != null) {
			uniform.type = Uniform.INT3;
			uniform.i3 = z;
		}
		if (w != null) {
			uniform.type = Uniform.INT4;
			uniform.i4 = w;
		}
		uniforms.set(name, uniform);
	}

	public function setFloat(name:String, x:Float, ?y:Float, ?z:Float, ?w:Float):Void {
		var uniform:Uniform = uniforms.get(name);
		if (uniform == null) {
			uniform = new Uniform();
		}

		uniform.type = Uniform.FLOAT;
		uniform.f1 = x;
		if (y != null) {
			uniform.type = Uniform.FLOAT2;
			uniform.f2 = y;
		}
		if (z != null) {
			uniform.type = Uniform.FLOAT3;
			uniform.f3 = z;
		}
		if (w != null) {
			uniform.type = Uniform.FLOAT4;
			uniform.f4 = w;
		}
		uniforms.set(name, uniform);
	}

	public function setMatrix(name:String, mat:Array<Float>):Void {
		var uniform:Uniform = uniforms.get(name);
		if (uniform == null) {
			uniform = new Uniform();
		}

		switch (mat.length) {
		case 4:
			uniform.type = Uniform.MATRIX2;
			if (uniform.m2 == null) {
				uniform.m2 = new Float32Array(4);
			}
			for (i in 0...4) {
				uniform.m2[i] = mat[i];
			}
		case 9:
			uniform.type = Uniform.MATRIX3;
			if (uniform.m3 == null) {
				uniform.m3 = new Float32Array(9);
			}
			for (i in 0...9) {
				uniform.m3[i] = mat[i];
			}
		case 16:
			uniform.type = Uniform.MATRIX4;
			if (uniform.m4 == null) {
				uniform.m4 = new Float32Array(16);
			}
			for (i in 0...16) {
				uniform.m4[i] = mat[i];
			}
		case _:
			throw new Error("invalid matrix array length: " + mat.length);
		}
		uniforms.set(name, uniform);
	}

	public function setTexture(name:String, tex:Texture):Void {
		var uniform:Uniform = uniforms.get(name);
		if (uniform == null) {
			uniform = new Uniform();
		}

		uniform.type = Uniform.TEX2D;
		uniform.tex = tex;
		uniforms.set(name, uniform);
	}

	public inline function hasUniform(name:String):Bool {
		return getUniformLocation(name) != null;
	}

	@:extern
	inline function vertexShader(source:String):Void {
		vSource = source;
		compiled = false;
	}

	@:extern
	inline function fragmentShader(source:String):Void {
		fSource = source;
		compiled = false;
	}

	@:extern
	inline function compile():Void {
		map = new Map();

		gl.shaderSource(vs, vSource);
		gl.compileShader(vs);
		if (!gl.getShaderParameter(vs, GL.COMPILE_STATUS)) {
			Browser.alert(gl.getShaderInfoLog(vs));
		}

		gl.shaderSource(fs, fSource);
		gl.compileShader(fs);
		if (!gl.getShaderParameter(fs, GL.COMPILE_STATUS)) {
			Browser.alert(gl.getShaderInfoLog(fs));
		}

		gl.attachShader(pg, vs);
		gl.attachShader(pg, fs);
		gl.linkProgram(pg);
		if (gl.getProgramParameter(pg, GL.LINK_STATUS)) {
			compiled = true;
		} else {
			trace(gl.getProgramInfoLog(pg));
		}
	}

	@:extern
	inline function bind():Void {
		if (!compiled) throw new Error("shader is not compiled");
		gl.useProgram(pg);
		bindUniforms();
	}

	@:extern
	inline function bindUniforms():Void {
		var texUnit:Int = 0;
		for (name in uniforms.keys()) {
			var uniform:Uniform = uniforms.get(name);
			if (uniform.type == Uniform.TEX2D) {
				uniform.texUnit = texUnit;
				texUnit++;
			}
		}
		for (name in uniforms.keys()) {
			var uniform:Uniform = uniforms.get(name);
			var loc = getUniformLocation(name);
			switch (uniform.type) {
			case Uniform.INT:
				gl.uniform1i(loc, uniform.i1);
			case Uniform.INT2:
				gl.uniform2i(loc, uniform.i1, uniform.i2);
			case Uniform.INT3:
				gl.uniform3i(loc, uniform.i1, uniform.i2, uniform.i3);
			case Uniform.INT4:
				gl.uniform4i(loc, uniform.i1, uniform.i2, uniform.i3, uniform.i4);
			case Uniform.FLOAT:
				gl.uniform1f(loc, uniform.f1);
			case Uniform.FLOAT2:
				gl.uniform2f(loc, uniform.f1, uniform.f2);
			case Uniform.FLOAT3:
				gl.uniform3f(loc, uniform.f1, uniform.f2, uniform.f3);
			case Uniform.FLOAT4:
				gl.uniform4f(loc, uniform.f1, uniform.f2, uniform.f3, uniform.f4);
			case Uniform.MATRIX2:
				gl.uniformMatrix2fv(loc, false, uniform.m2);
			case Uniform.MATRIX3:
				gl.uniformMatrix3fv(loc, false, uniform.m3);
			case Uniform.MATRIX4:
				gl.uniformMatrix4fv(loc, false, uniform.m4);
			case Uniform.TEX2D:
				gl.activeTexture(GL.TEXTURE0 + uniform.texUnit);
				gl.bindTexture(GL.TEXTURE_2D, uniform.tex == null ? null : uniform.tex.texture);
				gl.uniform1i(loc, uniform.texUnit);
			}
		}
		gl.activeTexture(GL.TEXTURE0);
	}

	@:extern
	inline function getUniformLocation(name:String):UniformLocation {
		if (map.exists(name)) {
			return map.get(name);
		}
		var ul:UniformLocation = gl.getUniformLocation(pg, name);
		map.set(name, ul);
		return ul;
	}
}

@:allow(pot.graphics.Shader)
private class Uniform {
	static inline var INT:Int = 0;
	static inline var INT2:Int = 1;
	static inline var INT3:Int = 2;
	static inline var INT4:Int = 3;
	static inline var FLOAT:Int = 4;
	static inline var FLOAT2:Int = 5;
	static inline var FLOAT3:Int = 6;
	static inline var FLOAT4:Int = 7;
	static inline var MATRIX2:Int = 8;
	static inline var MATRIX3:Int = 9;
	static inline var MATRIX4:Int = 10;
	static inline var TEX2D:Int = 11;

	var type:Int;
	var i1:Int;
	var i2:Int;
	var i3:Int;
	var i4:Int;
	var f1:Float;
	var f2:Float;
	var f3:Float;
	var f4:Float;
	var m2:Float32Array;
	var m3:Float32Array;
	var m4:Float32Array;
	var tex:Texture;
	var texUnit:Int;

	function new() {
	}
}
