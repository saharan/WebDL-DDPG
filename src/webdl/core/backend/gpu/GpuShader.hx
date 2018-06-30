package webdl.core.backend.gpu;
import haxe.ds.Vector;
import js.Browser;
import js.html.webgl.GL;
import js.html.webgl.Program;
import js.html.webgl.Shader;
import js.html.webgl.UniformLocation;
import webdl.core.backend.gpu.ShaderConsts.*;
using Lambda;

/**
 * GLSL shader for an atomic GPU operation
 */
class GpuShader {
	public static inline var FORWARD:Int = 0;
	public static inline var BACKWARD_ACCUMULATE:Int = 1;
	public static inline var BACKWARD_OVERWRITE:Int = 2;
	public var program(default, null):Program;

	var gl:GL;
	var vertexSource:String;
	var fragmentSource:String;
	var fragmentSourceHeader:String;
	var fragmentSourceFooter:String;
	var vertexShader:Shader;
	var fragmentShader:Shader;
	var map:Map<String, UniformLocation>;

	public function new(gl:GL, numSourceTextures:Int, mode:Int, fragmentSource:String) {
		this.gl = gl;
		vertexShader = gl.createShader(GL.VERTEX_SHADER);
		fragmentShader = gl.createShader(GL.FRAGMENT_SHADER);
		program = gl.createProgram();

		this.vertexSource = '
			attribute vec2 $A_POS;
			attribute vec2 $A_UV;
			varying vec2 $V_UV;

			void main() {
				gl_Position = vec4($A_POS, 0, 1);
				$V_UV = $A_UV;
			}
		';
		this.fragmentSource = fragmentSource;

		// set fragment shader header
		fragmentSourceHeader = '
			precision highp float;
			varying vec2 $V_UV;

			${sourceTextureUniformDefinition(numSourceTextures)}

			uniform sampler2D $U_DST_TEX;
			uniform int $U_DST_TEX_SIZE;
			uniform float $U_DST_INV_TEX_SIZE;
			uniform int $U_DST_ACTUAL_SIZE;
			uniform ivec4 $U_DST_SHAPE;
			uniform ivec4 $U_DST_STRIDE;

			int rem(int a, int b) {
				int tmp = a - a / b * b; // a % b
				tmp += b;
				return tmp - tmp / b * b; // (a % b + b) % b
			}

			struct elem {
				float value;
				float diff;
			};

			int index4To1(ivec4 idx4, ivec4 shape) {
				int idx1 = idx4.w;
				idx1 = idx1 * shape.z + idx4.z;
				idx1 = idx1 * shape.y + idx4.y;
				idx1 = idx1 * shape.x + idx4.x;
				return idx1;
			}

			ivec4 index1To4(int idx1, ivec4 shape) {
				ivec4 idx4 = ivec4(0);
				idx4.x = rem(idx1, shape.x); idx1 /= shape.x;
				idx4.y = rem(idx1, shape.y); idx1 /= shape.y;
				idx4.z = rem(idx1, shape.z); idx1 /= shape.z;
				idx4.w = rem(idx1, shape.w);
				return idx4;
			}

			int uvToIndex1(vec2 uv, int texSize) {
				ivec2 pixel = ivec2(floor(uv * float(texSize)));
				return pixel.y * texSize + pixel.x;
			}

			vec2 index4ToUv(ivec4 idx4, int texSize, ivec4 shape) {
				int idx1 = index4To1(idx4, shape);
				ivec2 pixel = ivec2(rem(idx1, texSize), idx1 / texSize);
				return (vec2(pixel) + 0.5) / float(texSize);
			}

			ivec4 addX(ivec4 idx4, int x) {
				return idx4 + ivec4(x, 0, 0, 0);
			}

			ivec4 addY(ivec4 idx4, int y) {
				return idx4 + ivec4(0, y, 0, 0);
			}

			ivec4 addZ(ivec4 idx4, int z) {
				return idx4 + ivec4(0, 0, z, 0);
			}

			ivec4 addW(ivec4 idx4, int w) {
				return idx4 + ivec4(0, 0, 0, w);
			}

			ivec4 insertX(ivec4 idx4, int x) {
				return ivec4(x, idx4.xyz);
			}

			ivec4 insertY(ivec4 idx4, int y) {
				return ivec4(idx4.x, y, idx4.yz);
			}

			ivec4 insertZ(ivec4 idx4, int z) {
				return ivec4(idx4.xy, z, idx4.z);
			}

			ivec4 insertW(ivec4 idx4, int w) {
				return ivec4(idx4.xyz, w);
			}

			ivec4 deleteX(ivec4 idx4) {
				return ivec4(idx4.yzw, 0);
			}

			ivec4 deleteY(ivec4 idx4) {
				return ivec4(idx4.xzw, 0);
			}

			ivec4 deleteZ(ivec4 idx4) {
				return ivec4(idx4.xyw, 0);
			}

			ivec4 deleteW(ivec4 idx4) {
				return ivec4(idx4.xyz, 0);
			}

			ivec4 replaceX(ivec4 idx4, int x) {
				return ivec4(x, idx4.yzw);
			}

			ivec4 replaceY(ivec4 idx4, int y) {
				return ivec4(idx4.x, y, idx4.zw);
			}

			ivec4 replaceZ(ivec4 idx4, int z) {
				return ivec4(idx4.xy, z, idx4.w);
			}

			ivec4 replaceW(ivec4 idx4, int w) {
				return ivec4(idx4.xyz, w);
			}

			float safePow(float a, float b) {
				if (a >= 0.0) {
					return pow(a, b);
				} else {
					int intB = int(floor(b + 0.5));
					int sign = 1 - rem(intB, 2) * 2; // 1 for even, -1 for odd
					return pow(abs(a), b) * float(sign);
				}
			}

			float tanh(float x) {
				if (x < 0.0) {
					float epx2 = exp(2.0 * x);
					return (epx2 - 1.0) / (epx2 + 1.0);
				} else {
					float emx2 = exp(-2.0 * x);
					return (1.0 - emx2) / (1.0 + emx2);
				}
			}

			float tanhGrad(float x) {
				float tanhx = tanh(x);
				return 1.0 - tanhx * tanhx;
			}

			float sigmoid(float x) {
				return 1.0 / (1.0 + exp(-x));
			}

			float sigmoidGrad(float x) {
				float y = sigmoid(x);
				return y * (1.0 - y);
			}

			float relu(float x) {
				if (x < 0.0) return 0.0;
				return x;
			}

			float reluGrad(float x) {
				if (x < 0.0) return 0.0;
				return 1.0;
			}

			${sourceSampleDefinition(numSourceTextures)}
		';

		fragmentSourceFooter = '
			void main() {
				int idx1 = uvToIndex1($V_UV, $U_DST_TEX_SIZE);
				if (idx1 >= $U_DST_ACTUAL_SIZE) {
					discard;
					return;
				}

				ivec4 idx4 = index1To4(idx1, $U_DST_SHAPE);
				vec4 dst = texture2D($U_DST_TEX, $V_UV);
				float result = run(idx4);
				gl_FragColor = vec4(${switch (mode) {
					//case _: 'idx1';
					case FORWARD: "result, dst.y, 0, 0";
					case BACKWARD_ACCUMULATE: "dst.x, dst.y + result, 0, 0";
					case BACKWARD_OVERWRITE: "dst.x, result, 0, 0";
					case _: throw "invalid shader mode";
				}});
			}
		';

		compile();
	}

	function sourceTextureUniformDefinition(numSourceTextures:Int):String {
		return [for (i in 1...numSourceTextures + 1) '
			uniform sampler2D ${U_SRC_TEX + i};
			uniform int ${U_SRC_TEX_SIZE + i};
			uniform float ${U_SRC_INV_TEX_SIZE + i};
			uniform ivec4 ${U_SRC_SHAPE + i};
			uniform ivec4 ${U_SRC_STRIDE + i};
		'].join("\n");
	}

	function sourceSampleDefinition(numSourceTextures:Int):String {
		return [for (i in 1...numSourceTextures + 1) '
			elem src$i(ivec4 idx4) {
				idx4 *= ivec4(notEqual(${U_SRC_SHAPE + i}, ivec4(1))); // broadcasting
				vec2 uv = index4ToUv(idx4, ${U_SRC_TEX_SIZE + i}, ${U_SRC_SHAPE + i});
				vec4 tex = texture2D(${U_SRC_TEX + i}, uv);
				return elem(tex.x, tex.y);
			}
			elem src$i(int idx1) {
				int pixY = idx1 / ${U_SRC_TEX_SIZE + i};
				int pixX = idx1 - pixY * ${U_SRC_TEX_SIZE + i};
				vec2 uv = vec2(pixX, pixY) * ${U_SRC_INV_TEX_SIZE + i};
				vec4 tex = texture2D(${U_SRC_TEX + i}, uv);
				return elem(tex.x, tex.y);
			}
		'].join("\n");
	}

	function compile():Void {
		gl.shaderSource(vertexShader, vertexSource);
		gl.compileShader(vertexShader);
		if (!gl.getShaderParameter(vertexShader, GL.COMPILE_STATUS)) {
			Browser.alert(gl.getShaderInfoLog(vertexShader));
			var lines:Array<String> = vertexSource.split("\n");
			trace(gl.getShaderInfoLog(vertexShader) + "\n" + [for (i in 0...lines.length) (i + 1) + " | " + lines[i]].join("\n"));
		}

		var combinedFragmentShaderSource:String = fragmentSourceHeader + fragmentSource + fragmentSourceFooter;
		gl.shaderSource(fragmentShader, combinedFragmentShaderSource);
		gl.compileShader(fragmentShader);
		if (!gl.getShaderParameter(fragmentShader, GL.COMPILE_STATUS)) {
			Browser.alert(gl.getShaderInfoLog(fragmentShader));
			var lines:Array<String> = combinedFragmentShaderSource.split("\n");
			trace(gl.getShaderInfoLog(fragmentShader) + "\n" + [for (i in 0...lines.length) (i + 1) + " | " + lines[i]].join("\n"));
		}

		gl.attachShader(program, vertexShader);
		gl.attachShader(program, fragmentShader);
		gl.linkProgram(program);
		if (!gl.getProgramParameter(program, GL.LINK_STATUS)) {
			trace(gl.getProgramInfoLog(program));
		}

		map = new Map();
	}

	function uniform(name:String):UniformLocation {
		if (map.exists(name)) {
			return map.get(name);
		}
		var unif:UniformLocation = gl.getUniformLocation(program, name);
		map.set(name, unif);
		return unif;
	}

	function attribute(name:String):Int {
		return gl.getAttribLocation(program, name);
	}

	public function setSrc(t:Tensor, index:Int):Void {
		if (!Std.is(t.data, GpuTensorData)) throw "backends mismatch";
		var gpuData:GpuTensorData = cast t.data;
		var src:GpuArray = gpuData.src;
		var shape:Vector<Int> = t.actualShape;
		var unit:Int = index + 1;
		src.setAsSrc(uniform(U_SRC_TEX + unit), unit);

		var texSize:Int = src.texSize;
		gl.uniform1i(uniform(U_SRC_TEX_SIZE + unit), texSize);
		gl.uniform1f(uniform(U_SRC_INV_TEX_SIZE + unit), 1 / texSize);
		switch (t.rank) {
		case 0:
			gl.uniform4i(uniform(U_SRC_SHAPE + unit), 1, 1, 1, 1);
			gl.uniform4i(uniform(U_SRC_STRIDE + unit), 1, 1, 1, 1);
		case 1:
			gl.uniform4i(uniform(U_SRC_SHAPE + unit), shape[0], 1, 1, 1);
			gl.uniform4i(uniform(U_SRC_STRIDE + unit), 1, shape[0], shape[0], shape[0]);
		case 2:
			gl.uniform4i(uniform(U_SRC_SHAPE + unit), shape[1], shape[0], 1, 1);
			gl.uniform4i(uniform(U_SRC_STRIDE + unit), 1, shape[1], shape[1] * shape[0], shape[1] * shape[0]);
		case 3:
			gl.uniform4i(uniform(U_SRC_SHAPE + unit), shape[2], shape[1], shape[0], 1);
			gl.uniform4i(uniform(U_SRC_STRIDE + unit), 1, shape[2], shape[2] * shape[1], shape[2] * shape[1] * shape[0]);
		case 4:
			gl.uniform4i(uniform(U_SRC_SHAPE + unit), shape[3], shape[2], shape[1], shape[0]);
			gl.uniform4i(uniform(U_SRC_STRIDE + unit), 1, shape[3], shape[3] * shape[2], shape[3] * shape[2] * shape[1]);
		case _:
			throw "dimension too high";
		}
	}

	public function setDst(t:Tensor):Void {
		if (!Std.is(t.data, GpuTensorData)) throw "backends mismatch";
		var gpuData:GpuTensorData = cast t.data;
		var src:GpuArray = gpuData.src;
		var dst:GpuArray = gpuData.dst;
		var shape:Vector<Int> = t.actualShape;
		src.setAsSrc(uniform(U_DST_TEX), 0);
		dst.setAsDst();

		var texSize:Int = dst.texSize;
		gl.uniform1i(uniform(U_DST_TEX_SIZE), texSize);
		gl.uniform1f(uniform(U_DST_INV_TEX_SIZE), 1 / texSize);
		gl.uniform1i(uniform(U_DST_ACTUAL_SIZE), t.actualSize);
		switch (t.rank) {
		case 0:
			gl.uniform4i(uniform(U_DST_SHAPE), 1, 1, 1, 1);
			gl.uniform4i(uniform(U_DST_STRIDE), 1, 1, 1, 1);
		case 1:
			gl.uniform4i(uniform(U_DST_SHAPE), shape[0], 1, 1, 1);
			gl.uniform4i(uniform(U_DST_STRIDE), 1, shape[0], shape[0], shape[0]);
		case 2:
			gl.uniform4i(uniform(U_DST_SHAPE), shape[1], shape[0], 1, 1);
			gl.uniform4i(uniform(U_DST_STRIDE), 1, shape[1], shape[1] * shape[0], shape[1] * shape[0]);
		case 3:
			gl.uniform4i(uniform(U_DST_SHAPE), shape[2], shape[1], shape[0], 1);
			gl.uniform4i(uniform(U_DST_STRIDE), 1, shape[2], shape[2] * shape[1], shape[2] * shape[1] * shape[0]);
		case 4:
			gl.uniform4i(uniform(U_DST_SHAPE), shape[3], shape[2], shape[1], shape[0]);
			gl.uniform4i(uniform(U_DST_STRIDE), 1, shape[3], shape[3] * shape[2], shape[3] * shape[2] * shape[1]);
		case _:
			throw "dimension too high";
		}
	}

	public static function loopOverDimensions(idx1s:String, idx1Offsets:String, sourceIndices:Array<Int>, components:Array<Array<String>>, inside:String):String {
		if (sourceIndices.length > 4) throw "too many indices";
		if (sourceIndices.length != components.length) throw "invalid argument";

		var res1:String = "";
		var res2:String = "";
		var idx1sType:String = ["int", "ivec2", "ivec3", "ivec4"][sourceIndices.length - 1];

		var numLoops:Int = components[0].length;
		var numIndices:Int = sourceIndices.length;

		if (numLoops == 0) {
			res1 += '$idx1sType $idx1s = $idx1Offsets;\n';
		}

		for (i in 0...numLoops) {
			var componentsInLoop:Array<String> = [for (j in 0...numIndices) components[j][i]];
			var idx1sInLoop:String = i == numLoops - 1 ? idx1s : idx1s + i + "_";
			var stridesInLoop:String = "strides" + i + "_";
			var loopCounter:String = "i" + i + "_";
			var nInLoop:String = "n" + i + "_";
			res1 += 'int $nInLoop = ${U_SRC_SHAPE + sourceIndices[0]}.${componentsInLoop[0]};\n';
			res1 += '$idx1sType $idx1sInLoop = $idx1Offsets;\n';
			res1 += '$idx1sType $stridesInLoop = $idx1sType(${[for (j in 0...numIndices) U_SRC_STRIDE + sourceIndices[j] + "." + componentsInLoop[j]].join(", ")});\n';
			res1 += 'for (int $loopCounter = 0; $loopCounter < $MAX_DIMENSION_SIZE; $loopCounter++) {\nif ($loopCounter == $nInLoop) break;\n';
			res2 = "\n}" + res2;
			res2 = '\n$idx1sInLoop += $stridesInLoop;' + res2;
			idx1Offsets = idx1sInLoop;
		}
		return res1 + inside + res2;
	}

	public static function loopOverDimensionsBroadcast(idx1:String, idx1Offset:String, broadcastTargetIndex:Int, inside:String):String {
		var doBroadcast:String = "doBroadcast";
		var res1:String = 'bvec4 $doBroadcast = bvec4(ivec4(equal($U_DST_SHAPE, ivec4(1))) * ivec4(notEqual(${U_SRC_SHAPE + broadcastTargetIndex}, ivec4(1))));\n';
		var res2:String = "";
		for (i in 0...4) {
			var componentInLoop:String = "xyzw".charAt(i);
			var idx1InLoop:String = i == 3 ? idx1 : idx1 + i + "_";
			var strideInLoop:String = "stride" + i + "_";
			var loopCounter:String = "i" + i + "_";
			var nInLoop:String = "n" + i + "_";
			res1 += 'int $nInLoop = $doBroadcast.$componentInLoop ? ${U_SRC_SHAPE + broadcastTargetIndex}.$componentInLoop : 1;\n';
			res1 += 'int $idx1InLoop = $idx1Offset;\n';
			res1 += 'int $strideInLoop = ${U_SRC_STRIDE + broadcastTargetIndex + "." + componentInLoop};\n';
			res1 += 'for (int $loopCounter = 0; $loopCounter < $MAX_DIMENSION_SIZE; $loopCounter++) {\nif ($loopCounter == $nInLoop) break;\n';
			res2 = "\n}" + res2;
			res2 = '\n$idx1InLoop += $strideInLoop;' + res2;
			idx1Offset = idx1InLoop;
		}
		return res1 + inside + res2;
	}

}
