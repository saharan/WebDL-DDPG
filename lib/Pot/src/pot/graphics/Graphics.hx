package pot.graphics;
import haxe.ds.Vector;
import js.Error;
import js.html.CanvasElement;
import js.html.Float32Array;
import js.html.Image;
import js.html.Int16Array;
import js.html.webgl.Buffer;
import js.html.webgl.GL;
import js.html.webgl.Program;
import pot.graphics.ShapeMode;

/**
 * ...
 */
class Graphics {
	var canvas:CanvasElement;
	var gl:GL;

	var positionBuf:VertexBuffer;
	var colorBuf:VertexBuffer;
	var normalBuf:VertexBuffer;
	var texCoordBuf:VertexBuffer;
	var indexBuf:IndexBuffer;

	var shapeMode:Int;
	var numVertices:Int;
	var normalX:Float;
	var normalY:Float;
	var normalZ:Float;
	var texCoordU:Float;
	var texCoordV:Float;
	var colorR:Float;
	var colorG:Float;
	var colorB:Float;
	var colorA:Float;

	var cameraFov:Float;
	var cameraNear:Float;
	var cameraFar:Float;
	var defaultCameraPosX:Float;
	var defaultCameraPosY:Float;
	var defaultCameraPosZ:Float;
	var cameraSet:Bool;
	var cameraPosX:Float;
	var cameraPosY:Float;
	var cameraPosZ:Float;
	var cameraAtX:Float;
	var cameraAtY:Float;
	var cameraAtZ:Float;
	var cameraUpX:Float;
	var cameraUpY:Float;
	var cameraUpZ:Float;
	var screenWidth:Float;
	var screenHeight:Float;

	var defaultShader:Shader;
	var defaultShaderTexture:Shader;
	var currentShader:Shader;

	var currentTexture:Texture;
	var texMode:TextureMode;

	var modelMat:UniformMatrix;
	var viewMat:UniformMatrix;
	var modelviewMat:UniformMatrix;
	var normalMat:UniformMatrix;
	var projMat:UniformMatrix;
	var mvpMat:UniformMatrix;

	var sceneOpen:Bool;

	static inline var MAT_STACK_SIZE:Int = 1024;
	var matStack:Vector<Float>;
	var matStackCount:Int;

	var currentRenderTarget:Texture;

	static inline var MAX_LIGHTS:Int = 8;
	var numLights:Int;
	var lightBuf:Vector<Light>;

	var materialAmb:Float;
	var materialDif:Float;
	var materialSpc:Float;
	var materialShn:Float;
	var materialEmi:Float;

	public function new(canvas:CanvasElement) {
		this.canvas = canvas;
		gl = canvas.getContextWebGL();

		initGL();
		initShaders();
		initBuffers();
		initMatricies();

		// camera
		cameraNear = 0.1;
		cameraFar = 10000;
		cameraFov = Math.PI / 3;
		cameraSet = false;
		cameraPosX = 0;
		cameraPosY = 0;
		cameraPosZ = 1;
		cameraAtX = 0;
		cameraAtY = 0;
		cameraAtZ = 0;
		cameraUpX = 0;
		cameraUpY = 1;
		cameraUpZ = 0;

		// screen
		screen(canvas.width, canvas.height);

		// shapes
		numVertices = 0;
		colorR = 1;
		colorG = 1;
		colorB = 1;
		colorA = 1;
		normalX = 0;
		normalY = 0;
		normalZ = 0;
		texCoordU = 0;
		texCoordV = 0;

		// stack
		matStack = new Vector(MAT_STACK_SIZE);
		matStackCount = 0;

		// textures
		currentTexture = null;
		currentRenderTarget = null;
		texMode = Normal;

		// scene
		sceneOpen = false;

		// lights
		lightBuf = new Vector(MAX_LIGHTS);
		for (i in 0...MAX_LIGHTS) {
			lightBuf[i] = new Light();
		}
		numLights = 0;
	}

	function initGL():Void {
		gl.getExtension("OES_texture_float");
		gl.getExtension("OES_texture_float_linear");
		gl.getExtension("WEBGL_color_buffer_float");
		gl.disable(GL.DEPTH_TEST);
		gl.enable(GL.BLEND);
		gl.frontFace(GL.CCW);
		gl.cullFace(GL.BACK);
		gl.disable(GL.CULL_FACE);
		gl.blendFuncSeparate(GL.SRC_ALPHA, GL.ONE_MINUS_SRC_ALPHA, GL.ONE, GL.ONE);
	}

	public function init2D():Void {
		gl.disable(GL.DEPTH_TEST);
		gl.frontFace(GL.CCW);
		gl.cullFace(GL.BACK);
		gl.disable(GL.CULL_FACE);
		resetCamera();
		resetViewport();
		noTexture();
	}

	public function init3D():Void {
		gl.enable(GL.DEPTH_TEST);
		gl.frontFace(GL.CCW);
		gl.cullFace(GL.BACK);
		gl.enable(GL.CULL_FACE);
		resetCamera();
		resetViewport();
		noTexture();
	}

	function initMatricies():Void {
		modelMat = new UniformMatrix();
		viewMat = new UniformMatrix();
		modelviewMat = new UniformMatrix();
		normalMat = new UniformMatrix();
		projMat = new UniformMatrix();
		mvpMat = new UniformMatrix();
	}

	function initShaders():Void {
		var vertexSource:String = "
			attribute vec4 position;
			attribute vec4 color;
			attribute vec3 normal;
			attribute vec2 texCoord;

			uniform mat4 transform;
			uniform mat4 modelview;
			uniform mat3 normalMatrix;

			varying vec4 vColor;
			varying vec3 vPosition;
			varying vec3 vNormal;
			varying vec2 vTexCoord;

			void main() {
				gl_Position = transform * position;

				vColor = color;
				vPosition = (modelview * position).xyz;
				vNormal = normalMatrix * normal;
				vTexCoord = texCoord;
			}
		";
		var fragmentSource:String = "
			precision mediump float;
			varying vec4 vColor;
			varying vec3 vPosition;
			varying vec3 vNormal;
			varying vec2 vTexCoord;
			uniform float ambient;
			uniform float diffuse;
			uniform float specular;
			uniform float shininess;
			uniform float emission;
			uniform int numLights;
			uniform vec4 lightPositions[8];
			uniform vec3 lightColors[8];
			uniform vec3 lightNormals[8];

			void main() {
				if (numLights == 0) {
					gl_FragColor = vColor;
					return;
				}

				vec3 eye = normalize(vPosition);
				vec3 n = normalize(vNormal);
				vec3 color = vColor.xyz;
				vec3 ambientTotal = vec3(0);
				vec3 diffuseTotal = vec3(0);
				vec3 specularTotal = vec3(0);
				vec3 emissionTotal = color * emission;

				if (!gl_FrontFacing) n = -n;

				for (int i = 0; i < 8; i++) {
					if (i == numLights) break;
					vec4 lp = lightPositions[i];
					vec3 lc = lightColors[i];
					vec3 ln = lightNormals[i];
					bool amb = lp.w == 0.0 && dot(ln, ln) == 0.0;
					bool dir = lp.w == 0.0;
					if (amb) {
						ambientTotal += lc * color * ambient;
					} else {
						if (!dir) {
							ln = normalize(vPosition - lp.xyz);
							// TODO: spot light
						}
						float ldot = -dot(ln, n);
						if (ldot < 0.0) ldot = 0.0;
						diffuseTotal += lc * color * ldot * diffuse;
						if (ldot > 0.0) {
							vec3 reflEye = eye - 2.0 * n * dot(eye, n);
							float rdot = -dot(reflEye, ln);
							if (rdot < 0.0) rdot = 0.0;
							specularTotal += lc * pow(rdot, shininess) * specular;
						}
					}
				}
				gl_FragColor = vec4(ambientTotal + diffuseTotal + specularTotal + emissionTotal, vColor.w);
			}
		";
		var fragmentTextureSource:String = "
			precision mediump float;
			uniform sampler2D texture;

			varying vec4 vColor;
			varying vec3 vPosition;
			varying vec3 vNormal;
			varying vec2 vTexCoord;
			uniform float ambient;
			uniform float diffuse;
			uniform float specular;
			uniform float shininess;
			uniform float emission;
			uniform int numLights;
			uniform vec4 lightPositions[8];
			uniform vec3 lightColors[8];
			uniform vec3 lightNormals[8];

			void main() {
				vec4 texMulColor = texture2D(texture, vTexCoord) * vColor;
				if (numLights == 0) {
					gl_FragColor = texMulColor;
					return;
				}

				vec3 eye = normalize(vPosition);
				vec3 n = normalize(vNormal);
				vec3 color = texMulColor.xyz;
				vec3 ambientTotal = vec3(0);
				vec3 diffuseTotal = vec3(0);
				vec3 specularTotal = vec3(0);
				vec3 emissionTotal = color * emission;

				if (!gl_FrontFacing) n = -n;

				for (int i = 0; i < 8; i++) {
					if (i == numLights) break;
					vec4 lp = lightPositions[i];
					vec3 lc = lightColors[i];
					vec3 ln = lightNormals[i];
					bool amb = lp.w == 0.0 && dot(ln, ln) == 0.0;
					bool dir = lp.w == 0.0;
					if (amb) {
						ambientTotal += lc * color * ambient;
					} else {
						if (!dir) {
							ln = normalize(vPosition - lp.xyz);
							// TODO: spot light
						}
						float ldot = -dot(ln, n);
						if (ldot < 0.0) ldot = 0.0;
						diffuseTotal += lc * color * ldot * diffuse;
						if (ldot > 0.0) {
							vec3 reflEye = eye - 2.0 * n * dot(eye, n);
							float rdot = -dot(reflEye, ln);
							if (rdot < 0.0) rdot = 0.0;
							specularTotal += lc * pow(rdot, shininess) * specular;
						}
					}
				}
				gl_FragColor = vec4(ambientTotal + diffuseTotal + specularTotal + emissionTotal, texMulColor.w);
			}
		";
		defaultShader = new Shader(gl);
		defaultShader.vertexShader(vertexSource);
		defaultShader.fragmentShader(fragmentSource);
		defaultShader.compile();
		defaultShaderTexture = new Shader(gl);
		defaultShaderTexture.vertexShader(vertexSource);
		defaultShaderTexture.fragmentShader(fragmentTextureSource);
		defaultShaderTexture.compile();
		currentShader = null;
	}

	function initBuffers():Void {
		positionBuf = new VertexBuffer(gl, "position", 3);
		colorBuf = new VertexBuffer(gl, "color", 4);
		normalBuf = new VertexBuffer(gl, "normal", 3);
		texCoordBuf = new VertexBuffer(gl, "texCoord", 2);
		indexBuf = new IndexBuffer(gl);
	}

	function chooseShader():Shader {
		if (currentShader != null) return currentShader;
		if (currentTexture != null) return defaultShaderTexture;
		return defaultShader;
	}

	/**
	 * Sets virtual screen size to `(width, height)`. This does *not* change viewport.
	 */
	public inline function screen(width:Float, height:Float):Void {
		screenWidth = width;
		screenHeight = height;
		if (currentRenderTarget == null) resetViewport();
		perspective();
	}

	public inline function viewport(x:Int, y:Int, width:Int, height:Int):Void {
		var targetHeight:Int = currentRenderTarget == null ? canvas.height : currentRenderTarget.textureHeight;
		gl.viewport(x, targetHeight - height - y, width, height);
	}

	public inline function resetViewport():Void {
		var width:Int;
		var height:Int;
		var targetHeight:Int;
		if (currentRenderTarget == null) {
			width = canvas.width;
			height = canvas.height;
			targetHeight = height;
		} else {
			width = currentRenderTarget.width;
			height = currentRenderTarget.height;
			targetHeight = currentRenderTarget.textureHeight;
		}
		gl.viewport(0, targetHeight - height, width, height);
	}

	public inline function getRawGL():GL {
		return gl;
	}

	public function beginScene():Void {
		if (sceneOpen) throw new Error("scene already begun");
		sceneOpen = true;

		modelMat.identity();
		numLights = 0;
		currentTexture = null;
		colorR = 1;
		colorG = 1;
		colorB = 1;
		colorA = 1;
		normalX = 0;
		normalY = 0;
		normalZ = 0;
		texCoordU = 0;
		texCoordV = 0;
		materialAmb = 1;
		materialDif = 1;
		materialSpc = 0;
		materialShn = 10;
		materialEmi = 0;

		if (!cameraSet) {
			defaultCameraPosX = screenWidth * 0.5;
			defaultCameraPosY = screenHeight * 0.5;
			defaultCameraPosZ = -screenHeight / (2 * Math.tan(cameraFov * 0.5));
			viewMat.lookAt(defaultCameraPosX, defaultCameraPosY, defaultCameraPosZ, defaultCameraPosX, defaultCameraPosY, 0, 0, -1, 0);
		}
	}

	public inline function endScene():Void {
		if (!sceneOpen) throw new Error("scene already ended");
		sceneOpen = false;
		gl.flush();
	}

	public inline function enableDepthTest():Void {
		gl.enable(GL.DEPTH_TEST);
	}

	public inline function disableDepthTest():Void {
		gl.disable(GL.DEPTH_TEST);
	}

	public inline function culling(face:Face):Void {
		if (face == None) {
			gl.disable(GL.CULL_FACE);
		} else {
			gl.enable(GL.CULL_FACE);
			gl.cullFace(face);
		}
	}

	public inline function clear(r:Float, g:Float, b:Float, ?a:Float = 1):Void {
		gl.clearColor(r, g, b, a);
		gl.clearDepth(1);
		gl.clear(GL.COLOR_BUFFER_BIT | GL.DEPTH_BUFFER_BIT);
	}

	public inline function createShader(vertexSource:String, fragmentSource:String):Shader {
		var shader:Shader = new Shader(gl);
		shader.vertexShader(vertexSource);
		shader.fragmentShader(fragmentSource);
		shader.compile();
		return shader;
	}

	public inline function shader(shader:Shader):Void {
		currentShader = shader;
	}

	public inline function resetShader():Void {
		currentShader = null;
	}

	public inline function createTexture(width:Int, height:Int, floating:Bool = false):Texture {
		var img:Texture = new Texture(gl);
		img.init(width, height, floating);
		return img;
	}

	public inline function loadImage(image:Image, scalingMode:ScalingMode = Scale, floating:Bool = false):Texture {
		var img:Texture = new Texture(gl);
		img.load(image, scalingMode, floating);
		return img;
	}

	public inline function loadImageTo(texture:Texture, image:Image, scalingMode:ScalingMode = Scale, floating:Bool = false):Void {
		texture.load(image, scalingMode, floating);
	}

	public inline function renderTarget(target:Texture):Void {
		currentRenderTarget = target;
		if (target == null) {
			gl.bindFramebuffer(GL.FRAMEBUFFER, null);
			gl.viewport(0, 0, canvas.width, canvas.height);
		} else {
			gl.bindFramebuffer(GL.FRAMEBUFFER, target.frameBuffer);
			gl.viewport(0, target.textureHeight - target.height, target.width, target.height);
		}
	}

	public function pushMatrix():Void {
		if (matStackCount > MAT_STACK_SIZE - 16) {
			throw new Error("matrix stack overflowed");
		}
		matStack[matStackCount++] = modelMat.array[0];
		matStack[matStackCount++] = modelMat.array[1];
		matStack[matStackCount++] = modelMat.array[2];
		matStack[matStackCount++] = modelMat.array[3];
		matStack[matStackCount++] = modelMat.array[4];
		matStack[matStackCount++] = modelMat.array[5];
		matStack[matStackCount++] = modelMat.array[6];
		matStack[matStackCount++] = modelMat.array[7];
		matStack[matStackCount++] = modelMat.array[8];
		matStack[matStackCount++] = modelMat.array[9];
		matStack[matStackCount++] = modelMat.array[10];
		matStack[matStackCount++] = modelMat.array[11];
		matStack[matStackCount++] = modelMat.array[12];
		matStack[matStackCount++] = modelMat.array[13];
		matStack[matStackCount++] = modelMat.array[14];
		matStack[matStackCount++] = modelMat.array[15];
	}

	public function popMatrix():Void {
		if (matStackCount < 16) {
			throw new Error("cannot pop matrix");
		}
		modelMat.array[15] = matStack[--matStackCount];
		modelMat.array[14] = matStack[--matStackCount];
		modelMat.array[13] = matStack[--matStackCount];
		modelMat.array[12] = matStack[--matStackCount];
		modelMat.array[11] = matStack[--matStackCount];
		modelMat.array[10] = matStack[--matStackCount];
		modelMat.array[9] = matStack[--matStackCount];
		modelMat.array[8] = matStack[--matStackCount];
		modelMat.array[7] = matStack[--matStackCount];
		modelMat.array[6] = matStack[--matStackCount];
		modelMat.array[5] = matStack[--matStackCount];
		modelMat.array[4] = matStack[--matStackCount];
		modelMat.array[3] = matStack[--matStackCount];
		modelMat.array[2] = matStack[--matStackCount];
		modelMat.array[1] = matStack[--matStackCount];
		modelMat.array[0] = matStack[--matStackCount];
	}

	public inline function scale(sx:Float, sy:Float, sz:Float = 1):Void {
		modelMat.appendScale(sx, sy, sz);
	}

	public inline function rotate(ang:Float):Void {
		modelMat.appendRotation(ang, 0, 0, 1);
	}

	public inline function rotateX(ang:Float):Void {
		modelMat.appendRotation(ang, 1, 0, 0);
	}

	public inline function rotateY(ang:Float):Void {
		modelMat.appendRotation(ang, 0, 1, 0);
	}

	public inline function rotateZ(ang:Float):Void {
		modelMat.appendRotation(ang, 0, 0, 1);
	}

	public inline function translate(tx:Float, ty:Float, tz:Float = 0):Void {
		modelMat.appendTranslation(tx, ty, tz);
	}

	public inline function resetCamera():Void {
		cameraSet = false;
	}

	public inline function camera(posX:Float, posY:Float, posZ:Float, atX:Float, atY:Float, atZ:Float, upX:Float, upY:Float, upZ:Float):Void {
		cameraSet = true;
		cameraPosX = posX;
		cameraPosY = posY;
		cameraPosZ = posZ;
		cameraAtX = atX;
		cameraAtY = atY;
		cameraAtZ = atZ;
		cameraUpX = upX;
		cameraUpY = upY;
		cameraUpZ = upZ;
		viewMat.lookAt(cameraPosX, cameraPosY, cameraPosZ, cameraAtX, cameraAtY, cameraAtZ, cameraUpX, cameraUpY, cameraUpZ);
	}

	public inline function perspective(?fovY:Float, ?near:Float, ?far:Float):Void {
		if (fovY == null) {
			fovY = Math.PI / 3;
		}
		if (near == null) {
			near = 0.1;
		}
		if (far == null) {
			far = 1000;
		}
		cameraFov = fovY;
		cameraNear = near;
		cameraFar = far;
		projMat.perspective(fovY, screenWidth / screenHeight, near, far);
	}

	public function beginShape(mode:ShapeMode):Void {
		shapeMode = mode;
		numVertices = 0;
		positionBuf.clear();
		colorBuf.clear();
		normalBuf.clear();
		texCoordBuf.clear();
		indexBuf.clear();
	}

	public function image(img:Texture, srcX:Float, srcY:Float, srcW:Float, srcH:Float, dstX:Float, dstY:Float, dstW:Float, dstH:Float):Void {
		var tmpTex:Texture = currentTexture;
		var tmpTexMode:TextureMode = texMode;
		currentTexture = img;
		texMode = Image;

		beginShape(TriangleStrip);
		normal(0, 0, -1);
		texCoord(srcX, srcY);
		vertex(dstX, dstY, 0);
		texCoord(srcX, srcY + srcH);
		vertex(dstX, dstY + dstH, 0);
		texCoord(srcX + srcW, srcY);
		vertex(dstX + dstW, dstY, 0);
		texCoord(srcX + srcW, srcY + srcH);
		vertex(dstX + dstW, dstY + dstH, 0);
		endShape();

		currentTexture = tmpTex;
		texMode = tmpTexMode;
	}

	public function rect(x:Float, y:Float, width:Float, height:Float):Void {
		var tmpTexMode:TextureMode = texMode;
		texMode = Normal;

		beginShape(TriangleStrip);
		normal(0, 0, -1);
		if (currentTexture != null) {
			texCoord(0, 0);
			vertex(x, y, 0);
			texCoord(0, 1);
			vertex(x, y + height, 0);
			texCoord(1, 0);
			vertex(x + width, y, 0);
			texCoord(1, 1);
			vertex(x + width, y + height, 0);
		} else {
			vertex(x, y, 0);
			vertex(x, y + height, 0);
			vertex(x + width, y, 0);
			vertex(x + width, y + height, 0);
		}
		endShape();

		texMode = tmpTexMode;
	}

	public function line(a:Float, b:Float, c:Float, d:Float, ?e:Float, ?f:Float):Void {
		var tmpTex:Texture = currentTexture;
		var tmpNumLights:Int = numLights;
		currentTexture = null;
		numLights = 0;

		beginShape(Lines);
		if (e != null && f != null) {
			vertex(a, b, c);
			vertex(d, e, f);
		} else {
			vertex(a, b, 0);
			vertex(c, d, 0);
		}
		endShape();

		currentTexture = tmpTex;
		numLights = tmpNumLights;
	}

	public function box(width:Float, height:Float, depth:Float):Void {
		var tmpTexMode:TextureMode = texMode;
		texMode = Normal;

		beginShape(Triangles);
		width *= 0.5;
		height *= 0.5;
		depth *= 0.5;
		var x1:Float = -width;
		var x2:Float = width;
		var y1:Float = -height;
		var y2:Float = height;
		var z1:Float = -depth;
		var z2:Float = depth;

		if (currentTexture != null) {
			normal(-1, 0, 0);
			boxFaceUV(
				x1, y1, z1,
				x1, y1, z2,
				x1, y2, z2,
				x1, y2, z1
			);
			normal(1, 0, 0);
			boxFaceUV(
				x2, y1, z1,
				x2, y2, z1,
				x2, y2, z2,
				x2, y1, z2
			);
			normal(0, -1, 0);
			boxFaceUV(
				x1, y1, z1,
				x2, y1, z1,
				x2, y1, z2,
				x1, y1, z2
			);
			normal(0, 1, 0);
			boxFaceUV(
				x1, y2, z1,
				x1, y2, z2,
				x2, y2, z2,
				x2, y2, z1
			);
			normal(0, 0, -1);
			boxFaceUV(
				x1, y1, z1,
				x1, y2, z1,
				x2, y2, z1,
				x2, y1, z1
			);
			normal(0, 0, 1);
			boxFaceUV(
				x1, y1, z2,
				x2, y1, z2,
				x2, y2, z2,
				x1, y2, z2
			);
		} else {
			normal(-1, 0, 0);
			boxFace(
				x1, y1, z1,
				x1, y1, z2,
				x1, y2, z2,
				x1, y2, z1
			);
			normal(1, 0, 0);
			boxFace(
				x2, y1, z1,
				x2, y2, z1,
				x2, y2, z2,
				x2, y1, z2
			);
			normal(0, -1, 0);
			boxFace(
				x1, y1, z1,
				x2, y1, z1,
				x2, y1, z2,
				x1, y1, z2
			);
			normal(0, 1, 0);
			boxFace(
				x1, y2, z1,
				x1, y2, z2,
				x2, y2, z2,
				x2, y2, z1
			);
			normal(0, 0, -1);
			boxFace(
				x1, y1, z1,
				x1, y2, z1,
				x2, y2, z1,
				x2, y1, z1
			);
			normal(0, 0, 1);
			boxFace(
				x1, y1, z2,
				x2, y1, z2,
				x2, y2, z2,
				x1, y2, z2
			);
		}
		endShape();

		texMode = tmpTexMode;
	}

	@:extern
	inline function boxFace(
		x1:Float, y1:Float, z1:Float,
		x2:Float, y2:Float, z2:Float,
		x3:Float, y3:Float, z3:Float,
		x4:Float, y4:Float, z4:Float
	):Void {
		vertex(x1, y1, z1);
		vertex(x2, y2, z2);
		vertex(x3, y3, z3);
		vertex(x1, y1, z1);
		vertex(x3, y3, z3);
		vertex(x4, y4, z4);
	}

	@:extern
	inline function boxFaceUV(
		x1:Float, y1:Float, z1:Float,
		x2:Float, y2:Float, z2:Float,
		x3:Float, y3:Float, z3:Float,
		x4:Float, y4:Float, z4:Float
	):Void {
		texCoord(0, 0);
		vertex(x1, y1, z1);
		texCoord(0, 1);
		vertex(x2, y2, z2);
		texCoord(1, 1);
		vertex(x3, y3, z3);
		texCoord(0, 0);
		vertex(x1, y1, z1);
		texCoord(1, 1);
		vertex(x3, y3, z3);
		texCoord(1, 0);
		vertex(x4, y4, z4);
	}

	public inline function color(r:Float, g:Float, b:Float, a:Float = 1):Void {
		colorR = r;
		colorG = g;
		colorB = b;
		colorA = a;
	}

	public inline function normal(x:Float, y:Float, z:Float):Void {
		normalX = x;
		normalY = y;
		normalZ = z;
	}

	public inline function noLights():Void {
		if (!sceneOpen) throw new Error("begin scene before setting lights");
		numLights = 0;
	}

	public inline function lights():Void {
		ambientLight(0.2, 0.2, 0.2);
		directionalLight(0.8, 0.8, 0.8, -viewMat.array[2], -viewMat.array[6], -viewMat.array[10]);
	}

	public inline function ambientLight(r:Float, g:Float, b:Float):Void {
		if (!sceneOpen) throw new Error("begin scene before setting lights");
		if (numLights == MAX_LIGHTS) throw new Error("too many lights");
		var light:Light = lightBuf[numLights++];
		light.r = r;
		light.g = g;
		light.b = b;
		light.posX = 0;
		light.posY = 0;
		light.posZ = 0;
		light.posW = 0;
		light.norX = 0;
		light.norY = 0;
		light.norZ = 0;
	}

	public inline function directionalLight(r:Float, g:Float, b:Float, dirX:Float, dirY:Float, dirZ:Float):Void {
		if (!sceneOpen) throw new Error("begin scene before setting lights");
		if (numLights == MAX_LIGHTS) throw new Error("too many lights");
		var light:Light = lightBuf[numLights++];

		var invLen:Float = Math.sqrt(dirX * dirX + dirY * dirY + dirZ * dirZ);
		if (invLen > 0) invLen = 1 / invLen;
		dirX *= invLen;
		dirY *= invLen;
		dirZ *= invLen;

		light.r = r;
		light.g = g;
		light.b = b;
		light.posX = 0;
		light.posY = 0;
		light.posZ = 0;
		light.posW = 0;
		light.norX = dirX;
		light.norY = dirY;
		light.norZ = dirZ;
	}

	public inline function pointLight(r:Float, g:Float, b:Float, x:Float, y:Float, z:Float):Void {
		if (!sceneOpen) throw new Error("begin scene before setting lights");
		if (numLights == MAX_LIGHTS) throw new Error("too many lights");
		var light:Light = lightBuf[numLights++];
		light.r = r;
		light.g = g;
		light.b = b;
		light.posX = x;
		light.posY = y;
		light.posZ = z;
		light.posW = 1;
		light.norX = 0;
		light.norY = 0;
		light.norZ = 0;
	}

	public inline function ambient(v:Float):Void {
		if (!sceneOpen) throw new Error("begin scene before setting material");
		materialAmb = v;
	}

	public inline function diffuse(v:Float):Void {
		if (!sceneOpen) throw new Error("begin scene before setting material");
		materialDif = v;
	}

	public inline function specular(v:Float):Void {
		if (!sceneOpen) throw new Error("begin scene before setting material");
		materialSpc = v;
	}

	public inline function shininess(v:Float):Void {
		if (!sceneOpen) throw new Error("begin scene before setting material");
		materialShn = v;
	}

	public inline function emission(v:Float):Void {
		if (!sceneOpen) throw new Error("begin scene before setting material");
		materialEmi = v;
	}

	public inline function texCoord(u:Float, v:Float):Void {
		if (currentTexture == null) throw new Error("set texture before calling texCoord");

		switch (texMode) {
		case Image:
			u *= currentTexture.imageToU;
			v *= currentTexture.imageToV;
		case Normal:
			u *= currentTexture.normalToU;
			v *= currentTexture.normalToV;
		case Raw:
			// do nothing
		}

		texCoordU = u;
		texCoordV = 1 - v;
	}

	public inline function textureMode(mode:TextureMode):Void {
		this.texMode = mode;
	}

	public inline function texture(tex:Texture):Void {
		currentTexture = tex;
	}

	public inline function noTexture():Void {
		currentTexture = null;
	}

	public function vertex(x:Float, y:Float, z:Float = 0):Void {
		indexBuf.push(numVertices++);
		positionBuf.push(x);
		positionBuf.push(y);
		positionBuf.push(z);
		colorBuf.push(colorR);
		colorBuf.push(colorG);
		colorBuf.push(colorB);
		colorBuf.push(colorA);
		normalBuf.push(normalX);
		normalBuf.push(normalY);
		normalBuf.push(normalZ);
		texCoordBuf.push(texCoordU);
		texCoordBuf.push(texCoordV);
	}

	public inline function endShape():Void {
		var shader:Shader = chooseShader();
		var pg:Program = shader.pg;

		modelviewMat.mul(viewMat, modelMat);
		normalMat.inverse(modelviewMat);
		normalMat.transpose(normalMat);

		// matrix
		if (shader.hasUniform("model")) {
			shader.setMatrix("model", modelMat.array);
		}
		if (shader.hasUniform("view")) {
			shader.setMatrix("view", viewMat.array);
		}
		if (shader.hasUniform("projection")) {
			shader.setMatrix("projection", projMat.array);
		}
		if (shader.hasUniform("transform")) {
			mvpMat.mul(projMat, modelviewMat);
			shader.setMatrix("transform", mvpMat.array);
		}
		if (shader.hasUniform("modelview")) {
			shader.setMatrix("modelview", modelviewMat.array);
		}
		if (shader.hasUniform("normalMatrix")) {
			normalMat.toMat3();
			shader.setMatrix("normalMatrix", normalMat.array3);
		}

		// texture
		if (currentTexture != null) {
			shader.setTexture("texture", currentTexture);
			if (shader.hasUniform("texResolution")) {
				shader.setFloat("texResolution", currentTexture.textureWidth, currentTexture.textureHeight);
			}
			if (shader.hasUniform("texViewport")) {
				var x:Float = 0;
				var y:Float = 1 - currentTexture.height / currentTexture.textureHeight;
				var w:Float = currentTexture.width / currentTexture.textureWidth;
				var h:Float = currentTexture.height / currentTexture.textureHeight;
				shader.setFloat("texViewport", x, y, w, h);
			}
		}

		// material
		if (shader.hasUniform("ambient")) {
			shader.setFloat("ambient", materialAmb);
		}
		if (shader.hasUniform("diffuse")) {
			shader.setFloat("diffuse", materialDif);
		}
		if (shader.hasUniform("specular")) {
			shader.setFloat("specular", materialSpc);
		}
		if (shader.hasUniform("shininess")) {
			shader.setFloat("shininess", materialShn);
		}
		if (shader.hasUniform("emission")) {
			shader.setFloat("emission", materialEmi);
		}

		// lights
		if (shader.hasUniform("numLights")) {
			shader.setInt("numLights", numLights);
		}
		if (shader.hasUniform("lightPositions")) {
			for (i in 0...numLights) {
				var light:Light = lightBuf[i];
				var x:Float = light.posX * viewMat.array[0] + light.posY * viewMat.array[4] + light.posZ * viewMat.array[8] + viewMat.array[12];
				var y:Float = light.posX * viewMat.array[1] + light.posY * viewMat.array[5] + light.posZ * viewMat.array[9] + viewMat.array[13];
				var z:Float = light.posX * viewMat.array[2] + light.posY * viewMat.array[6] + light.posZ * viewMat.array[10] + viewMat.array[14];
				shader.setFloat('lightPositions[$i]', x, y, z, light.posW);
			}
		}
		if (shader.hasUniform("lightNormals")) {
			for (i in 0...numLights) {
				var light:Light = lightBuf[i];
				var x:Float = light.norX * viewMat.array[0] + light.norY * viewMat.array[4] + light.norZ * viewMat.array[8];
				var y:Float = light.norX * viewMat.array[1] + light.norY * viewMat.array[5] + light.norZ * viewMat.array[9];
				var z:Float = light.norX * viewMat.array[2] + light.norY * viewMat.array[6] + light.norZ * viewMat.array[10];
				shader.setFloat('lightNormals[$i]', x, y, z);
			}
		}
		if (shader.hasUniform("lightColors")) {
			for (i in 0...numLights) {
				var light:Light = lightBuf[i];
				shader.setFloat('lightColors[$i]', light.r, light.g, light.b);
			}
		}

		indexBuf.upload();
		positionBuf.upload();
		colorBuf.upload();
		normalBuf.upload();
		texCoordBuf.upload();

		positionBuf.bind(pg);
		colorBuf.bind(pg);
		normalBuf.bind(pg);
		texCoordBuf.bind(pg);

		shader.bind();
		indexBuf.draw(pg, shapeMode);
	}

}

private class Light {
	public var posX:Float;
	public var posY:Float;
	public var posZ:Float;
	public var posW:Float; // 0.0 if directional
	public var norX:Float;
	public var norY:Float;
	public var norZ:Float;
	public var r:Float;
	public var g:Float;
	public var b:Float;

	public function new() {
	}
}

private class UniformMatrix {
	// column-major order
	public var array:Array<Float>;
	public var array3:Array<Float>;

	public function new() {
		array = [for (i in 0...16) 0];
		array3 = [for (i in 0...9) 0];
	}

	public function identity():Void {
		set(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1
		);
	}

	public function lookAt(eyeX:Float, eyeY:Float, eyeZ:Float, atX:Float, atY:Float, atZ:Float, upX:Float, upY:Float, upZ:Float):Void {
		var zx:Float = eyeX - atX;
		var zy:Float = eyeY - atY;
		var zz:Float = eyeZ - atZ;
		var tmp:Float = 1 / Math.sqrt(zx * zx + zy * zy + zz * zz);
		zx *= tmp;
		zy *= tmp;
		zz *= tmp;
		var xx:Float = upY * zz - upZ * zy;
		var xy:Float = upZ * zx - upX * zz;
		var xz:Float = upX * zy - upY * zx;
		tmp = 1 / Math.sqrt(xx * xx + xy * xy + xz * xz);
		xx *= tmp;
		xy *= tmp;
		xz *= tmp;
		var yx:Float = zy * xz - zz * xy;
		var yy:Float = zz * xx - zx * xz;
		var yz:Float = zx * xy - zy * xx;
		set(
			xx, xy, xz, -(xx * eyeX + xy * eyeY + xz * eyeZ),
			yx, yy, yz, -(yx * eyeX + yy * eyeY + yz * eyeZ),
			zx, zy, zz, -(zx * eyeX + zy * eyeY + zz * eyeZ),
			0, 0, 0, 1
		);
	}

	public function perspective(fovY:Float, aspect:Float, near:Float, far:Float):Void {
		var h:Float = 1 / Math.tan(fovY * 0.5);
		var fnf:Float = far / (near - far);
		set(
			h / aspect, 0, 0, 0,
			0, h, 0, 0,
			0, 0, fnf, near * fnf,
			0, 0, -1, 0
		);
	}

	public function ortho(width:Float, height:Float, near:Float, far:Float):Void {
		var nf:Float = 1 / (near - far);
		set(
			2 / width, 0, 0, 0,
			0, 2 / height, 0, 0,
			0, 0, nf, near * nf,
			0, 0, 0, 1
		);
	}

	public function appendScale(sx:Float, sy:Float, sz:Float):Void {
		array[0] *= sx;
		array[1] *= sx;
		array[2] *= sx;
		array[3] *= sx;
		array[4] *= sy;
		array[5] *= sy;
		array[6] *= sy;
		array[7] *= sy;
		array[8] *= sz;
		array[9] *= sz;
		array[10] *= sz;
		array[11] *= sz;
	}

	public function appendRotation(rad:Float, ax:Float, ay:Float, az:Float):Void {
		var s:Float = Math.sin(rad);
		var c:Float = Math.cos(rad);
		var c1:Float = 1 - c;
		var r00:Float = ax * ax * c1 + c;
		var r01:Float = ax * ay * c1 - az * s;
		var r02:Float = ax * az * c1 + ay * s;
		var r10:Float = ay * ax * c1 + az * s;
		var r11:Float = ay * ay * c1 + c;
		var r12:Float = ay * az * c1 - ax * s;
		var r20:Float = az * ax * c1 - ay * s;
		var r21:Float = az * ay * c1 + ax * s;
		var r22:Float = az * az * c1 + c;
		var e00:Float = array[0];
		var e10:Float = array[1];
		var e20:Float = array[2];
		var e30:Float = array[3];
		var e01:Float = array[4];
		var e11:Float = array[5];
		var e21:Float = array[6];
		var e31:Float = array[7];
		var e02:Float = array[8];
		var e12:Float = array[9];
		var e22:Float = array[10];
		var e32:Float = array[11];
		var e03:Float = array[12];
		var e13:Float = array[13];
		var e23:Float = array[14];
		var e33:Float = array[15];
		set(
			e00 * r00 + e01 * r10 + e02 * r20,
			e00 * r01 + e01 * r11 + e02 * r21,
			e00 * r02 + e01 * r12 + e02 * r22,
			e03,
			e10 * r00 + e11 * r10 + e12 * r20,
			e10 * r01 + e11 * r11 + e12 * r21,
			e10 * r02 + e11 * r12 + e12 * r22,
			e13,
			e20 * r00 + e21 * r10 + e22 * r20,
			e20 * r01 + e21 * r11 + e22 * r21,
			e20 * r02 + e21 * r12 + e22 * r22,
			e23,
			e30 * r00 + e31 * r10 + e32 * r20,
			e30 * r01 + e31 * r11 + e32 * r21,
			e30 * r02 + e31 * r12 + e32 * r22,
			e33
		);
	}

	public function mul(a:UniformMatrix, b:UniformMatrix):Void {
		var a00:Float = a.array[0];
		var a10:Float = a.array[1];
		var a20:Float = a.array[2];
		var a30:Float = a.array[3];
		var a01:Float = a.array[4];
		var a11:Float = a.array[5];
		var a21:Float = a.array[6];
		var a31:Float = a.array[7];
		var a02:Float = a.array[8];
		var a12:Float = a.array[9];
		var a22:Float = a.array[10];
		var a32:Float = a.array[11];
		var a03:Float = a.array[12];
		var a13:Float = a.array[13];
		var a23:Float = a.array[14];
		var a33:Float = a.array[15];
		var b00:Float = b.array[0];
		var b10:Float = b.array[1];
		var b20:Float = b.array[2];
		var b30:Float = b.array[3];
		var b01:Float = b.array[4];
		var b11:Float = b.array[5];
		var b21:Float = b.array[6];
		var b31:Float = b.array[7];
		var b02:Float = b.array[8];
		var b12:Float = b.array[9];
		var b22:Float = b.array[10];
		var b32:Float = b.array[11];
		var b03:Float = b.array[12];
		var b13:Float = b.array[13];
		var b23:Float = b.array[14];
		var b33:Float = b.array[15];
		set(
			a00 * b00 + a01 * b10 + a02 * b20 + a03 * b30,
			a00 * b01 + a01 * b11 + a02 * b21 + a03 * b31,
			a00 * b02 + a01 * b12 + a02 * b22 + a03 * b32,
			a00 * b03 + a01 * b13 + a02 * b23 + a03 * b33,
			a10 * b00 + a11 * b10 + a12 * b20 + a13 * b30,
			a10 * b01 + a11 * b11 + a12 * b21 + a13 * b31,
			a10 * b02 + a11 * b12 + a12 * b22 + a13 * b32,
			a10 * b03 + a11 * b13 + a12 * b23 + a13 * b33,
			a20 * b00 + a21 * b10 + a22 * b20 + a23 * b30,
			a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31,
			a20 * b02 + a21 * b12 + a22 * b22 + a23 * b32,
			a20 * b03 + a21 * b13 + a22 * b23 + a23 * b33,
			a30 * b00 + a31 * b10 + a32 * b20 + a33 * b30,
			a30 * b01 + a31 * b11 + a32 * b21 + a33 * b31,
			a30 * b02 + a31 * b12 + a32 * b22 + a33 * b32,
			a30 * b03 + a31 * b13 + a32 * b23 + a33 * b33
		);
	}

	public function transpose(a:UniformMatrix):Void {
		set(
			a.array[0], a.array[1], a.array[2], a.array[3],
			a.array[4], a.array[5], a.array[6], a.array[7],
			a.array[8], a.array[9], a.array[10], a.array[11],
			a.array[12], a.array[13], a.array[14], a.array[15]
		);
	}

	public function inverse(a:UniformMatrix):Void {
		var e00:Float = a.array[0];
		var e10:Float = a.array[1];
		var e20:Float = a.array[2];
		var e30:Float = a.array[3];
		var e01:Float = a.array[4];
		var e11:Float = a.array[5];
		var e21:Float = a.array[6];
		var e31:Float = a.array[7];
		var e02:Float = a.array[8];
		var e12:Float = a.array[9];
		var e22:Float = a.array[10];
		var e32:Float = a.array[11];
		var e03:Float = a.array[12];
		var e13:Float = a.array[13];
		var e23:Float = a.array[14];
		var e33:Float = a.array[15];
		var d01_01:Float = e00 * e11 - e01 * e10;
		var d01_02:Float = e00 * e12 - e02 * e10;
		var d01_03:Float = e00 * e13 - e03 * e10;
		var d01_12:Float = e01 * e12 - e02 * e11;
		var d01_13:Float = e01 * e13 - e03 * e11;
		var d01_23:Float = e02 * e13 - e03 * e12;
		var d23_01:Float = e20 * e31 - e21 * e30;
		var d23_02:Float = e20 * e32 - e22 * e30;
		var d23_03:Float = e20 * e33 - e23 * e30;
		var d23_12:Float = e21 * e32 - e22 * e31;
		var d23_13:Float = e21 * e33 - e23 * e31;
		var d23_23:Float = e22 * e33 - e23 * e32;
		var d00:Float = e11 * d23_23 - e12 * d23_13 + e13 * d23_12;
		var d01:Float = e10 * d23_23 - e12 * d23_03 + e13 * d23_02;
		var d02:Float = e10 * d23_13 - e11 * d23_03 + e13 * d23_01;
		var d03:Float = e10 * d23_12 - e11 * d23_02 + e12 * d23_01;
		var d10:Float = e01 * d23_23 - e02 * d23_13 + e03 * d23_12;
		var d11:Float = e00 * d23_23 - e02 * d23_03 + e03 * d23_02;
		var d12:Float = e00 * d23_13 - e01 * d23_03 + e03 * d23_01;
		var d13:Float = e00 * d23_12 - e01 * d23_02 + e02 * d23_01;
		var d20:Float = e31 * d01_23 - e32 * d01_13 + e33 * d01_12;
		var d21:Float = e30 * d01_23 - e32 * d01_03 + e33 * d01_02;
		var d22:Float = e30 * d01_13 - e31 * d01_03 + e33 * d01_01;
		var d23:Float = e30 * d01_12 - e31 * d01_02 + e32 * d01_01;
		var d30:Float = e21 * d01_23 - e22 * d01_13 + e23 * d01_12;
		var d31:Float = e20 * d01_23 - e22 * d01_03 + e23 * d01_02;
		var d32:Float = e20 * d01_13 - e21 * d01_03 + e23 * d01_01;
		var d33:Float = e20 * d01_12 - e21 * d01_02 + e22 * d01_01;
		var invDet:Float = e00 * d00 - e01 * d01 + e02 * d02 - e03 * d03;
		if (invDet != 0) invDet = 1 / invDet;
		set(
			d00 * invDet, -d10 * invDet, d20 * invDet, -d30 * invDet,
			-d01 * invDet, d11 * invDet, -d21 * invDet, d31 * invDet,
			d02 * invDet, -d12 * invDet, d22 * invDet, -d32 * invDet,
			-d03 * invDet, d13 * invDet, -d23 * invDet, d33 * invDet
		);
	}

	public function appendTranslation(tx:Float, ty:Float, tz:Float):Void {
		array[12] += array[0] * tx + array[4] * ty + array[8] * tz;
		array[13] += array[1] * tx + array[5] * ty + array[9] * tz;
		array[14] += array[2] * tx + array[6] * ty + array[10] * tz;
		array[15] += array[3] * tx + array[7] * ty + array[11] * tz;
	}

	public function toMat3():Void {
		array3[0] = array[0];
		array3[1] = array[1];
		array3[2] = array[2];
		array3[3] = array[4];
		array3[4] = array[5];
		array3[5] = array[6];
		array3[6] = array[8];
		array3[7] = array[9];
		array3[8] = array[10];
	}

	@:extern
	inline function set(
		e00:Float, e01:Float, e02:Float, e03:Float,
		e10:Float, e11:Float, e12:Float, e13:Float,
		e20:Float, e21:Float, e22:Float, e23:Float,
		e30:Float, e31:Float, e32:Float, e33:Float
	):Void {
		array[0] = e00;
		array[1] = e10;
		array[2] = e20;
		array[3] = e30;
		array[4] = e01;
		array[5] = e11;
		array[6] = e21;
		array[7] = e31;
		array[8] = e02;
		array[9] = e12;
		array[10] = e22;
		array[11] = e32;
		array[12] = e03;
		array[13] = e13;
		array[14] = e23;
		array[15] = e33;
	}
}

private class VertexBuffer {
	public var buffer:Buffer;
	public var name:String;
	public var size:Int;
	public var array:Float32Array;
	public var length:Int;

	var gl:GL;

	public function new(gl:GL, name:String, size:Int) {
		this.gl = gl;
		this.name = name;
		this.size = size;

		// init array
		array = new Float32Array(64);
		length = 0;

		// create buffer
		buffer = gl.createBuffer();
	}

	@:extern
	public inline function clear():Void {
		length = 0;
	}

	@:extern
	public inline function push(f:Float):Void {
		if (length == array.length) {
			expand();
		}
		array[length++] = f;
	}

	@:extern
	public inline function upload():Void {
		gl.bindBuffer(GL.ARRAY_BUFFER, buffer);
		gl.bufferData(GL.ARRAY_BUFFER, new Float32Array(array.buffer, 0, length), GL.STATIC_DRAW);
		gl.bindBuffer(GL.ARRAY_BUFFER, null);
	}

	@:extern
	public inline function bind(pg:Program):Void {
		var idx:Int = gl.getAttribLocation(pg, name);
		if (idx != -1) {
			gl.bindBuffer(GL.ARRAY_BUFFER, buffer);
			gl.enableVertexAttribArray(idx);
			gl.vertexAttribPointer(idx, size, GL.FLOAT, false, 0, 0);
			gl.bindBuffer(GL.ARRAY_BUFFER, null);
		}
	}

	@:extern
	public inline function expand():Void {
		// expand array
		var oldLength:Int = array.length;
		var newArray:Float32Array = new Float32Array(oldLength << 1);
		for (i in 0...oldLength) {
			newArray[i] = array[i];
		}
		array = newArray;
	}
}

private class IndexBuffer {
	public var buffer:Buffer;
	public var mode:ShapeMode;
	public var array:Int16Array;
	public var length:Int;

	var gl:GL;

	public function new(gl:GL) {
		this.gl = gl;

		// init array
		array = new Int16Array(512);
		length = 0;

		// create buffer
		buffer = gl.createBuffer();
	}

	@:extern
	public inline function clear():Void {
		length = 0;
	}

	@:extern
	public inline function push(i:Int):Void {
		if (length == array.length) {
			expand();
		}
		array[length++] = i;
	}

	@:extern
	public inline function upload():Void {
		gl.bindBuffer(GL.ELEMENT_ARRAY_BUFFER, buffer);
		gl.bufferData(GL.ELEMENT_ARRAY_BUFFER, new Int16Array(array.buffer, 0, length), GL.STATIC_DRAW);
		gl.bindBuffer(GL.ELEMENT_ARRAY_BUFFER, null);
	}

	@:extern
	public inline function draw(pg:Program, mode:Int):Void {
		gl.bindBuffer(GL.ELEMENT_ARRAY_BUFFER, buffer);
		gl.drawElements(mode, length, GL.UNSIGNED_SHORT, 0);
		gl.bindBuffer(GL.ELEMENT_ARRAY_BUFFER, null);
	}

	@:extern
	public inline function expand():Void {
		// expand array
		var oldLength:Int = array.length;
		var newArray:Int16Array = new Int16Array(oldLength << 1);
		for (i in 0...oldLength) {
			newArray[i] = array[i];
		}
		array = newArray;
	}
}
