package pot.graphics;
import js.html.webgl.GL;

/**
 * ...
 */
@:enum
abstract TextureWrap(Int) to Int {
	var Repeat = GL.REPEAT;
	var Mirror = GL.MIRRORED_REPEAT;
	var Clamp = GL.CLAMP_TO_EDGE;
}
