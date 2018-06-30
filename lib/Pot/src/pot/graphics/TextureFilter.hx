package pot.graphics;
import js.html.webgl.GL;

/**
 * ...
 */
@:enum
abstract TextureFilter(Int) {
	var Nearest = GL.NEAREST;
	var Linear = GL.LINEAR;
}
