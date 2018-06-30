package pot.graphics;
import js.html.webgl.GL;

/**
 * ...
 */
@:enum
abstract Face(Int) to Int {
	var Front = GL.FRONT;
	var Back = GL.BACK;
	var None = GL.NONE;
}
