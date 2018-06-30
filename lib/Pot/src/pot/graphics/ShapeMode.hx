package pot.graphics;
import js.html.webgl.GL;

/**
 * list of the shape drawing modes
 */
@:enum
abstract ShapeMode(Int) to Int {
	var Points = GL.POINTS;
	var Lines = GL.LINES;
	var LineStrip = GL.LINE_STRIP;
	var LineLoop = GL.LINE_LOOP;
	var Triangles = GL.TRIANGLES;
	var TriangleStrip = GL.TRIANGLE_STRIP;
	var TriangleFan = GL.TRIANGLE_FAN;
}
