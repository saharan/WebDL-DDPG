package pot.input;
import js.html.CanvasElement;
import js.html.Element;

/**
 * ...
 */
class InputTools {
	@:extern
	public static inline function clientX(e:Element):Float {
		return e.getBoundingClientRect().left;
	}

	@:extern
	public static inline function clientY(e:Element):Float {
		return e.getBoundingClientRect().top;
	}

	@:extern
	public static inline function scaleX(canvas:CanvasElement):Float {
		return canvas.width / canvas.clientWidth;
	}

	@:extern
	public static inline function scaleY(canvas:CanvasElement):Float {
		return canvas.height / canvas.clientHeight;
	}
}
