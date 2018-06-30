package pot.input;
import js.Browser;
import js.html.CanvasElement;
import js.html.Element;

/**
 * ...
 */
@:allow(pot.core)
class Input {
	public var mouse(default, null):Mouse;
	public var touches(default, null):Touches;
	public var keyboard(default, null):Keyboard;

	function new(canvas:CanvasElement) {
		mouse = new Mouse();
		touches = new Touches();
		keyboard = new Keyboard();
		addEvents(canvas);
	}

	function addEvents(canvas:CanvasElement):Void {
		mouse.addEvents(canvas, canvas);
		touches.addEvents(canvas, canvas);
		keyboard.addEvents(canvas, Browser.document.body);
	}

	function update():Void {
		mouse.update();
		touches.update();
		keyboard.update();
	}

}
