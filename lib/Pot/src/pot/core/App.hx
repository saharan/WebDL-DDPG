package pot.core;
import js.html.CanvasElement;
import pot.input.Input;

/**
 * Main application class
 */
@:allow(pot.core)
class App {
	/**
	 * Pot instance
	 */
	var pot:Pot;

	/**
	 * User input
	 */
	var input:Input;

	/**
	 * The canvas element
	 */
	var canvas:CanvasElement;

	/**
	 * The number of `App.frame` calls
	 */
	var frameCount:Int;

	public function new(canvas:CanvasElement) {
		this.canvas = canvas;
		input = new Input(canvas);
		pot = new Pot(this, canvas);
		frameCount = 0;
		setup();
	}

	/**
	 * Called on initialization
	 */
	function setup():Void {
	}

	/**
	 * Called every frame
	 */
	function loop():Void {
	}
}
