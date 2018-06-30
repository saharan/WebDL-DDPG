package pot.core;
import js.html.CanvasElement;
import pot.graphics.Graphics;
import pot.util.Timer;

/**
 * Pot Engine
 */
class Pot {
	public var graphics(default, null):Graphics;
	public var width(default, null):Int;
	public var height(default, null):Int;
	var app:App;
	var canvas:CanvasElement;
	var timer:Timer;

	public function new(app:App, canvas:CanvasElement) {
		this.app = app;
		this.canvas = canvas;
		graphics = new Graphics(canvas);
		timer = new Timer(frame);
	}

	public function size(width:Int, height:Int):Void {
		this.width = width;
		this.height = height;
		canvas.width = width;
		canvas.height = height;
		canvas.style.width = width + "px";
		canvas.style.height = height + "px";
		graphics.screen(width, height);
	}

	public function frameRate(fps:Float):Void {
		timer.setFrameRate(fps);
	}

	public function start():Void {
		timer.start();
	}

	public function stop():Void {
		timer.stop();
	}

	function frame():Void {
		app.input.update();
		app.loop();
		app.frameCount++;
	}
}
