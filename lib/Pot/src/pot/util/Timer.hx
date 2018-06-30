package pot.util;
import js.Browser;

/**
 * An accurate timer
 */
class Timer {
	static inline var MIN_SLEEP_TIME:Int = 4;
	var frame:Void -> Void;
	var targetSleep:Float;
	var nextTime:Float;
	var running:Bool;

	public function new(frame:Void -> Void) {
		this.frame = frame;
		targetSleep = 1000 / 60;
	}

	public function start():Void {
		if (running) return;
		nextTime = now();
		running = true;
		Browser.window.setTimeout(loop, 0);
	}

	public function stop():Void {
		if (!running) return;
		running = false;
	}

	public function setFrameRate(frameRate:Float):Void {
		targetSleep = 1000 / frameRate;
	}

	function loop():Void {
		if (!running) return;
		frame();
		var currentTime:Float = now();

		nextTime += targetSleep;
		if (nextTime < currentTime + MIN_SLEEP_TIME) {
			nextTime = currentTime + MIN_SLEEP_TIME;
		}
		var sleep:Int = Std.int(nextTime - currentTime + 0.5);
		Browser.window.setTimeout(loop, sleep);
	}

	inline function now():Float {
		return untyped __js__("Date.now()");
	}

}
