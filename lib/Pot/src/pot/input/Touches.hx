package pot.input;
import js.html.CanvasElement;
import js.html.Element;
import js.html.TouchEvent;
import js.html.TouchList;
using pot.input.InputTools;

/**
 * ...
 */
@:allow(pot.input.Input)
abstract Touches(TouchesData) {
	inline function new() {
		this = new TouchesData();
	}

	public var length(get, never):Int;

	inline function get_length():Int {
		return this.touches.length;
	}

	@:arrayAccess
	inline function get(index:Int):Touch {
		return this.touches[index];
	}

	inline function addEvents(canvas:CanvasElement, elem:Element):Void {
		elem.addEventListener("touchstart", (e:TouchEvent) -> {
			if (e.cancelable) e.preventDefault();
			var touches:TouchList = e.changedTouches;
			for (i in 0...touches.length) {
				var rawTouch:js.html.Touch = touches[i];
				var rawId:Int = rawTouch.identifier;
				var touch:Touch = this.getByRawId(rawId, true);
				var x:Float = (rawTouch.clientX - elem.clientX()) * canvas.scaleX();
				var y:Float = (rawTouch.clientY - elem.clientY()) * canvas.scaleY();
				touch.begin(x, y);
			}
		});
		elem.addEventListener("touchmove", (e:TouchEvent) -> {
			if (e.cancelable) e.preventDefault();
			var touches:TouchList = e.changedTouches;
			for (i in 0...touches.length) {
				var rawTouch:js.html.Touch = touches[i];
				var rawId:Int = rawTouch.identifier;
				var touch:Touch = this.getByRawId(rawId);
				if (touch != null) {
					var x:Float = (rawTouch.clientX - elem.clientX()) * canvas.scaleX();
					var y:Float = (rawTouch.clientY - elem.clientY()) * canvas.scaleY();
					touch.move(x, y);
				}
			}
		});
		var end:TouchEvent -> Void = (e:TouchEvent) -> {
			if (e.cancelable) e.preventDefault();
			var touches:TouchList = e.changedTouches;
			for (i in 0...touches.length) {
				var rawTouch:js.html.Touch = touches[i];
				var rawId:Int = rawTouch.identifier;
				var touch:Touch = this.getByRawId(rawId);
				if (touch != null) {
					var x:Float = (rawTouch.clientX - elem.clientX()) * canvas.scaleX();
					var y:Float = (rawTouch.clientY - elem.clientY()) * canvas.scaleY();
					touch.end(x, y);
				}
			}
		};
		elem.addEventListener("touchend", end);
		elem.addEventListener("touchcancel", end);
	}

	inline function update():Void {
		var i:Int = 0;
		while (i < this.touches.length) {
			var touch:Touch = this.touches[i];
			if (!touch.ptouching && !touch.touching && !touch.ntouching) {
				// outdated
				this.touches.remove(touch);
			} else {
				touch.update();
				i++;
			}
		}
	}

	inline function iterator():Iterator<Touch> {
		return this.touches.iterator();
	}
}
