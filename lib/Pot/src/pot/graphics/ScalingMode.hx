package pot.graphics;

/**
 * Texture scaling mode
 */
@:enum
abstract ScalingMode(Int) {
	/**
	 * Scales the original image so that it has the size of power of two.
	 */
	var Scale = 0;

	/**
	 * Does not scale and put the original image to the left-top of the scaled texture.
	 */
	var Original = 1;
}
