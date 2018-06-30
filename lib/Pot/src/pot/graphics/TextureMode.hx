package pot.graphics;

/**
 * list of the texture coordinate modes
 */
@:enum
abstract TextureMode(Int) {
	/**
	 * Scales so that `1` shows a pixel of the original image.
	 */
	var Image = 0;

	/**
	 * Scales so that `1` shows the width or height of the original image.
	 */
	var Normal = 1;

	/**
	 * Does not scale, texture coordinates will be sent to shaders directly.
	 */
	var Raw = 2;
}
