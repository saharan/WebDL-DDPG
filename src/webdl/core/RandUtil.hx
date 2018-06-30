package webdl.core;

/**
 * Utility class for generating random values.
 */
class RandUtil {
	/**
	 * Returns a random number generated from the normal distribution with mean `mu` and variance `sigma2`.
	 */
	public static inline function normal(mu:Float = 0, sigma2:Float = 1):Float {
		return mu + Math.sqrt(-2 * sigma2 * Math.log(Math.random())) * Math.sin(2 * Math.PI * Math.random());
	}

	/**
	 * Returns a random number generated from the uniform distribution of range (`min`, `max`).
	 */
	public static inline function uniform(min:Float = -1, max:Float = 1):Float {
		return Math.random() * (max - min) + min;
	}

	/**
	 * Returns the value at `x` of the probability density function of the normal distribution
	 * with mean `mu` and variance `sigma2`.
	 */
	public static inline function normalPdf(x:Float, mu:Float, sigma2:Float):Float {
		return 1 / Math.sqrt(2 * Math.PI * sigma2) * Math.exp(-(x - mu) * (x - mu) / (2 * sigma2));
	}
}
