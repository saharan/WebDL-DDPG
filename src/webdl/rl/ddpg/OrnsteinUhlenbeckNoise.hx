package webdl.rl.ddpg;
import webdl.core.RandUtil;

/**
 * A generator of Ornstein-Uhlenbeck process.
 */
class OrnsteinUhlenbeckNoise {
	var dim:Int;
	var xs:Array<Float>;
	var mu:Float;
	var theta:Float;
	var sigma:Float;
	var dt:Float;

	public function new(dim:Int, mu:Float = 0, theta:Float = 0.15, sigma:Float = 0.3, dt:Float = 0.01) {
		this.dim = dim;
		this.mu = mu;
		this.theta = theta;
		this.sigma = sigma;
		this.dt = dt;
		xs = [for (i in 0...dim) 0];
	}

	public function next():Array<Float> {
		for (i in 0...dim) {
			xs[i] += (mu - xs[i]) * theta * dt + sigma * RandUtil.normal(0, dt);
		}
		return xs;
	}
}
