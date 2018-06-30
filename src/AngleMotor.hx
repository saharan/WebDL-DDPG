package;
import oimo.common.Vec3;
import oimo.dynamics.constraint.joint.Joint;
import oimo.dynamics.constraint.joint.RevoluteJoint;
import oimo.dynamics.constraint.joint.RotationalLimitMotor;
import oimo.dynamics.rigidbody.RigidBody;

/**
 * ...
 */
class AngleMotor {
	var j:Joint;
	var getAngle:Void -> Float;
	var getAxis:Void -> Vec3;
	var limot:RotationalLimitMotor;
	public var speed:Float;
	public var target:Float;
	public var torque:Float;
	public var min(default, null):Float;
	public var max(default, null):Float;

	public var angle(get, never):Float;
	function get_angle():Float {
		return getAngle();
	}

	public var axis(get, never):Vec3;
	function get_axis():Vec3 {
		return getAxis();
	}

	public var limitMotor(get, never):RotationalLimitMotor;
	function get_limitMotor():RotationalLimitMotor {
		return limot;
	}

	public var angularVelocity(get, never):Float;
	function get_angularVelocity():Float {
		var b1:RigidBody = j.getRigidBody1();
		var b2:RigidBody = j.getRigidBody2();
		var relAngVel:Vec3 = b2.getAngularVelocity().subEq(b1.getAngularVelocity());
		return relAngVel.dot(getAxis());
	}

	public function new(j:Joint, limot:RotationalLimitMotor, getAngle:Void -> Float, getAxis:Void -> Vec3) {
		this.j = j;
		this.limot = limot;
		this.getAngle = getAngle;
		this.getAxis = getAxis;
		speed = Math.PI * 2;
		target = 0;
		torque = 1.5;
		min = limot.lowerLimit;
		max = limot.upperLimit;
	}

	public function update():Void {
		min = limot.lowerLimit;
		max = limot.upperLimit;
		if (limot.lowerLimit < limot.upperLimit) {
			if (target < limot.lowerLimit) target = limot.lowerLimit;
			if (target > limot.upperLimit) target = limot.upperLimit;
		}
		var angle:Float = getAngle();
		var diff:Float = target - angle;
		var eps:Float = 15 * Math.PI / 180;
		var rot:Float = speed / 60;

		var spdMult:Float = (diff < 0 ? -diff : diff) / eps;
		if (spdMult > 1) spdMult = 1;

		rot *= spdMult;
		rot *= 60;

		limot.motorSpeed = diff > 0 ? rot : -rot;
		limot.motorTorque = torque;
	}

}
