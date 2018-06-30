package;
import oimo.collision.geometry.BoxGeometry;
import oimo.collision.geometry.CapsuleGeometry;
import oimo.common.MathUtil;
import oimo.common.Vec3;
import oimo.dynamics.World;
import oimo.dynamics.constraint.joint.RagdollJoint;
import oimo.dynamics.constraint.joint.RagdollJointConfig;
import oimo.dynamics.constraint.joint.RevoluteJoint;
import oimo.dynamics.constraint.joint.RevoluteJointConfig;
import oimo.dynamics.constraint.joint.SphericalJoint;
import oimo.dynamics.constraint.joint.SphericalJointConfig;
import oimo.dynamics.constraint.joint.UniversalJoint;
import oimo.dynamics.constraint.joint.UniversalJointConfig;
import oimo.dynamics.rigidbody.RigidBody;
import oimo.dynamics.rigidbody.RigidBodyConfig;
import oimo.dynamics.rigidbody.Shape;
import oimo.dynamics.rigidbody.ShapeConfig;

/**
 * ...
 */
class Robot {
	public var head:RigidBody;
	public var upperBody:RigidBody;
	public var lowerBody:RigidBody;
	public var upperLegL:RigidBody;
	public var upperLegR:RigidBody;
	public var lowerLegL:RigidBody;
	public var lowerLegR:RigidBody;
	public var upperArmL:RigidBody;
	public var upperArmR:RigidBody;
	public var lowerArmL:RigidBody;
	public var lowerArmR:RigidBody;

	public var bodyHead:RagdollJoint;
	public var bodyLegL:UniversalJoint;
	public var bodyLegR:UniversalJoint;
	public var bodyArmL:UniversalJoint;
	public var bodyArmR:UniversalJoint;
	public var legLegL:RevoluteJoint;
	public var legLegR:RevoluteJoint;
	public var armArmL:RevoluteJoint;
	public var armArmR:RevoluteJoint;
	public var bodyBody:RevoluteJoint;

	public var bodyLegLMotor1:AngleMotor;
	public var bodyLegLMotor2:AngleMotor;
	public var bodyLegRMotor1:AngleMotor;
	public var bodyLegRMotor2:AngleMotor;
	public var bodyArmLMotor1:AngleMotor;
	public var bodyArmLMotor2:AngleMotor;
	public var bodyArmRMotor1:AngleMotor;
	public var bodyArmRMotor2:AngleMotor;
	public var legLegLMotor:AngleMotor;
	public var legLegRMotor:AngleMotor;
	public var armArmLMotor:AngleMotor;
	public var armArmRMotor:AngleMotor;
	public var bodyBodyMotor:AngleMotor;
	public var motors:Array<AngleMotor>;

	var offsetX:Float;
	var offsetY:Float;
	var offsetZ:Float;
	var world:World;

	public function new() {
	}

	public function init(w:World, x:Float, y:Float, z:Float, scale:Float = 1.0):Void {
		offsetX = x;
		offsetY = y;
		offsetZ = z;
		world = w;

		var headWidth:Float = scale * 0.2;
		var headHeight:Float = scale * 0.3;
		var headDepth:Float = scale * 0.2;

		var lowerBodyWidth:Float = scale * 0.5;
		var upperBodyWidth:Float = scale * 0.5;
		var lowerBodyHeight:Float = scale * 0.45;
		var upperBodyHeight:Float = scale * 0.45;
		var lowerBodyDepth:Float = scale * 0.3;
		var upperBodyDepth:Float = scale * 0.3;

		var lowerLegWidth:Float = scale * 0.2;
		var upperLegWidth:Float = scale * 0.2;
		var lowerLegHeight:Float = scale * 0.55;
		var upperLegHeight:Float = scale * 0.55;
		var lowerLegDepth:Float = scale * 0.2;
		var upperLegDepth:Float = scale * 0.2;
		var legSpace:Float = scale * 0.1;

		var lowerArmWidth:Float = scale * 0.5;
		var upperArmWidth:Float = scale * 0.5;
		var lowerArmHeight:Float = scale * 0.2;
		var upperArmHeight:Float = scale * 0.2;
		var lowerArmDepth:Float = scale * 0.2;
		var upperArmDepth:Float = scale * 0.2;

		var xplus:Vec3 = new Vec3(1, 0, 0);
		var xminus:Vec3 = new Vec3(-1, 0, 0);
		var yplus:Vec3 = new Vec3(0, 1, 0);
		var yminus:Vec3 = new Vec3(0, -1, 0);
		var zplus:Vec3 = new Vec3(0, 0, 1);
		var zminus:Vec3 = new Vec3(0, 0, -1);

		var pi:Float = Math.PI;
		var hpi:Float = Math.PI * 0.5;
		var tpi:Float = Math.PI * 2.0;

		head = box(0, lowerBodyHeight + upperBodyHeight + headHeight * 0.5, 0, headWidth, headHeight, headDepth);
		head.setAutoSleep(false);

		lowerBody = box(0, lowerBodyHeight * 0.5, 0, lowerBodyWidth, lowerBodyHeight, lowerBodyDepth);
		upperBody = box(0, lowerBodyHeight + upperBodyHeight * 0.5, 0, lowerBodyWidth, lowerBodyHeight, lowerBodyDepth);

		///*
		upperLegL = box((upperLegWidth + legSpace) * -0.5, -upperLegHeight * 0.5, 0, upperLegWidth, upperLegHeight, upperLegDepth);
		upperLegR = box((upperLegWidth + legSpace) * 0.5, -upperLegHeight * 0.5, 0, upperLegWidth, upperLegHeight, upperLegDepth);
		lowerLegL = box((lowerLegWidth + legSpace) * -0.5, -upperLegHeight - lowerLegHeight * 0.5, 0, lowerLegWidth, lowerLegHeight, lowerLegDepth);
		lowerLegR = box((lowerLegWidth + legSpace) * 0.5, -upperLegHeight - lowerLegHeight * 0.5, 0, lowerLegWidth, lowerLegHeight, lowerLegDepth);
		//*/

		/*
		upperLegL = capsuleY((upperLegWidth + legSpace) * -0.5, -upperLegHeight * 0.5, 0, upperLegWidth * 1.414, upperLegHeight);
		upperLegR = capsuleY((upperLegWidth + legSpace) * 0.5, -upperLegHeight * 0.5, 0, upperLegWidth * 1.414, upperLegHeight);
		lowerLegL = capsuleY((lowerLegWidth + legSpace) * -0.5, -upperLegHeight - lowerLegHeight * 0.5, 0, lowerLegWidth * 1.414, lowerLegHeight);
		lowerLegR = capsuleY((lowerLegWidth + legSpace) * 0.5, -upperLegHeight - lowerLegHeight * 0.5, 0, lowerLegWidth * 1.414, lowerLegHeight);
		*/

		///*
		upperArmL = box((upperBodyWidth + upperArmWidth) * -0.5, lowerBodyHeight + upperBodyHeight - upperArmHeight * 0.5, 0, upperArmWidth, upperArmHeight, upperArmDepth);
		upperArmR = box((upperBodyWidth + upperArmWidth) * 0.5, lowerBodyHeight + upperBodyHeight - upperArmHeight * 0.5, 0, upperArmWidth, upperArmHeight, upperArmDepth);
		lowerArmL = box((upperBodyWidth + upperArmWidth * 2 + lowerArmWidth) * -0.5, lowerBodyHeight + upperBodyHeight - upperArmHeight * 0.5, 0, lowerArmWidth, lowerArmHeight, lowerArmDepth);
		lowerArmR = box((upperBodyWidth + upperArmWidth * 2 + lowerArmWidth) * 0.5, lowerBodyHeight + upperBodyHeight - upperArmHeight * 0.5, 0, lowerArmWidth, lowerArmHeight, lowerArmDepth);
		//*/

		/*
		upperArmL = capsuleX((upperBodyWidth + upperArmWidth) * -0.5, lowerBodyHeight + upperBodyHeight - upperArmHeight * 0.5, 0, upperArmHeight * 1.414, upperArmWidth);
		upperArmR = capsuleX((upperBodyWidth + upperArmWidth) * 0.5, lowerBodyHeight + upperBodyHeight - upperArmHeight * 0.5, 0, upperArmHeight * 1.414, upperArmWidth);
		lowerArmL = capsuleX((upperBodyWidth + upperArmWidth * 2 + lowerArmWidth) * -0.5, lowerBodyHeight + upperBodyHeight - upperArmHeight * 0.5, 0, lowerArmHeight * 1.414, lowerArmWidth);
		lowerArmR = capsuleX((upperBodyWidth + upperArmWidth * 2 + lowerArmWidth) * 0.5, lowerBodyHeight + upperBodyHeight - upperArmHeight * 0.5, 0, lowerArmHeight * 1.414, lowerArmWidth);
		*/

		bodyLegL = universal(lowerBody, upperLegL, (lowerLegWidth + legSpace) * -0.5, 0, 0, xminus, zminus, -hpi, hpi, 0, hpi * 0.4);
		bodyLegR = universal(lowerBody, upperLegR, (lowerLegWidth + legSpace) * 0.5, 0, 0, xminus, zplus, -hpi, hpi, 0, hpi * 0.4);
		bodyArmL = universal(upperBody, upperArmL, upperBodyWidth * -0.5, lowerBodyHeight + upperBodyHeight - upperArmHeight * 0.5, 0, zminus, yplus, -hpi * 0.75, hpi * 0.25, 0, hpi);
		bodyArmR = universal(upperBody, upperArmR, upperBodyWidth * 0.5, lowerBodyHeight + upperBodyHeight - upperArmHeight * 0.5, 0, zplus, yminus, -hpi * 0.75, hpi * 0.25, 0, hpi);
		legLegL = revolute(upperLegL, lowerLegL, (lowerLegWidth + legSpace) * -0.5, -upperLegHeight, 0, xplus, 0, pi * 0.8);
		legLegR = revolute(upperLegR, lowerLegR, (lowerLegWidth + legSpace) * 0.5, -upperLegHeight, 0, xplus, 0, pi * 0.8);
		armArmL = revolute(upperArmL, lowerArmL, (upperBodyWidth * 2 + upperArmWidth) * -0.5, lowerBodyHeight + upperBodyHeight - upperArmHeight * 0.5, 0, yplus, 0, pi * 0.8);
		armArmR = revolute(upperArmR, lowerArmR, (upperBodyWidth * 2 + upperArmWidth) * 0.5, lowerBodyHeight + upperBodyHeight - upperArmHeight * 0.5, 0, yminus, 0, pi * 0.8);
		bodyBody = revolute(lowerBody, upperBody, 0, lowerBodyHeight, 0, xplus, -pi * 0.15, pi * 0.15);
		bodyHead = ragdoll(upperBody, head, 0, lowerBodyHeight + upperBodyHeight, 0, yplus, xplus, -0.01, 0.01, 0.01, 0.01);
		bodyHead.getSwingSpringDamper().setSpring(5, 1);

		bodyLegLMotor1 = new AngleMotor(bodyLegL, bodyLegL.getLimitMotor1(), bodyLegL.getAngle1, bodyLegL.getAxis1);
		bodyLegLMotor2 = new AngleMotor(bodyLegL, bodyLegL.getLimitMotor2(), bodyLegL.getAngle2, bodyLegL.getAxis2);
		bodyLegRMotor1 = new AngleMotor(bodyLegR, bodyLegR.getLimitMotor1(), bodyLegR.getAngle1, bodyLegR.getAxis1);
		bodyLegRMotor2 = new AngleMotor(bodyLegR, bodyLegR.getLimitMotor2(), bodyLegR.getAngle2, bodyLegR.getAxis2);
		bodyArmLMotor1 = new AngleMotor(bodyArmL, bodyArmL.getLimitMotor1(), bodyArmL.getAngle1, bodyArmL.getAxis1);
		bodyArmLMotor2 = new AngleMotor(bodyArmL, bodyArmL.getLimitMotor2(), bodyArmL.getAngle2, bodyArmL.getAxis2);
		bodyArmRMotor1 = new AngleMotor(bodyArmR, bodyArmR.getLimitMotor1(), bodyArmR.getAngle1, bodyArmR.getAxis1);
		bodyArmRMotor2 = new AngleMotor(bodyArmR, bodyArmR.getLimitMotor2(), bodyArmR.getAngle2, bodyArmR.getAxis2);
		armArmLMotor = new AngleMotor(armArmL, armArmL.getLimitMotor(), armArmL.getAngle, armArmL.getAxis1);
		armArmRMotor = new AngleMotor(armArmR, armArmR.getLimitMotor(), armArmR.getAngle, armArmR.getAxis1);
		legLegLMotor = new AngleMotor(legLegL, legLegL.getLimitMotor(), legLegL.getAngle, legLegL.getAxis1);
		legLegRMotor = new AngleMotor(legLegR, legLegR.getLimitMotor(), legLegR.getAngle, legLegR.getAxis1);
		bodyBodyMotor = new AngleMotor(bodyBody, bodyBody.getLimitMotor(), bodyBody.getAngle, bodyBody.getAxis1);

		motors = [
			bodyLegLMotor1,
			bodyLegLMotor2,
			bodyLegRMotor1,
			bodyLegRMotor2,
			bodyArmLMotor1,
			bodyArmLMotor2,
			bodyArmRMotor1,
			bodyArmRMotor2,
			armArmLMotor,
			armArmRMotor,
			legLegLMotor,
			legLegRMotor,
			bodyBodyMotor
		];
	}

	public function update():Void {
		for (motor in motors) {
			motor.update();
		}
	}

	@:extern
	inline function box(x:Float, y:Float, z:Float, w:Float, h:Float, d:Float):RigidBody {
		x += offsetX;
		y += offsetY;
		z += offsetZ;
		var rc:RigidBodyConfig = new RigidBodyConfig();
		rc.position.init(x, y, z);
		var rb:RigidBody = new RigidBody(rc);
		var sc:ShapeConfig = new ShapeConfig();
		sc.geometry = new BoxGeometry(new Vec3(w * 0.5, h * 0.5, d * 0.5));
		rb.addShape(new Shape(sc));
		world.addRigidBody(rb);
		return rb;
	}

	@:extern
	inline function capsuleY(x:Float, y:Float, z:Float, rad:Float, height:Float):RigidBody {
		x += offsetX;
		y += offsetY;
		z += offsetZ;
		var rc:RigidBodyConfig = new RigidBodyConfig();
		rc.position.init(x, y, z);
		var rb:RigidBody = new RigidBody(rc);
		var sc:ShapeConfig = new ShapeConfig();
		sc.geometry = new CapsuleGeometry(rad * 0.5, (height - rad * 0.5) * 0.5);
		rb.addShape(new Shape(sc));
		world.addRigidBody(rb);
		return rb;
	}

	@:extern
	inline function capsuleX(x:Float, y:Float, z:Float, rad:Float, width:Float):RigidBody {
		x += offsetX;
		y += offsetY;
		z += offsetZ;
		var rc:RigidBodyConfig = new RigidBodyConfig();
		rc.position.init(x, y, z);
		var rb:RigidBody = new RigidBody(rc);
		var sc:ShapeConfig = new ShapeConfig();
		sc.geometry = new CapsuleGeometry(rad * 0.5, (width - rad * 0.5) * 0.5);
		sc.rotation.appendRotationEq(MathUtil.PI * 0.5, 0, 0, 1);
		rb.addShape(new Shape(sc));
		world.addRigidBody(rb);
		return rb;
	}

	@:extern
	inline function spherical(b1:RigidBody, b2:RigidBody, x:Float, y:Float, z:Float):SphericalJoint {
		x += offsetX;
		y += offsetY;
		z += offsetZ;
		var jc = new SphericalJointConfig();
		jc.init(b1, b2, new Vec3(x, y, z));
		var j = new SphericalJoint(jc);
		world.addJoint(j);
		return j;
	}

	@:extern
	inline function universal(b1:RigidBody, b2:RigidBody, x:Float, y:Float, z:Float, axis1:Vec3, axis2:Vec3, lower1:Float = 1, upper1:Float = 0, lower2:Float = 1, upper2:Float = 0):UniversalJoint {
		x += offsetX;
		y += offsetY;
		z += offsetZ;
		var jc = new UniversalJointConfig();
		jc.init(b1, b2, new Vec3(x, y, z), axis1, axis2);
		jc.limitMotor1.setLimits(lower1, upper1);
		jc.limitMotor2.setLimits(lower2, upper2);
		jc.springDamper1.setSpring(20, 1);
		jc.springDamper2.setSpring(20, 1);
		var j = new UniversalJoint(jc);
		world.addJoint(j);
		return j;
	}

	@:extern
	inline function revolute(b1:RigidBody, b2:RigidBody, x:Float, y:Float, z:Float, axis:Vec3, lower:Float = 1, upper:Float = 0):RevoluteJoint {
		x += offsetX;
		y += offsetY;
		z += offsetZ;
		var jc = new RevoluteJointConfig();
		jc.init(b1, b2, new Vec3(x, y, z), axis).limitMotor.setLimits(lower, upper);
		jc.springDamper.setSpring(20, 1);
		var j = new RevoluteJoint(jc);
		world.addJoint(j);
		return j;
	}

	@:extern
	inline function ragdoll(b1:RigidBody, b2:RigidBody, x:Float, y:Float, z:Float, axisT:Vec3, axisS:Vec3, lowerT:Float = 1, upperT:Float = 0, swing1:Float = 3.1415926, swing2:Float = 3.1415926):RagdollJoint {
		x += offsetX;
		y += offsetY;
		z += offsetZ;
		var jc = new RagdollJointConfig();
		jc.init(b1, b2, new Vec3(x, y, z), axisT, axisS);
		jc.twistLimitMotor.setLimits(lowerT, upperT);
		jc.maxSwingAngle1 = swing1;
		jc.maxSwingAngle2 = swing2;
		var j = new RagdollJoint(jc);
		world.addJoint(j);
		return j;
	}

}
