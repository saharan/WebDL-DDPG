package;

import haxe.Timer;
import haxe.io.Bytes;
import oimo.collision.geometry.BoxGeometry;
import oimo.common.MathUtil;
import oimo.common.Quat;
import oimo.common.Setting;
import oimo.common.Vec3;
import oimo.dynamics.Contact;
import oimo.dynamics.World;
import oimo.dynamics.callback.ContactCallback;
import oimo.dynamics.constraint.PositionCorrectionAlgorithm;
import oimo.dynamics.rigidbody.RigidBody;
import oimo.dynamics.rigidbody.RigidBodyConfig;
import oimo.dynamics.rigidbody.RigidBodyType;
import oimo.dynamics.rigidbody.Shape;
import oimo.dynamics.rigidbody.ShapeConfig;
import sys.io.File;
import webdl.core.WebDL;
import webdl.rl.ddpg.Action;
import webdl.rl.ddpg.ActionResult;
import webdl.rl.ddpg.Agent;
import webdl.rl.ddpg.DDPG;
import webdl.rl.ddpg.DDPGSetting;
import webdl.rl.ddpg.State;

/**
 * learn bipedal walking humanoid
 */
class RobotDDPGCudaTest {
	var humanoid:Humanoid;
	var ddpg:DDPG;
	var time:Float;
	var prevFC:Int;
	var end:Bool = false;

	public function new() {
		time = Timer.stamp();
		trace("setting up... t=" + time);
		WebDL.setBackend("gpu");

		var min:Float = 60;
		var hour:Float = 60 * min;
		var day:Float = 24 * hour;
		runUntil = time + day - hour * 2;

		prevFC = 0;

		view = false;
		noise = true;
		fast = true;

		try {
			bytes = File.getBytes("./save.dat");
			trace("save.dat found. loading rewards... t=" + Timer.stamp());

			var rewardData:String = File.getContent("./rewards.txt");
			trace("rewards.txt found. loading rewards... t=" + Timer.stamp());

			rewards = StringTools.trim(rewardData).split(",").map((s) -> {
				var f:Float = Std.parseFloat(s);
				if (!Math.isFinite(f)) {
					throw "rewards must be finite";
				}
				return f;
			});
			trace("rewards loaded. t=" + Timer.stamp());
		} catch (s:String) {
			trace(s);
			bytes = null;
			rewards = [];
		} catch (a:Any) {
			trace("failed to load save.dat. searching for load.txt... t=" + Timer.stamp());
			bytes = null;
			rewards = [];
		}

		if (bytes == null) {
			try {
				var saveData:String = File.getContent("./load.txt t=" + Timer.stamp());
				trace("load.txt found. loading parameters... t=" + Timer.stamp());

				var txt:String = StringTools.trim(saveData);
				var ds:Array<String> = txt.split("\n");
				if (ds.length != 2) throw "invalid format";
				var rew:Array<Float> = ds[0].split(",").map((s) -> {
					var f:Float = Std.parseFloat(s);
					if (!Math.isFinite(f)) {
						throw "rewards must be finite";
					}
					return f;
				});
				var wei:Array<Float> = ds[1].split(",").map((s) -> {
					var f:Float = Std.parseFloat(s);
					if (!Math.isFinite(f)) {
						throw "weights must be finite";
					}
					return f;
				});
				rewards = rew;
				data = wei;
				trace("data preloaded. t=" + Timer.stamp());
			} catch (s:String) {
				trace(s);
			} catch (a:Any) {
				trace("failed to load load.txt. starting with new data... t=" + Timer.stamp());
			}
		}

		initSim();

		while (!end) {
			loop();
		}
	}

	function initSim():Void {
		trace("initSim() t=" + Timer.stamp());
		humanoid = new Humanoid();
		var ds:DDPGSetting = new DDPGSetting(humanoid, Humanoid.STATE_DIM, [for (i in 0...Humanoid.ACTION_DIM) [-1, 1]]);
		ds.numHiddenUnits = 600;
		ds.numHiddenLayers = 3;
		ds.maxEpisodeLength = 2400;
		ds.minibatchSize = 64;
		ds.replayBufferSize = 3000000; // note this requires a HUGE memory
		ds.startReplay = 5000;
		ds.updateInterval = 2;
		ds.noiseTheta = 0.15;
		ds.noiseSigma = 0.2;
		ds.rewardScaleFactor = 0.01;
		ds.normalizeObserbation = true;
		ds.evaluation = view;

		ddpg = new DDPG(ds);

		if (view) {
			ddpg.episode = 0;
			ddpg.playingMode = view;
		}

		if (bytes != null) {
			trace("loading bytes... t=" + Timer.stamp());
			ddpg.importBytes(bytes);
			bytes = null; // GC
			ddpg.episode = rewards.length;
			trace("bytes loaded. episode = " + rewards.length + " t=" + Timer.stamp());
		} else if (data != null) {
			trace("loading weights... t=" + Timer.stamp());
			ddpg.importWeights(data);
			data = null; // GC
			ddpg.episode = rewards.length;
			ddpg.continueFromData = true;
			trace("weights loaded. episode = " + rewards.length + " t=" + Timer.stamp());
		}

		Setting.defaultFriction = 1.0;
		Setting.defaultRestitution = 0.0;
		Setting.defaultJointPositionCorrectionAlgorithm = PositionCorrectionAlgorithm.BAUMGARTE;
	}

	var fast:Bool = false;
	var noise:Bool = true;
	var view:Bool = true;
	var scoreTestCount:Int = 0;
	var scores:Array<Float> = [];
	var testedScore:Float = 0;

	var runUntil:Float;

	var bytes:Bytes = null;
	var data:Array<Float> = null;
	var rewards:Array<Float> = [];

	var fps:Int = 0;

	function save():Bool {
		var weights:Array<Float> = ddpg.exportWeights();
		for (w in weights) if (!Math.isFinite(w)) return false;
		data = weights;
		return true;
	}

	var frameCount:Int = 0;

	function loop():Void {
		frameCount++;
		var nextTime:Float = Timer.stamp();
		if (nextTime > time + 1) {
			time = nextTime;
			var newFC:Int = frameCount;
			fps = newFC - prevFC;
			prevFC = newFC;
		}
		ddpg.applyNoise = noise;
		if (view) {
			if (scoreTestCount > 0) {
				for (i in 0...10) {
					if (scoreTestCount > 0) {
						ddpg.step(false);
						if (ddpg.episodeTime == 0) {
							if (scores.length == 0) {
								// initial episode; ignore it
								scores.push(0);
							} else {
								scores.push(ddpg.totalRewardInLastEpisode);
								scoreTestCount--;
								if (scoreTestCount == 0) {
									var sum:Float = 0;
									for (s in scores) {
										sum += s;
									}
									sum /= (scores.length - 1);
									testedScore = sum;
								}
							}
						}
					}
				}
			} else {
				ddpg.step(false);
			}
		} else {
			//trace("------------------------step begin------------------------");
			var numSteps:Int = fast ? 4 : 1;
			for (i in 0...numSteps) {
				ddpg.step();
				if (ddpg.episodeTime == 0) {
					rewards.push(ddpg.totalRewardInLastEpisode);

					if (ddpg.episode == 1 || ddpg.episode % 100 == 0) {
						var st = Timer.stamp();
						var data:String = rewards.join(",") + "\n" + ddpg.exportWeights().join(",");
						File.saveContent("./data/" + (Std.int(ddpg.episode / 100)) + "_" + ddpg.episode + ".txt", data);
						var en = Timer.stamp();
						trace("weights saved. time = " + (en - st) + "sec");
					} else if (ddpg.episode == 1 || ddpg.episode % 10 == 0) {
						var st = Timer.stamp();
						File.saveContent("./rewards.txt", rewards.join(","));
						var en = Timer.stamp();
						trace("rewards.txt updated. time = " + (en - st) + "sec");
					}

					if (Timer.stamp() > runUntil) {
						saveBytes();
						end = true;
					}

				}
			}
			//trace("------------------------step end------------------------");
		}

		trace('ep: ${ddpg.episode}, elpsed: ${ddpg.episodeTime}, last total rew: ${ddpg.totalRewardInLastEpisode}, last rew: ${ddpg.lastReward}, last Q: ${ddpg.lastValueWithoutNoise}, FPS: $fps');
	}

	function saveBytes():Void {
		trace("saving data...");
		var st = Timer.stamp();
		File.saveBytes("./save.dat", ddpg.exportBytes());
		File.saveContent("./rewards.txt", rewards.join(","));
		var en = Timer.stamp();
		trace("data saved. time = " + (en - st) + "sec");
	}

	static function main():Void {
		new RobotDDPGCudaTest();
	}
}

private class HumanoidContactCallback extends ContactCallback {
	var onContact:RigidBody -> RigidBody -> Void;

	public function new(onContact:RigidBody -> RigidBody -> Void) {
		super();
		this.onContact = onContact;
	}

	override public function beginContact(c:Contact):Void {
		onContact(c.getShape1().getRigidBody(), c.getShape2().getRigidBody());
	}
}

private class Humanoid implements Agent {
	public static var STATE_DIM:Int = 210;
	public static var ACTION_DIM:Int = 13;
	public static var MAX_TORQUE:Float = 5;
	public static var MAX_SPEED:Float = Math.PI * 2;

	var w:World;
	public var torso:RigidBody;

	var ground:RigidBody;

	var ams:Array<AngleMotor>;

	var bodies:Array<RigidBody>;
	var impulses:Array<Float>;
	var gears:Array<Float>;
	var damping:Array<Float>;

	var end:Bool;
	var count:Int;
	var robot:Robot;

	public function new() {
	}

	public function reset():State {
		w = new World();
		w.setNumPositionIterations(5);
		w.setNumVelocityIterations(10);
		ground = box(0, -0.5, 1000, 20, 0.5, 1010, true);
		ground.getShapeList().setContactCallback(new HumanoidContactCallback(onContact));

		createRobot(0, 1.2, 0);
		{
			var b:RigidBody = w.getRigidBodyList();
			while (b != null) {
				b.setLinearVelocity(MathUtil.randVec3().scale(0.5));
				b.setAngularVelocity(MathUtil.randVec3().scale(0.5));
				b = b.getNext();
			}
		}
		torso.setLinearVelocity(MathUtil.randVec3().scale3Eq(0.01, 0, 0.01).add3Eq(0, 0, 0.5));
		var burnIn:Int = 7;
		robot.bodyLegLMotor1.limitMotor.setMotor(3, 1);
		robot.bodyLegRMotor1.limitMotor.setMotor(3, 1);
		robot.armArmLMotor.limitMotor.setMotor(2, 1);
		robot.armArmRMotor.limitMotor.setMotor(2, 1);
		robot.legLegLMotor.limitMotor.setMotor(2, 1);
		robot.legLegRMotor.limitMotor.setMotor(2, 1);
		for (i in 0...burnIn) {
			w.step(1 / 60);
			w.step(1 / 60);
		}
		{
			robot.armArmLMotor.limitMotor.setMotor(0, 0);
			robot.armArmRMotor.limitMotor.setMotor(0, 0);
			robot.legLegLMotor.limitMotor.setMotor(0, 0);
			robot.legLegRMotor.limitMotor.setMotor(0, 0);
			var b:RigidBody = w.getRigidBodyList();
			while (b != null) {
				b.setLinearVelocity(MathUtil.randVec3().scale(0.05));
				b.setAngularVelocity(MathUtil.randVec3().scale(0.05));
				b = b.getNext();
			}
			w.step(1 / 60);
		}

		end = false;
		count = 0;

		impulses = [for (i in 0...bodies.length * 6) 0.0];
		clearImpulses();

		return state();
	}

	function onContact(b1:RigidBody, b2:RigidBody):Void {
		if (b1 == ground || b2 == ground) {
			var other:RigidBody = b1 == ground ? b2 : b1;
			if (other != robot.lowerLegL && other != robot.lowerLegR) {
				end = true;
			}
		}
	}

	public function step(action:Action):ActionResult {
		var rot = torso.getRotation();
		var height = torso.getPosition().y;
		var up1 = rot.getCol(1).y;
		var up2 = robot.lowerBody.getRotation().getCol(1).y;
		if (up1 < 0.3 || up2 < 0.3 || height < 1.3) end = true;

		var i:Int = 0;
		var controlCost:Float = 0;

		for (am in ams) {
			var signedTorque:Float = MathUtil.clamp(action[i], -1, 1) * MAX_TORQUE;
			var absTorque:Float = signedTorque < 0 ? -signedTorque : signedTorque;
			var targetSpeed:Float = signedTorque / MAX_TORQUE * MAX_SPEED;
			var outRatio:Float = absTorque / MAX_TORQUE; // min: 0, max: 1

			var maxTorque:Float = MAX_TORQUE * gears[i] * 0.1;
			var minTorque:Float = MAX_TORQUE * damping[i] * 0.01;
			var interpolatedTorque:Float = minTorque + outRatio * (maxTorque - minTorque);

			am.limitMotor.setMotor(targetSpeed, interpolatedTorque);
			controlCost += outRatio * outRatio;

			i++;
		}

		controlCost *= 0.1;

		// step
		var dt:Float = 1 / 60;

		var prevpos:Float = cogZ();

		clearImpulses();

		var div:Int = 1;
		for (i in 0...div) {
			w.step(dt / div);
			accumulateImpulses();
		}

		scaleImpulses(1 / dt);

		var impulseCost:Float = 0;
		for (impulse in impulses) {
			impulseCost += impulse * 0.1;
		}
		if (impulseCost > 3) impulseCost = 3;

		var postpos:Float = cogZ();

		// rewards
		var aliveReward:Float = 3.0;
		var heightReward:Float = (height - 1.0) * 0.4;
		var velocityReward:Float = 4.0 * (postpos - prevpos) / dt;
		var totalReward:Float = aliveReward + velocityReward - controlCost - impulseCost;

		if (Math.isNaN(totalReward)) { // something's wrong!
			return {
				nextState: reset(),
				reward: 0,
				done: true
			};
		}

		return {
			nextState: state(),
			reward: end ? -50 : totalReward,
			done: end
		};
	}

	function cogZ():Float {
		var z:Float = 0;
		var denom:Float = 0;
		for (b in bodies) {
			z += b.getPosition().z * b.getMass();
			denom += b.getMass();
		}
		return z / denom;
	}

	function clearImpulses():Void {
		for (i in 0...impulses.length) {
			impulses[i] = 0;
		}
	}

	function accumulateImpulses():Void {
		var i:Int = 0;
		for (b in bodies) {
			var li:Vec3 = b.getLinearContactImpulse();
			var ai:Vec3 = b.getAngularContactImpulse();
			impulses[i++] += li.x;
			impulses[i++] += li.y;
			impulses[i++] += li.z;
			impulses[i++] += ai.x;
			impulses[i++] += ai.y;
			impulses[i++] += ai.z;
		}
	}

	function scaleImpulses(scale:Float):Void {
		for (i in 0...impulses.length) {
			impulses[i] *= scale;
		}
	}

	function state():State {
		var state:State = [];
		var torsoPos:Vec3 = torso.getPosition();

		// torso
		state.push(torsoPos.y);
		state = state.concat(bodyVels(torso));

		// the other parts
		for (b in bodies) {
			if (b != torso) {
				state = state.concat(bodyPos(b, torsoPos));
				state = state.concat(bodyVels(b));
			}
		}

		// joints
		for (am in ams) {
			state = state.concat(jointState(am));
		}

		// impulses
		state = state.concat(impulses);

		return state;
	}

	function bodyVels(b:RigidBody):Array<Float> {
		var lv:Vec3 = b.getLinearVelocity();
		var av:Vec3 = b.getAngularVelocity();
		return [lv.x, lv.y, lv.z, av.x, av.y, av.z];
	}

	function bodyPos(b:RigidBody, basePos:Vec3):Array<Float> {
		var q:Quat = b.getOrientation();
		var p:Vec3 = b.getPosition().subEq(basePos);
		return [p.x, p.y, p.z, q.x, q.y, q.z, q.w];
	}

	function jointState(am:AngleMotor):Array<Float> {
		return [am.angularVelocity, am.angle];
	}

	function createRobot(x:Float, y:Float, z:Float):Void {
		robot = new Robot();
		robot.init(w, x, y, z);
		bodies = [
			robot.upperBody, robot.lowerBody,
			robot.upperArmL, robot.lowerArmL, robot.upperLegL, robot.lowerLegL,
			robot.upperArmR, robot.lowerArmR, robot.upperLegR, robot.lowerLegR
		];
		ams = [
			robot.bodyLegLMotor1,
			robot.bodyLegLMotor2,
			robot.bodyLegRMotor1,
			robot.bodyLegRMotor2,
			robot.bodyArmLMotor1,
			robot.bodyArmLMotor2,
			robot.bodyArmRMotor1,
			robot.bodyArmRMotor2,
			robot.armArmLMotor,
			robot.armArmRMotor,
			robot.legLegLMotor,
			robot.legLegRMotor,
			robot.bodyBodyMotor
		];
		gears = [
			3.0, 3.0, 3.0, 3.0, // body-leg
			0.5, 0.5, 0.5, 0.5, // body-arm
			0.3, 0.3, // arm-arm
			2.0, 2.0, // leg-leg
			2.0 // body-body
		];
		damping = [
			3.0, 3.0, 3.0, 3.0, // body-leg
			1.0, 1.0, 1.0, 1.0, // body-arm
			1.0, 1.0, // arm-arm
			2.0, 2.0, // leg-leg
			3.0 // body-body
		];
		torso = robot.upperBody;
	}

	function box(x:Float, y:Float, z:Float, w:Float, h:Float, d:Float, wall:Bool):RigidBody {
		var rc = new RigidBodyConfig();
		rc.type = wall ? RigidBodyType.STATIC : RigidBodyType.DYNAMIC;
		rc.position.init(x, y, z);
		var sc = new ShapeConfig();
		sc.geometry = new BoxGeometry(new Vec3(w, h, d));
		var rb = new RigidBody(rc);
		rb.addShape(new Shape(sc));
		this.w.addRigidBody(rb);
		return rb;
	}
}
