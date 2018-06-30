package;

import haxe.Timer;
import js.Browser;
import js.html.AnchorElement;
import js.html.Blob;
import js.html.FileReader;
import js.html.SelectElement;
import js.html.URL;
import js.html.XMLHttpRequest;
import oimo.collision.geometry.BoxGeometry;
import oimo.common.MathUtil;
import oimo.common.Quat;
import oimo.common.Setting;
import oimo.common.Vec3;
import oimo.dynamics.Contact;
import oimo.dynamics.World;
import oimo.dynamics.callback.ContactCallback;
import oimo.dynamics.common.DebugDraw;
import oimo.dynamics.constraint.PositionCorrectionAlgorithm;
import oimo.dynamics.constraint.solver.ConstraintSolverType;
import oimo.dynamics.rigidbody.RigidBody;
import oimo.dynamics.rigidbody.RigidBodyConfig;
import oimo.dynamics.rigidbody.RigidBodyType;
import oimo.dynamics.rigidbody.Shape;
import oimo.dynamics.rigidbody.ShapeConfig;
import pot.core.App;
import pot.graphics.Graphics;
import webdl.core.RandUtil;
import webdl.core.WebDL;
import webdl.rl.ddpg.Action;
import webdl.rl.ddpg.ActionResult;
import webdl.rl.ddpg.Actor;
import webdl.rl.ddpg.Agent;
import webdl.rl.ddpg.Critic;
import webdl.rl.ddpg.DDPG;
import webdl.rl.ddpg.DDPGSetting;
import webdl.rl.ddpg.OrnsteinUhlenbeckNoise;
import webdl.rl.ddpg.State;

/**
 * Play a learned DDPG model on a browser
 */
class RobotDDPGTest extends App {
	var g:Graphics;
	var humanoid:Humanoid;
	var ddpg:DDPG;
	var time:Float;
	var prevFC:Int;

	override function setup():Void {
		pot.size(640, 480);
		g = pot.graphics;
		g.init3D();

		WebDL.setBackend("gpu");
		time = Timer.stamp();
		prevFC = 0;

		var loadText:String -> Void = (s) -> {
			try {
				var txt:String = StringTools.trim(s);
				var ds:Array<String> = txt.split("\n");
				if (ds.length != 2) throw "invalid number of lines";
				var rew:Array<Float> = ds[0].split(",").map((s) -> {
					var f:Float = Std.parseFloat(s);
					if (!Math.isFinite(f)) {
						throw "reward must be finite";
					}
					return f;
				});
				var wei:Array<Float> = ds[1].split(",").map((s) -> {
					var f:Float = Std.parseFloat(s);
					if (!Math.isFinite(f)) {
						throw "weight must be finite";
					}
					return f;
				});
				rewards = rew;
				data = wei;
				initSim();
			} catch (e:Any) {
				Browser.alert("load failed");
			}
		};

		if (Browser.document.getElementById("load") != null) {
			Browser.document.getElementById("load").addEventListener("change", (e) -> {
				var file:Array<Blob> = e.target.files;
				var reader:FileReader = new FileReader();
				reader.readAsText(file[0]);
				reader.onload = (e) -> {
					loadText(reader.result);
				};
			});
		}

		if (Browser.document.getElementById("save") != null) {
			Browser.document.getElementById("save").addEventListener("click", (e) -> {
				var data:String = rewards.join(",") + "\n" + ddpg.exportWeights().join(",");
				var blob:Blob = new Blob([data], { "type": "text/plain"});
				var a:AnchorElement = Browser.document.createAnchorElement();
				a.href = URL.createObjectURL(blob);
				a.target = "_blank";
				a.setAttribute("download", "data.txt");
				a.click();
			});
			view = false;
			noise = true;
		}

		if (Browser.document.getElementById("test") != null) {
			Browser.document.getElementById("test").addEventListener("click", function(e):Void {
				if (!view) {
					Browser.alert("score test can only be run in view mode");
					return;
				}
				scoreTestCount = 100;
				scores = [];
				noise = false;
			});
			view = true;
			noise = false;
		}

		if (Browser.document.getElementById("preset") != null) {
			var select:SelectElement = cast Browser.document.getElementById("preset");
			select.addEventListener("change", function(e):Void {
				if (select.selectedIndex <= 0) return;
				var xhr:XMLHttpRequest = new XMLHttpRequest();
				xhr.addEventListener("load", (e) -> {
					trace("loaded");
					loadText(xhr.responseText);
				});
				var val:String = untyped __js__('{0}[{1}].value', select.options, select.selectedIndex);
				xhr.open("get", "./" + val);
				xhr.send(null);
			});
			view = true;
			noise = false;
		}

		initSim();
		pot.start();
	}

	function initSim():Void {
		humanoid = new Humanoid(g);
		var ds:DDPGSetting = new DDPGSetting(humanoid, Humanoid.STATE_DIM, [for (i in 0...Humanoid.ACTION_DIM) [-1, 1]]);
		ds.numHiddenUnits = 600;
		ds.numHiddenLayers = 3;
		ds.maxEpisodeLength = 2400;
		ds.minibatchSize = 64;
		ds.replayBufferSize = 30000;
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

		if (data != null) {
			ddpg.importWeights(data);
			ddpg.episode = rewards.length;
			ddpg.continueFromData = true;
			trace("weights loaded. episode = " + rewards.length);
		}

		Setting.defaultFriction = 1.0;
		Setting.defaultRestitution = 0.0;
		Setting.defaultJointConstraintSolverType = ConstraintSolverType.DIRECT;
		Setting.defaultJointPositionCorrectionAlgorithm = PositionCorrectionAlgorithm.NGS;
	}

	var fast:Bool = false;
	var noise:Bool = true;
	var view:Bool = true;
	var scoreTestCount:Int = 0;
	var scores:Array<Float> = [];
	var testedScore:Float = 0;

	var data:Array<Float> = null;
	var rewards:Array<Float> = [];

	var fps:Int = 0;

	function save():Bool {
		var weights:Array<Float> = ddpg.exportWeights();
		for (w in weights) if (!Math.isFinite(w)) return false;
		data = weights;
		return true;
	}

	override function loop():Void {
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
				if (input.mouse.dleft == 1) {
					humanoid.torso.addLinearVelocity(new Vec3(MathUtil.randIn(-2, 2), 10, MathUtil.randIn(-2, 2)));
				}
				if (input.mouse.dright == 1) noise = !noise;
				ddpg.step(false);
			}
		} else {
			if (input.mouse.dleft == 1) {
				fast = !fast;
			}
			/*
			if (input.mouse.dright == 1) {
				var data:String = rewards.join(",") + "\n" + ddpg.exportWeights().join(",");
				Browser.console.log(data);
			}
			*/

			if (input.mouse.dright == 1) {
				noise = !noise;
			}

			//trace("------------------------step begin------------------------");
			var numSteps:Int = fast ? 4 : 1;
			for (i in 0...numSteps) {
				ddpg.step();
				if (ddpg.episodeTime == 0) {
					rewards.push(ddpg.totalRewardInLastEpisode);
					if (ddpg.episode % 100 == 0) {
						if (!save()) { // NaN hazard
							trace("NaN!!!");
							initSim();
							return;
						}
					}
				}
			}
			//trace("------------------------step end------------------------");
		}

		g.beginScene();
		g.clear(0, 0, 0);
		var pos:Vec3 = humanoid.torso.getPosition();
		g.camera(-6 + pos.x, 5, pos.z, pos.x, 1, pos.z, 0, 1, 0);
		g.lights();

		g.beginShape(Triangles);
		humanoid.draw();
		g.endShape();

		g.endScene();
		Browser.document.getElementById("text").innerHTML = 'Episode: ${ddpg.episode}<br>Elapsed: ${ddpg.episodeTime}<br>Total reward in the last episode: ${ddpg.totalRewardInLastEpisode}<br>the last reward: ${ddpg.lastReward}<br>the last Q-value: ${ddpg.lastValueWithoutNoise}<br>noise: ${noise ? "ON" : "OFF"}<br>FPS: $fps<br>average score: ${scoreTestCount > 0 ? "testing... (" + scoreTestCount + " left)" : testedScore == 0 ? "---" : "" + Std.int(testedScore * 100) / 100}';
	}

	static function main():Void {
		new RobotDDPGTest(cast Browser.document.getElementById("canvas"));
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
	var g:Graphics;
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

	public function new(g:Graphics) {
		this.g = g;
	}

	public function reset():State {
		w = new World();
		w.setNumPositionIterations(5);
		w.setNumVelocityIterations(10);
		w.setDebugDraw(new Renderer(g));

		for (i in -10...100) {
			box(-5, 0.2, i, 0.2, 0.2, 0.2, true).getShapeList().setCollisionMask(0);
			box(5, 0.2, i, 0.2, 0.2, 0.2, true).getShapeList().setCollisionMask(0);
		}

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

	public function draw():Void {
		w.debugDraw();
	}
}

private class Renderer extends DebugDraw {
	var g:Graphics;

	public function new(g:Graphics) {
		super();
		this.g = g;
	}

	override public function triangle(v1:Vec3, v2:Vec3, v3:Vec3, n1:Vec3, n2:Vec3, n3:Vec3, color:Vec3):Void {
		g.color(color.x, color.y, color.z);
		g.normal(n1.x, n1.y, n1.z);
		g.vertex(v1.x, v1.y, v1.z);
		g.normal(n2.x, n2.y, n2.z);
		g.vertex(v2.x, v2.y, v2.z);
		g.normal(n3.x, n3.y, n3.z);
		g.vertex(v3.x, v3.y, v3.z);
		var floorY:Float = 0.0;
		if (v1.y > floorY || v2.y > floorY || v3.y > floorY) {
			g.color(0.2, 0.2, 0.2);
			g.normal(0, 0, 0);
			g.vertex(v1.x, floorY + 0.005, v1.z);
			g.vertex(v2.x, floorY + 0.005, v2.z);
			g.vertex(v3.x, floorY + 0.005, v3.z);
		}
	}
}
