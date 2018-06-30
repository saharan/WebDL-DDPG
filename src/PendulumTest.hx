package;
import haxe.Timer;
import js.Browser;
import pot.core.App;
import pot.graphics.Graphics;
import pot.input.KeyCode;
import webdl.core.RandUtil;
import webdl.core.Tensor;
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
using Lambda;
using webdl.core.WebDL;

/**
 * Pendulum test (like "Pendulum-v0")
 */
class PendulumTest extends App {
	var g:Graphics;

	override function setup():Void {
		WebDL.setBackend("gpu");

		pot.size(400, 300);
		pot.start();
		g = pot.graphics;
		g.init2D();

		initSim();
	}

	var ddpg:DDPG;
	var updates:Array<Tensor>;
	var agent:PendulumAgent;

	function initSim() {
		agent = new PendulumAgent(1 / 20);
		var ds:DDPGSetting = new DDPGSetting(agent, 3, [[-PendulumAgent.MAX_TORQUE, PendulumAgent.MAX_TORQUE]]);
		ds.numHiddenUnits = 64;
		ds.numHiddenLayers = 2;
		ds.maxEpisodeLength = 240;
		ds.updateInterval = 4;
		ddpg = new DDPG(ds);
		updates = ddpg.actor.network.updates.concat(ddpg.critic.network.updates);
	}

	var fastMode:Bool = false;

	var actorPlot:Array<Float> = null;
	var criticPlot:Array<Float> = null;

	var rewards:Array<Float> = [];

	override function loop():Void {
		if (input.mouse.dleft == 1) fastMode = !fastMode;
		if (input.mouse.dright == 1) Browser.console.log("[" + rewards.join(",") + "];");

		if (fastMode) {
			for (i in 0...20) {
				if (ddpg.step()) {
					updates.run();
				}
				if (ddpg.episodeTime == 0) rewards.push(ddpg.totalRewardInLastEpisode);
			}
		} else {
			var st:Float = Timer.stamp();
			if (ddpg.step()) {
				updates.run();
			}
			if (ddpg.episodeTime == 0) rewards.push(ddpg.totalRewardInLastEpisode);
			var en:Float = Timer.stamp();
		}

		g.beginScene();
		g.clear(0.9, 0.9, 0.9);
		g.pushMatrix();
		g.translate(pot.width * 0.5, pot.height * 0.5);
		g.color(0, 0, 0);
		g.rotate(-agent.theta);
		var len:Float = Math.min(pot.width, pot.height) * 0.3;
		var wid:Float = len * 0.05;
		g.rect(-wid, -wid - len, wid * 2, wid * 2 + len);
		g.popMatrix();

		//trace("DRAW BEGIN");
		///*
		plot2D(-Math.PI, Math.PI, -PendulumAgent.MAX_VELOCITY, PendulumAgent.MAX_VELOCITY, 32, 128, (angVels) -> {
			if (actorPlot == null || frameCount % 1 == 0) {
				var states:Array<State> = [for (angVel in angVels) [Math.cos(angVel[0]), Math.sin(angVel[0]), angVel[1]]];
				ddpg.actor.network.isTraining.set0D(input.keyboard[A].down ? 1 : 0);
				var actions:Array<Action> = ddpg.computeActions(states);
				ddpg.actor.network.isTraining.set0D(0);
				actorPlot = [for (action in actions) action[0] / PendulumAgent.MAX_TORQUE];
			}
			return actorPlot;
		}, [
			[Math.atan2(ddpg.lastState[1], ddpg.lastState[0]), ddpg.lastState[2]]
		]);
		//*/

		///*
		g.translate(pot.width - 128, 0);
		var currentState:State = ddpg.lastState;
		var currentAction:Action = ddpg.lastActionWithoutNoise;
		var currentValue:Float = ddpg.lastValueWithoutNoise;
		plot2D(-PendulumAgent.MAX_VELOCITY, PendulumAgent.MAX_VELOCITY, -PendulumAgent.MAX_TORQUE, PendulumAgent.MAX_TORQUE, 32, 128, (velTorques) -> {
			if (criticPlot == null || frameCount % 3 == 0) {
				var states:Array<State> = [for (velTorque in velTorques) [currentState[0], currentState[1], velTorque[0]]];
				var actions:Array<Action> = [for (velTorque in velTorques) [velTorque[1]]];
				ddpg.critic.network.isTraining.set0D(input.keyboard[A].down ? 1 : 0);
				var values:Array<Float> = ddpg.computeValues(states, actions);
				ddpg.critic.network.isTraining.set0D(0);
				criticPlot = [for (value in values) value - currentValue];
			}
			return criticPlot;
		}, [
			[currentState[2], currentAction[0]]
		]);
		//trace("DRAW END");
		//*/
		//var value:Float = ddpg.critic.network.getValue(ddpg.lastState, ddpg.lastActionWithoutNoise);
		//Browser.document.getElementById("text").innerHTML = 'Action with noise: ${ddpg.lastAction}<br>Action without noise: ${ddpg.lastActionWithoutNoise}<br>Value: $value<br>Episode: ${ddpg.episode}<br>Elapsed: ${ddpg.episodeTime}';

		Browser.document.getElementById("text").innerHTML = 'Action with noise: ${ddpg.lastAction}<br>Action without noise: ${ddpg.lastActionWithoutNoise}<br>Episode: ${ddpg.episode}<br>Value: ${ddpg.lastValueWithoutNoise}<br>Elapsed: ${ddpg.episodeTime}';

		g.endScene();
	}

	function plot2D(sx1:Float, sx2:Float, sy1:Float, sy2:Float, div:Int, size:Float, xysToVals:Array<Array<Float>> -> Array<Float>, points:Array<Array<Float>>):Void {
		var dx:Float = (sx2 - sx1) / div;
		var dy:Float = (sy2 - sy1) / div;
		var blockSize:Float = size / div;

		g.beginShape(Triangles);
		//var state:Array<Float> = agent.state();

		var xys:Array<Array<Float>> = [];
		{
			var y:Float = sy1;
			for (i in 0...div) {
				var x:Float = sx1;
				for (j in 0...div) {
					xys.push([x, y]);
					x += dx;
				}
				y += dy;
			}
		}
		var vals:Array<Float> = xysToVals(xys);
		{
			var c:Int = 0;
			var y:Float = sy1;
			for (i in 0...div) {
				var x:Float = sx1;
				for (j in 0...div) {
					var val:Float = vals[c++];
					g.color(0.5 + val * 0.5, 0, 0.5 - val * 0.5);

					var bx1:Float = blockSize * j;
					var by1:Float = blockSize * i;
					var bx2:Float = blockSize * j + blockSize;
					var by2:Float = blockSize * i + blockSize;
					g.vertex(bx1, by1);
					g.vertex(bx1, by2);
					g.vertex(bx2, by2);

					g.vertex(bx1, by1);
					g.vertex(bx2, by2);
					g.vertex(bx2, by1);
					x += dx;
				}
				y += dy;
			}
		}
		g.color(1, 1, 1);
		for (point in points) {
			var posx:Float = (point[0] - sx1) / (sx2 - sx1);
			var posy:Float = (point[1] - sy1) / (sy2 - sy1);
			posx *= size;
			posy *= size;
			g.vertex(posx - 2, posy - 2);
			g.vertex(posx - 2, posy + 2);
			g.vertex(posx + 2, posy + 2);

			g.vertex(posx - 2, posy - 2);
			g.vertex(posx + 2, posy + 2);
			g.vertex(posx + 2, posy - 2);
		}
		g.endShape();
	}

	// ---

	static function main():Void {
		new PendulumTest(cast Browser.document.getElementById("canvas"));
	}

	function setText(text:String):Void {
		Browser.document.getElementById("text").textContent = text;
	}

}

private class PendulumAgent implements Agent {
	public static inline var MAX_TORQUE:Float = 4;
	public static inline var MAX_VELOCITY:Float = 8;
	public var theta:Float;
	public var thetaDot:Float;
	var torque:Float;
	var dt:Float;

	public function new(dt:Float) {
		this.dt = dt;
	}

	public function reset():State {
		theta = RandUtil.uniform() * Math.PI;
		thetaDot = RandUtil.uniform() * 1;
		return [Math.cos(theta), Math.sin(theta), thetaDot];
	}

	public function step(action:Action):ActionResult {
		var g:Float = 15;
		torque = clamp(action[0], -MAX_TORQUE, MAX_TORQUE);
		thetaDot += (torque + g * Math.sin(theta)) * dt;
		thetaDot = clamp(thetaDot, -MAX_VELOCITY, MAX_VELOCITY);
		theta += thetaDot * dt;

		var pi:Float = Math.PI;
		var pi2:Float = pi * 2;
		theta = ((theta + pi) % pi2 + pi2) % pi2 - pi;

		var cost:Float = theta * theta + thetaDot * thetaDot * 0.001 + torque * torque * 0.001;
		cost = theta * theta;

		return {
			nextState: [Math.cos(theta), Math.sin(theta), thetaDot],
			reward: -cost,
			done: false
		};
	}

	inline function clamp(x:Float, min:Float, max:Float):Float {
		return x < min ? min : x > max ? max : x;
	}
}
