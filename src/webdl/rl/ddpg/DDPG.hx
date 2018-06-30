package webdl.rl.ddpg;
import haxe.io.Bytes;
import haxe.io.BytesInput;
import haxe.io.BytesOutput;
import webdl.core.Tensor;
import webdl.core.WebDL;
import webdl.core.optimizer.AdamOptimizer;
import webdl.rl.ddpg.er.Experience;
import webdl.rl.ddpg.er.PrioritizedReplayBuffer;
import webdl.rl.ddpg.er.ReplayBuffer;
using webdl.core.WebDL;
using webdl.rl.ddpg.BytesTools;

/**
 * An implementation of Deep Deterministic Policy Gradient.
 */
class DDPG {
	public var actor:Actor;
	public var critic:Critic;
	public var agent:Agent;

	var minibatchSize:Int;
	var startReplay:Int;
	var replayBufferSize:Int;

	var gamma:Float;
	var tau:Float;
	var updateInterval:Int;

	var stateDim:Int;
	var actionDim:Int;

	var actorOptimizer:AdamOptimizer;
	var criticOptimizer:AdamOptimizer;

	public var lastState(default, null):State;
	public var lastAction(default, null):Action;
	public var lastReward(default, null):Float;
	public var lastValueWithoutNoise(default, null):Float;
	public var lastActionWithoutNoise(default, null):Action;
	public var totalReward(default, null):Float;
	public var totalRewardInLastEpisode(default, null):Float;
	public var totalRewards(default, null):Array<Float>;
	var noise:OrnsteinUhlenbeckNoise;
	var prioritizedExperienceReplay:Bool;
	var replayBuffer:ReplayBuffer;
	var prioritizedReplayBuffer:PrioritizedReplayBuffer;
	var rewardScaleFactor:Float;


	var maxEpisodeTime:Int;
	public var episodeTime(default, null):Int;
	public var episode:Int;
	public var numSteps:Int;

	public var applyNoise:Bool;
	public var normalizeObserbation:Bool;
	public var playingMode:Bool;
	public var continueFromData:Bool; // if true, do not recompute std and mean, and keep exploration noise moderate

	// target networks
	var initTargetNetworksOps:Array<Tensor>;
	var updateTargetNetworksOps:Array<Tensor>;

	// critic update
	var predictAction:Array<Tensor>;
	var predictQValue:Array<Tensor>;
	var criticMiniBatchReward:Tensor;
	var criticMiniBatchNotDone:Tensor;
	var criticMiniBatchTdError:Tensor; // for prioritized replay

	// actor update
	var prepareForActorUpdate:Array<Tensor>;

	// sampled experiences
	var experiences:Array<Experience>;

	var stateMeanStd:MeanStdTracker;
	var unnormalizedState:Tensor;
	var assignNormalizedStateToActor:Tensor;
	var assignNormalizedStateToActorTarget:Tensor;
	var assignNormalizedStateToCritic:Tensor;
	var assignNormalizedStateToCriticTarget:Tensor;

	public function new(ddpgSetting:DDPGSetting) {
		var actionRanges:Array<Array<Float>> = ddpgSetting.actionRanges;

		stateDim = ddpgSetting.stateDim;
		actionDim = actionRanges.length;

		var numHiddenLayers:Int = ddpgSetting.numHiddenLayers;
		var numHiddenUnits:Int = ddpgSetting.numHiddenUnits;

		actor = Actor.createActor(stateDim, actionDim, [for (i in 0...numHiddenLayers) numHiddenUnits], actionRanges);
		critic = Critic.createCritic(stateDim, actionDim, [for (i in 0...numHiddenLayers) numHiddenUnits]);
		agent = ddpgSetting.agent;
		maxEpisodeTime = ddpgSetting.maxEpisodeLength;
		noise = new OrnsteinUhlenbeckNoise(actionDim, 0, ddpgSetting.noiseTheta, ddpgSetting.noiseSigma);
		prioritizedExperienceReplay = ddpgSetting.prioritizedReplay;
		gamma = ddpgSetting.gamma;
		tau = ddpgSetting.targetUpdateTau;
		updateInterval = ddpgSetting.updateInterval;
		rewardScaleFactor = ddpgSetting.rewardScaleFactor;

		applyNoise = ddpgSetting.noise;
		playingMode = ddpgSetting.evaluation;
		normalizeObserbation = ddpgSetting.normalizeObserbation;

		replayBufferSize = ddpgSetting.replayBufferSize;
		minibatchSize = ddpgSetting.minibatchSize;
		startReplay = ddpgSetting.startReplay;

		var aNet:ActorNetwork = actor.network;
		var cNet:CriticNetwork = critic.network;
		var aTargetNet:ActorNetwork = actor.targetNetwork;
		var cTargetNet:CriticNetwork = critic.targetNetwork;

		if (actor.stateDim != critic.stateDim) throw "state dimensions between actor and critic mismatch";
		if (actor.actionDim != critic.actionDim) throw "action dimensions between actor and critic mismatch";
		actionDim = actor.actionDim;
		stateDim = actor.stateDim;

		// mean and std tracker
		stateMeanStd = new MeanStdTracker([1, stateDim]);

		// init placeholders
		var normalizedActionGradient:Tensor = WebDL.tensorOfShape([-1, actionDim]);
		var predictedQValue:Tensor = WebDL.tensorOfShape([-1]);
		unnormalizedState = WebDL.tensorOfShape([-1, stateDim]);
		criticMiniBatchReward = WebDL.tensorOfShape([-1]);
		criticMiniBatchNotDone = WebDL.tensorOfShape([-1]);
		criticMiniBatchTdError = WebDL.tensorOfShape([-1]);

		// assign states
		var normalizedState:Tensor = unnormalizedState.sub(stateMeanStd.mean).div(stateMeanStd.std);
		assignNormalizedStateToActor = aNet.inState.assign(normalizedState);
		assignNormalizedStateToCritic = cNet.inState.assign(normalizedState);
		assignNormalizedStateToActorTarget = aTargetNet.inState.assign(normalizedState);
		assignNormalizedStateToCriticTarget = cTargetNet.inState.assign(normalizedState);

		// --- actor part ---

		// compute and assign action gradient

		var unnormalizedActionGradient:Tensor = cNet.outValue.gradients([cNet.inAction])[0].mulConst(-1);
		normalizedActionGradient = unnormalizedActionGradient.mulConst(1 / minibatchSize);

		// compute policy gradients using deterministic policy gradient theorem
		var policyGrads:Array<Tensor> = aNet.outAction.gradients(aNet.trainables, normalizedActionGradient);
		actorOptimizer = new AdamOptimizer(aNet.trainables, policyGrads, ddpgSetting.actorLearningRate);
		actorOptimizer.setL2Decay(ddpgSetting.actorWeightDecay);

		// --- critic part ---

		// state --[action]--> state2 --[action2]--> ... -> ... -> ...
		//                       :                    :      :      :
		//                     reward                Q'(state2, action2)
		//
		// Q(state, action) -> reward + γQ'(state2, action2)

		// compute action2 using actor target network (and assign it to critic target network)
		predictAction = [cTargetNet.inAction.assign(aTargetNet.outAction)];
		// predict Q-values separately in order to stop differentiation (target networks must not be differentiated)
		predictQValue = [predictedQValue.assign(
			cTargetNet.outValue.mulConst(gamma).mul(criticMiniBatchNotDone).add(criticMiniBatchReward)
		)];
		// compute TD-error
		var tdError:Tensor = criticMiniBatchTdError.assign(predictedQValue.sub(cNet.outValue));
		var criticLoss:Tensor = tdError.square().reduceMean(0);
		var criticGradients:Array<Tensor> = criticLoss.gradients(cNet.trainables);
		criticOptimizer = new AdamOptimizer(cNet.trainables, criticGradients, ddpgSetting.criticLearningRate);
		criticOptimizer.setL2Decay(ddpgSetting.criticWeightDecay);

		// --- target network part ---

		// create updating process of the target networks
		var allTensorsToSave:Array<Tensor> = aNet.trainables.concat(cNet.tensorsToSave);
		var allTargetTensorsToSave:Array<Tensor> = aTargetNet.trainables.concat(cTargetNet.tensorsToSave);
		initTargetNetworksOps = [];
		updateTargetNetworksOps = [];
		for (i in 0...allTensorsToSave.length) {
			var t:Tensor = allTensorsToSave[i];
			var tTarget:Tensor = allTargetTensorsToSave[i];
			initTargetNetworksOps.push(tTarget.assign(t));
			updateTargetNetworksOps.push(tTarget.assign(WebDL.linComb(t, tTarget, tau, 1 - tau)));
		}

		if (prioritizedExperienceReplay) {
			prioritizedReplayBuffer = new PrioritizedReplayBuffer(replayBufferSize);
		} else {
			replayBuffer = new ReplayBuffer(replayBufferSize);
		}
		lastState = agent.reset();
		episode = 0;
		episodeTime = 0;
		totalReward = 0;
		totalRewardInLastEpisode = 0;
		totalRewards = [];
		numSteps = 0;
		experiences = null;
		continueFromData = false;

		// init target networks
		initTargetNetworksOps.run();
	}

	public function computeActions(states:Array<State>):Array<Action> {
		unnormalizedState.set2D(states);
		[assignNormalizedStateToActor].run();
		[actor.network.outAction].run();
		return actor.network.outAction.get2D();
	}

	public function computeValues(states:Array<State>, actions:Array<Action>):Array<Float> {
		unnormalizedState.set2D(states);
		[assignNormalizedStateToCritic].run();
		critic.network.inAction.set2D(actions);
		[critic.network.outValue].run();
		return critic.network.outValue.get1D();
	}

	/**
	 * Runs a step of DDPG. Returns whether optimizations are performed.
	 */
	public function step(disableEarlyReset:Bool = false):Bool {
		numSteps++;

		// --- compute noiseless action

		// set unnormalized state
		unnormalizedState.set2D([lastState]);

		// set normalized state to actor and critic network
		[assignNormalizedStateToActor, assignNormalizedStateToCritic].run();

		// run actor (compute action)
		[actor.network.outAction].run();
		// get action
		lastActionWithoutNoise = actor.network.outAction.get2D()[0];

		// --- compute Q-value of the noiseless action

		// set action
		critic.network.inAction.set2D([lastActionWithoutNoise]);
		// run critic (compute estimated Q-value)
		[critic.network.outValue].run();
		// get result of critic
		lastValueWithoutNoise = critic.network.outValue.get1D()[0];

		// --- apply exploration noise
		var bufferSize:Int = prioritizedExperienceReplay ? prioritizedReplayBuffer.size : replayBuffer.size;

		lastAction = lastActionWithoutNoise.copy();
		var explorationNoise:Array<Float> = noise.next();
		var noiseScale:Float = !continueFromData && bufferSize * 2 < startReplay ? 3 : 1; // scale the noise amplitude during the first half of the initial exploration
		if (applyNoise) {
			for (i in 0...actor.actionDim) {
				lastAction[i] += explorationNoise[i] * noiseScale;
			}
		}

		// --- step the agent

		var result:ActionResult = agent.step(lastAction);
		var experience:Experience = new Experience(lastState, lastAction, result.reward, result.nextState, result.done);
		if (prioritizedExperienceReplay) {
			prioritizedReplayBuffer.push(experience);
		} else {
			replayBuffer.push(experience);
		}
		bufferSize++;
		lastReward = result.reward;
		totalReward += result.reward;

		// prepare for the next step
		lastState = result.nextState;

		// update state mean and standard deviation durling the initial exploration
		if (!playingMode && !continueFromData && bufferSize < startReplay && normalizeObserbation) {
			stateMeanStd.update(result.nextState).run();
		}

		// --- episode termination check

		episodeTime++;
		if (!disableEarlyReset && result.done || episodeTime >= maxEpisodeTime) {
			lastState = agent.reset();
			episodeTime = 0;
			episode++;
			totalRewards.push(totalReward);
			totalRewardInLastEpisode = totalReward;
			totalReward = 0;
			// clean up for the next episode
			for (i in 0...1000) noise.next();
		}

		// --- learning phase

		if (!playingMode && bufferSize >= startReplay && numSteps % updateInterval == 0) {
			// update previous experiences' TD-error
			if (experiences != null && prioritizedExperienceReplay) {
				// update abs of TD-error
				var tdError:Array<Float> = criticMiniBatchTdError.get1D();
				for (i in 0...minibatchSize) {
					experiences[i].absTdError = tdError[i] < 0 ? -tdError[i] : tdError[i];
				}
				prioritizedReplayBuffer.update(experiences);
			}

			// sample new experiences
			experiences =
				if (prioritizedExperienceReplay) {
					prioritizedReplayBuffer.sample(minibatchSize);
				} else {
					replayBuffer.sample(minibatchSize);
				}
			;

			var state:Array<Array<Float>> = [for (e in experiences) e.state];
			var action:Array<Array<Float>> = [for (e in experiences) e.action];
			var state2:Array<Array<Float>> = [for (e in experiences) e.state2];
			var reward:Array<Float> = [for (e in experiences) e.reward * rewardScaleFactor];
			var notDone:Array<Float> = [for (e in experiences) e.done ? 0.0 : 1.0];

			critic.network.isTraining.set0D(1);
			critic.targetNetwork.isTraining.set0D(1);
			actor.network.isTraining.set0D(1);
			actor.targetNetwork.isTraining.set0D(1);

			// assign states
			unnormalizedState.set2D(state2);
			[assignNormalizedStateToActorTarget, assignNormalizedStateToCriticTarget].run();
			unnormalizedState.set2D(state);
			[assignNormalizedStateToCritic, assignNormalizedStateToActor].run();

			// --- critic update

			// predict action
			predictAction.run(); // this sets critic.targetNetwork.inAction

			// assign states
			critic.network.inAction.set2D(action);
			criticMiniBatchReward.set1D(reward);
			criticMiniBatchNotDone.set1D(notDone);

			// predict Q-values
			predictQValue.run();

			// optimize critic
			criticOptimizer.run();

			// --- actor update

			// set action to critic
			critic.network.inAction.set2D(action);

			// optimize actor
			actorOptimizer.run();

			// --- update target networks

			updateTargetNetworksOps.run();

			// clear training flag
			critic.network.isTraining.set0D(0);
			critic.targetNetwork.isTraining.set0D(0);
			actor.network.isTraining.set0D(0);
			actor.targetNetwork.isTraining.set0D(0);

			return true;
		}
		return false;
	}

	public function exportWeights():Array<Float> {
		var tensors:Array<Tensor> =
			actor.network.tensorsToSave
			.concat(critic.network.tensorsToSave)
			.concat([stateMeanStd.mean, stateMeanStd.std])
		;
		return tensors.exportElements();
	}

	public function importWeights(weights:Array<Float>):Void {
		var tensors:Array<Tensor> =
			actor.network.tensorsToSave
			.concat(critic.network.tensorsToSave)
			.concat([stateMeanStd.mean, stateMeanStd.std])
		;
		tensors.importElements(weights);
		initTargetNetworksOps.run();
	}

	public function exportBytes():Bytes {
		if (prioritizedExperienceReplay) {
			throw "exporting bytes in prioritized experience replay mode is not supported";
		}
		var bo:BytesOutput = new BytesOutput();

		// write header
		bo.writeString("DDPG");

		// write weights
		bo.writeFloatArray(exportWeights());

		// write actor & critic optimizer data
		bo.writeFloatArray(actorOptimizer.exportData());
		bo.writeFloatArray(criticOptimizer.exportData());

		// write experience replay buffer
		replayBuffer.writeBytes(bo);

		var res:Bytes = bo.getBytes();
		bo.close();
		return res;
	}

	public function importBytes(bytes:Bytes):Void {
		if (prioritizedExperienceReplay) {
			throw "importing bytes in prioritized experience replay mode is not supported";
		}
		var bi:BytesInput = new BytesInput(bytes);

		// read header
		if (bi.readString(4) != "DDPG") throw "invalid data";

		// read weights
		importWeights(bi.readFloatArray());

		// read actor & critic optimizer data
		actorOptimizer.importData(bi.readFloatArray());
		criticOptimizer.importData(bi.readFloatArray());

		// write experience replay buffer
		replayBuffer = ReplayBuffer.readBytes(bi);

		bi.close();
	}

}
