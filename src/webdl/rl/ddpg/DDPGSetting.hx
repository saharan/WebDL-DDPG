package webdl.rl.ddpg;

/**
 * A setting of DDPG
 */
class DDPGSetting {
	/**
	 * If `true`, DDPG enters evaluation mode in which training and noise applying are disabled.
	 */
	public var evaluation:Bool = false;

	/**
	 * If `true`, Ornstein-Uhlenbeck noise is added to an action during exploration.
	 */
	public var noise:Bool = true;

	/**
	 * A parameter of Ornstein-Uhlenbeck noise: represents how fast the value regresses toward its mean.
	 */
	public var noiseTheta:Float = 0.15;

	/**
	 * A parameter of Ornstein-Uhlenbeck noise: represents the strength of the noise.
	 */
	public var noiseSigma:Float = 0.3;

	/**
	 * The number of hidden layers used.
	 */
	public var numHiddenLayers:Int = 2;

	/**
	 * The number of hidden units per hidden layer.
	 */
	public var numHiddenUnits:Int = 64;

	/**
	 * The size of the replay buffer to store experiences.
	 */
	public var replayBufferSize:Int = 500000;

	/**
	 * The number of steps before experience replay starts.
	 */
	public var startReplay:Int = 5000;

	/**
	 * Whether to use prioritized experience replay.
	 */
	public var prioritizedReplay:Bool = false;

	/**
	 * Whether to normalize obserbation.
	 */
	public var normalizeObserbation:Bool = true;

	/**
	 * The learning rate of the actor network.
	 */
	public var actorLearningRate:Float = 0.0001;

	/**
	 * The learning rate of the critic network.
	 */
	public var criticLearningRate:Float = 0.001;

	/**
	 * The weight decay applied to the actor network.
	 */
	public var actorWeightDecay:Float = 0.001;

	/**
	 * The weight decay applied to the critic network.
	 */
	public var criticWeightDecay:Float = 0.01;
	/**
	 * The discount rate.
	 */
	public var gamma:Float = 0.995;

	/**
	 * Update ratio of target networks.
	 */
	public var targetUpdateTau:Float = 0.01;

	/**
	 * The scale factor for rewards.
	 */
	public var rewardScaleFactor:Float = 0.01;

	/**
	 * The interval of learning and target network update.
	 */
	public var updateInterval:Int = 1;

	/**
	 * The size of the minibatches.
	 */
	public var minibatchSize:Int = 64;

	/**
	 * The maximum steps per episode.
	 */
	public var maxEpisodeLength:Int = 1000;

	/**
	 * The agent to be trained.
	 */
	public var agent:Agent = null;

	/**
	 * The dimension size of the observable state space.
	 */
	public var stateDim:Int = -1;

	/**
	 * The list of bounds of the action space, like `[[action1Min, action1Max], [action2Min, action2Max], ...]`.
	 */
	public var actionRanges:Array<Array<Float>> = null;

	public function new(agent:Agent, stateDim:Int, actionRanges:Array<Array<Float>>) {
		this.agent = agent;
		this.stateDim = stateDim;
		this.actionRanges = actionRanges;
	}

}
