package webdl.rl.ddpg.er;

/**
 * An experience to replay. Consists of a state, an action, reward and the state after the action.
 */
class Experience {
	public var state(default, null):State;
	public var action(default, null):Action;
	public var reward(default, null):Float;
	public var state2(default, null):State;
	public var done(default, null):Bool;
	public var absTdError:Float; // for prioritized replay
	public var p:Float;          // for prioritized replay
	public var index:Int; // index in replay buffer

	/**
	 * Experience: performed `action` at `state`, then got `reward` and turned into `state2`.
	 * `done` is `true` if the episode has ended.
	 */
	public function new(state:State, action:Action, reward:Float, state2:State, done:Bool) {
		this.state = state;
		this.action = action;
		this.reward = reward;
		this.state2 = state2;
		this.done = done;
		absTdError = 10000; // set high TD-error in order to visit this experience as soon as possible
		p = 0;
		index = -1;
	}
}
