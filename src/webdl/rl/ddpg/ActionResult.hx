package webdl.rl.ddpg;

/**
 * The result of an action an agent took.
 */
typedef ActionResult = {
	/**
	 * The next state of the agent.
	 */
	nextState:State,
	/**
	 * The reward the agent earned.
	 */
	reward:Float,
	/**
	 * Whether the episode has ended.
	 */
	done:Bool
}
