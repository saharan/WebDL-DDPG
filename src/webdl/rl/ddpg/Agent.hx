package webdl.rl.ddpg;

/**
 * Interface of an agent to train.
 */
interface Agent {
	public function reset():State;
	public function step(action:Action):ActionResult;
}
