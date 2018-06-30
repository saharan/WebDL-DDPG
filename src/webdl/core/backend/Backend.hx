package webdl.core.backend;
import webdl.core.Operation;
import webdl.core.Tensor;
import webdl.core.nn.Activation;

/**
 * Core operations to be implemented.
 */
interface Backend {
	public function requestTensorData(size:Int):TensorData;
	public function disposeTensorData(data:TensorData):Void;

	// broadcastable binop
	public function add(a:Tensor, b:Tensor, dst:Tensor):Operation;
	public function addConst(a:Tensor, b:Float, dst:Tensor):Operation;
	public function sub(a:Tensor, b:Tensor, dst:Tensor):Operation;
	public function linComb(a:Tensor, b:Tensor, dst:Tensor, bScale:Float, bScale:Float):Operation;
	public function mul(a:Tensor, b:Tensor, dst:Tensor):Operation;
	public function mulConst(a:Tensor, b:Float, dst:Tensor):Operation;
	public function div(a:Tensor, b:Tensor, dst:Tensor):Operation;
	public function pow(a:Tensor, b:Tensor, dst:Tensor):Operation;
	public function powConst(a:Tensor, b:Float, dst:Tensor):Operation;

	// binop including contraction
	public function matMul(a:Tensor, b:Tensor, dst:Tensor):Operation;
	public function tensorDot(a:Tensor, b:Tensor, dst:Tensor, count:Int, axes:Array<Array<Int>>):Operation; // either count or axes will be passed

	// unop
	public function abs(a:Tensor, dst:Tensor):Operation;
	public function log(a:Tensor, dst:Tensor):Operation;
	public function exp(a:Tensor, dst:Tensor):Operation;

	// for neural network
	public function biasAdd(a:Tensor, b:Tensor, dst:Tensor):Operation;
	public function activation(a:Tensor, dst:Tensor, activation:Activation):Operation;

	// unop including contraction
	public function reduceSum(a:Tensor, dst:Tensor, axis:Int, keepDim:Bool):Operation;
	public function reduceMean(a:Tensor, dst:Tensor, axis:Int, keepDim:Bool):Operation;

	// split and merge
	public function split(a:Tensor, dsts:Array<Tensor>, axis:Int, sizes:Array<Int>):Operation;
	public function merge(as:Array<Tensor>, dst:Tensor, axis:Int):Operation;

	// differentiation
	public function gradients(y:Tensor, xs:Array<Tensor>, dsts:Array<Tensor>, gradY:Tensor):Operation;

	// assignment
	public function assign(a:Tensor, dummyDst:Tensor, dst:Tensor):Operation;

	// condition
	public function where(cond:Tensor, a:Tensor, b:Tensor, dst:Tensor):Operation;

	// adam update
	public function adamUpdate(count:Tensor, t:Tensor, g:Tensor, m:Tensor, v:Tensor, alpha:Tensor, beta1:Tensor, beta2:Tensor, epsilon:Tensor, l2Decay:Tensor, dummyDst:Tensor):Operation;
}
