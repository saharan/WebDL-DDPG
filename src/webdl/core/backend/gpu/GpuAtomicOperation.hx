package webdl.core.backend.gpu;
import haxe.Timer;

/**
 * ...
 */
class GpuAtomicOperation {
	var name:String;
	var inputs:Array<Tensor>;
	var output:Tensor;
	public var shader(default, null):GpuShader;

	public function new(inputs:Array<Tensor>, output:Tensor, name:String, shader:GpuShader) {
		this.inputs = inputs;
		this.output = output;
		this.name = name;
		this.shader = shader;
	}

	public function bindUniforms():Void {
		shader.setDst(output);
		for (i in 0...inputs.length) {
			shader.setSrc(inputs[i], i);
		}
	}

	public function preDraw():Void {
	}

	public function postDraw():Void {
		var outputData:GpuTensorData = cast output.data;
		outputData.flip();
	}

}
