package webdl.core.backend.cuda;
import pycuda.Function;
import pycuda.PyCuda;
import pycuda.SourceModule;
import python.Tuple;
import webdl.core.Tensor;

/**
 * ...
 */
class CudaAtomicOperation {
	var data:Array<Tensor>;
	var name:String;
	var func:Function;
	var prepared:Bool;

	public function new(data:Array<Tensor>, name:String, source:String) {
		this.data = data; // use data[0] as destination
		this.name = name;
		var kernelSource:String = '
			#define val(i,idx4) value##i[idx4to1(idx4,shape##i)]
			#define dif(i,idx4) diff##i[idx4to1(idx4,shape##i)]
			#define def_idx4(name) int name[4];idx1to4(idx1,shape0,name)

			__device__ void idx1to4(int idx1, int *shape, int *idx4) {
				idx4[0] = idx1 % shape[0]; idx1 /= shape[0];
				idx4[1] = idx1 % shape[1]; idx1 /= shape[1];
				idx4[2] = idx1 % shape[2]; idx1 /= shape[2];
				idx4[3] = idx1;
			}

			__device__ int idx4to1(int *idx4, int *shape) {
				int               idx1  = idx4[3] % shape[3];
				idx1 *= shape[2]; idx1 += idx4[2] % shape[2];
				idx1 *= shape[1]; idx1 += idx4[1] % shape[1];
				idx1 *= shape[0]; idx1 += idx4[0] % shape[0];
				return idx1;
			}

			__device__ void ins(int *idx4, int axis, int value) {
				for (int i = axis + 1; i < 4; i++) {
					idx4[i] = idx4[i - 1];
				}
				idx4[axis] = value;
			}

			__device__ void del(int *idx4, int axis) {
				for (int i = axis + 1; i < 4; i++) {
					idx4[i - 1] = idx4[i];
				}
				idx4[3] = 0;
			}

			__device__ float tanhGrad(float x) {
				float tanhx = tanhf(x);
				return 1 - tanhx * tanhx;
			}

			__device__ float sigmoid(float x) {
				return 1 / (1 + expf(-x));
			}

			__device__ float sigmoidGrad(float x) {
				float y = sigmoid(x);
				return y * (1 - y);
			}

			__device__ float relu(float x) {
				return x < 0 ? 0 : x;
			}

			__device__ float reluGrad(float x) {
				return x < 0 ? 0 : 1;
			}

			__global__ void run(${createKernelArgs(data.length)}) {
				int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
				if (idx1 >= shape0[0] * shape0[1] * shape0[2] * shape0[3]) return;
				int idx4[4];
				idx1to4(idx1, shape0, idx4);
				$source
			}
		';
		func = new SourceModule(kernelSource).getFunction("run");
		prepared = false;
	}

	function createKernelArgs(numData:Int):String {
		var src:Array<String> = [for (i in 0...numData) 'float *value$i, float *diff$i, int *shape$i'];
		return src.join(", ");
	}

	public function run():Void {
		var args:Array<Any> = [];
		for (t in data) {
			var cdata:CudaTensorData = cast t.data;
			cdata.shape.uploadDataInt(
				switch (t.rank) {
				case 0:
					[1, 1, 1, 1];
				case 1:
					[t.actualShape[0], 1, 1, 1];
				case 2:
					[t.actualShape[1], t.actualShape[0], 1, 1];
				case 3:
					[t.actualShape[2], t.actualShape[1], t.actualShape[0], 1];
				case _:
					[t.actualShape[3], t.actualShape[2], t.actualShape[1], t.actualShape[0]];
				}
			);
			args.push(cdata.val.device);
			args.push(cdata.dif.device);
			args.push(cdata.shape.device);
		}

		var numThreads:Int = data[0].actualSize;
		var blockSize:Int = 256;

		if (!prepared) {
			func.prepare(args.map((a) -> "P").join(""));
			prepared = true;
		}

		func.preparedCall(new Tuple([Math.ceil(numThreads / blockSize), 1, 1]), new Tuple([blockSize, 1, 1]), args);

		//trace("running: " + name);
		//trace("result(val): " + data[0].print());
		//trace("result(dif): " + data[0].printDiff());
	}
}
