package webdl.test;
import haxe.Timer;
import webdl.core.Tensor;
import webdl.core.WebDL;
import webdl.core.layer.BatchNormalizationLayer;
import webdl.core.layer.DenseLayer;
import webdl.core.layer.Layer;
import webdl.core.optimizer.AdamOptimizer;
using webdl.core.WebDL;

/**
 * unit test of backends
 */
class WebDLTest {
	static inline var EPS:Float = 1e-3;
	var tests:Array<Test>;

	public function new() {
		tests = [];

		WebDL.setBackend("gpu");
		testBinOp();
		testUnOp();
		testSplitMerge();
		testBroadCasting();
		testAssign();
		testAdam();
		//testBatchNorm(false);
		//testBatchNorm(true);

		run();
	}

	function testBinOp():Void {
		add("add", () -> {
			var f = WebDL.add;
			equalBinOp(f,
				[[0, 1], [2, 3], [4, 5]],
				[[6, 7], [8, 9], [10, 11]],
				[[6, 8], [10, 12], [14, 16]],
				[[1, 1], [1, 1], [1, 1]],
				[[1, 1], [1, 1], [1, 1]]
			);
			equalBinOp(f,
				123,
				456,
				579,
				1,
				1
			);
		});
		add("sub", () -> {
			var f = WebDL.sub;
			equalBinOp(f,
				[[0, 1], [2, 3], [4, 5]],
				[[6, 7], [8, 9], [10, 11]],
				[[-6, -6], [-6, -6], [-6, -6]],
				[[1, 1], [1, 1], [1, 1]],
				[[-1, -1], [-1, -1], [-1, -1]]
			);
			equalBinOp(f,
				123,
				456,
				-333,
				1,
				-1
			);
		});
		add("mul", () -> {
			var f = WebDL.mul;
			equalBinOp(f,
				[[0, 1], [2, 3], [4, 5]],
				[[6, 7], [8, 9], [10, 11]],
				[[0, 7], [16, 27], [40, 55]],
				[[6, 7], [8, 9], [10, 11]],
				[[0, 1], [2, 3], [4, 5]]
			);
			equalBinOp(f,
				123,
				456,
				56088,
				456,
				123
			);
		});
		add("div", () -> {
			var f = WebDL.div;
			equalBinOp(f,
				[[0, 1], [2, 3], [4, 5]],
				[[6, 7], [8, 9], [10, 11]],
				[[0 / 6, 1 / 7], [2 / 8, 3 / 9], [4 / 10, 5 / 11]],
				[[1 / 6, 1 / 7], [1 / 8, 1 / 9], [1 / 10, 1 / 11]],
				[[-0 / 36, -1 / 49], [-2 / 64, -3 / 81], [-4 / 100, -5 / 121]]
			);
			equalBinOp(f,
				123,
				456,
				123 / 456,
				1 / 456,
				-123 / (456 * 456)
			);
		});
		add("linComb", () -> {
			var f = (a, b) -> WebDL.linComb(a, b, 2, 3);
			equalBinOp(f,
				[[0, 1], [2, 3], [4, 5]],
				[[6, 7], [8, 9], [10, 11]],
				[[18, 23], [28, 33], [38, 43]],
				[[2, 2], [2, 2], [2, 2]],
				[[3, 3], [3, 3], [3, 3]]
			);
			equalBinOp(f,
				123,
				456,
				1614,
				2,
				3
			);
		});
		add("pow", () -> {
			var f = WebDL.pow;
			equalBinOp(f,
				[[1, 2], [3, 4], [5, 6]],
				[[6, 5], [4, 3], [2, 1]],
				[[1, 32], [81, 64], [25, 6]],
				[[6, 80], [108, 48], [10, 1]],
				[[0, Math.log(2) * 32], [Math.log(3) * 81, Math.log(4) * 64], [Math.log(5) * 25, Math.log(6) * 6]]
			);
			equalBinOp(f,
				5,
				3,
				125,
				3 * 25,
				Math.log(5) * 125
			);
		});
		add("matMul", () -> {
			var f = WebDL.matMul;
			equalBinOp(f,
				[[[1, 2], [3, 4], [5, 6]], [[5, 6], [3, 4], [1, 2]]],
				[[[7, 8, 9], [10, 11, 12]], [[10, 11, 12], [7, 8, 9]]],
				[[[27, 30, 33], [61, 68, 75], [95, 106, 117]], [[92, 103, 114], [58, 65, 72], [24, 27, 30]]],
				[[[24, 33], [24, 33], [24, 33]], [[33, 24], [33, 24], [33, 24]]],
				[[[9, 9, 9], [12, 12, 12]], [[9, 9, 9], [12, 12, 12]]]
			);
			equalBinOp(f,
				[[123]],
				[[456]],
				[[56088]],
				[[456]],
				[[123]]
			);
		});
		add("tensorDot", () -> {
			var f0 = (a, b) -> WebDL.tensorDot(a, b, 0);
			var f1 = (a, b) -> WebDL.tensorDot(a, b, 1);
			var f2 = (a, b) -> WebDL.tensorDot(a, b, 2);
			equalBinOp(f0,
				[0, 1, 2],
				[[3, 4], [5, 6]],
				[[[0, 0], [0, 0]], [[3, 4], [5, 6]], [[6, 8], [10, 12]]],
				[18, 18, 18],
				[[3, 3], [3, 3]]
			);
			equalBinOp(f1,
				[0, 1, 2],
				[[3, 4], [5, 6], [7, 8]],
				[19, 22],
				[7, 11, 15],
				[[0, 0], [1, 1], [2, 2]]
			);
			equalBinOp(f2,
				[[0, 1, 2], [3, 4, 5]],
				[[6, 7, 8], [9, 10, 11]],
				145,
				[[6, 7, 8], [9, 10, 11]],
				[[0, 1, 2], [3, 4, 5]]
			);
		});
		add("biasAdd", () -> {
			var f = WebDL.biasAdd;
			equalBinOp(f,
				[[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]],
				[100, 200],
				[[[100, 201], [102, 203], [104, 205]], [[106, 207], [108, 209], [110, 211]]],
				[[[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1]]],
				[6, 6]
			);
			equalBinOp(f,
				[123],
				[456],
				[579],
				[1],
				[1]
			);
		});
		add("where", () -> {
			var f = (a, b) -> WebDL.where([[0, 0], [0, 1], [1, 0], [1, 1]].tensorOfValue(), a, b);
			equalBinOp(f,
				[[0, 1], [2, 3], [4, 5], [6, 7]],
				[[8, 9], [10, 11], [12, 13], [14, 15]],
				[[8, 9], [10, 3], [4, 13], [6, 7]],
				[[0, 0], [0, 1], [1, 0], [1, 1]],
				[[1, 1], [1, 0], [0, 1], [0, 0]]
			);
			equalBinOp(f,
				123,
				456,
				456,
				0,
				1
			);
		});
	}

	function testUnOp():Void {
		add("addConst", () -> {
			var f = (a) -> WebDL.addConst(a, 3);
			equalUnOp(f,
				[[0, 1], [2, 3], [4, 5]],
				[[3, 4], [5, 6], [7, 8]],
				[[1, 1], [1, 1], [1, 1]]
			);
			equalUnOp(f,
				123,
				126,
				1
			);
		});
		add("subConst", () -> {
			var f = (a) -> WebDL.subConst(a, 3);
			equalUnOp(f,
				[[0, 1], [2, 3], [4, 5]],
				[[-3, -2], [-1, 0], [1, 2]],
				[[1, 1], [1, 1], [1, 1]]
			);
			equalUnOp(f,
				123,
				120,
				1
			);
		});
		add("mulConst", () -> {
			var f = (a) -> WebDL.mulConst(a, 3);
			equalUnOp(f,
				[[0, 1], [2, 3], [4, 5]],
				[[0, 3], [6, 9], [12, 15]],
				[[3, 3], [3, 3], [3, 3]]
			);
			equalUnOp(f,
				123,
				369,
				3
			);
		});
		add("powConst", () -> {
			var f = (a) -> WebDL.powConst(a, 3);
			equalUnOp(f,
				[[0, 1], [2, 3], [4, 5]],
				[[0, 1], [8, 27], [64, 125]],
				[[0, 3], [12, 27], [48, 75]]
			);
			equalUnOp(f,
				6,
				216,
				108
			);
		});
		add("abs", () -> {
			var f = WebDL.abs;
			equalUnOp(f,
				[[-3, -2], [-1, 1], [2, 3]],
				[[3, 2], [1, 1], [2, 3]],
				[[-1, -1], [-1, 1], [1, 1]]
			);
			equalUnOp(f,
				-123,
				123,
				-1
			);
		});
		add("log", () -> {
			var f = WebDL.log;
			equalUnOp(f,
				[[1, 2], [3, 4], [5, 6]],
				[[0, Math.log(2)], [Math.log(3), Math.log(4)], [Math.log(5), Math.log(6)]],
				[[1, 1 / 2], [1 / 3, 1 / 4], [1 / 5, 1 / 6]]
			);
			equalUnOp(f,
				123,
				Math.log(123),
				1 / 123
			);
		});
		add("exp", () -> {
			var f = WebDL.exp;
			equalUnOp(f,
				[[0, 1], [2, 3], [4, 5]],
				[[1, Math.exp(1)], [Math.exp(2), Math.exp(3)], [Math.exp(4), Math.exp(5)]],
				[[1, Math.exp(1)], [Math.exp(2), Math.exp(3)], [Math.exp(4), Math.exp(5)]]
			);
			equalUnOp(f,
				6,
				Math.exp(6),
				Math.exp(6)
			);
		});
		add("mean1", () -> {
			var f = (a) -> WebDL.reduceMean(a, 0);
			equalUnOp(f,
				[[0, 1], [2, 3], [4, 5]],
				[2, 3],
				[[1 / 3, 1 / 3], [1 / 3, 1 / 3], [1 / 3, 1 / 3]]
			);
			equalUnOp(f,
				[1, 2, 3],
				2,
				[1 / 3, 1 / 3, 1 / 3]
			);
		});
		add("mean2", () -> {
			var f = (a) -> WebDL.reduceMean(a, 0, true);
			equalUnOp(f,
				[[0, 1], [2, 3], [4, 5]],
				[[2, 3]],
				[[1 / 3, 1 / 3], [1 / 3, 1 / 3], [1 / 3, 1 / 3]]
			);
			equalUnOp(f,
				[1, 2, 3],
				[2],
				[1 / 3, 1 / 3, 1 / 3]
			);
		});
		add("mean3", () -> {
			var f = (a) -> WebDL.reduceMean(a, 1);
			equalUnOp(f,
				[[0, 1], [2, 3], [4, 5]],
				[1 / 2, 5 / 2, 9 / 2],
				[[1 / 2, 1 / 2], [1 / 2, 1 / 2], [1 / 2, 1 / 2]]
			);
		});
		add("sum1", () -> {
			var f = (a) -> WebDL.reduceSum(a, 0);
			equalUnOp(f,
				[[0, 1], [2, 3], [4, 5]],
				[6, 9],
				[[1, 1], [1, 1], [1, 1]]
			);
			equalUnOp(f,
				[1, 2, 3],
				6,
				[1, 1, 1]
			);
		});
		add("sum2", () -> {
			var f = (a) -> WebDL.reduceSum(a, 0, true);
			equalUnOp(f,
				[[0, 1], [2, 3], [4, 5]],
				[[6, 9]],
				[[1, 1], [1, 1], [1, 1]]
			);
			equalUnOp(f,
				[1, 2, 3],
				[6],
				[1, 1, 1]
			);
		});
		add("sum3", () -> {
			var f = (a) -> WebDL.reduceSum(a, 1);
			equalUnOp(f,
				[[0, 1], [2, 3], [4, 5]],
				[1, 5, 9],
				[[1, 1], [1, 1], [1, 1]]
			);
		});
		add("activation_linear", () -> {
			var f = (a) -> WebDL.activation(a, Linear);
			equalUnOp(f,
				[[0, -1], [2, -3], [4, -5]],
				[[0, -1], [2, -3], [4, -5]],
				[[1, 1], [1, 1], [1, 1]]
			);
		});
		add("activation_tanh", () -> {
			var f = (a) -> WebDL.activation(a, Tangent);
			var fw:Float -> Float = (x) -> (Math.exp(2 * x) - 1.0) / (Math.exp(2 * x) + 1.0);
			var bw:Float -> Float = (x) -> 1 - fw(x) * fw(x);
			equalUnOp(f,
				[[0, -1], [2, -3], [4, -5]],
				[[fw(0), fw(-1)], [fw(2), fw(-3)], [fw(4), fw(-5)]],
				[[bw(0), bw(-1)], [bw(2), bw(-3)], [bw(4), bw(-5)]]
			);
		});
		add("activation_sigmoid", () -> {
			var f = (a) -> WebDL.activation(a, Sigmoid);
			var fw:Float -> Float = (x) -> 1 / (1 + Math.exp(-x));
			var bw:Float -> Float = (x) -> fw(x) * (1 - fw(x));
			equalUnOp(f,
				[[0, -1], [2, -3], [4, -5]],
				[[fw(0), fw(-1)], [fw(2), fw(-3)], [fw(4), fw(-5)]],
				[[bw(0), bw(-1)], [bw(2), bw(-3)], [bw(4), bw(-5)]]
			);
		});
		add("activation_relu", () -> {
			var f = (a) -> WebDL.activation(a, Relu);
			equalUnOp(f,
				[[0, -1], [2, -3], [4, -5]],
				[[0, 0], [2, 0], [4, 0]],
				[[1, 0], [1, 0], [1, 0]]
			);
		});
	}

	function testSplitMerge():Void {
		add("split1", () -> {
			var t = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]].tensorOfValue();
			var spl = WebDL.split(t, 1, [1, 1, 2]);
			var grd = spl.map((s) -> s.gradients([t])[0]);
			grd.run();
			tensorEq(spl[0], [[1], [5], [9]].tensorOfValue());
			tensorEq(spl[1], [[2], [6], [10]].tensorOfValue());
			tensorEq(spl[2], [[3, 4], [7, 8], [11, 12]].tensorOfValue());
			tensorEq(grd[0], [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]].tensorOfValue());
			tensorEq(grd[1], [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]].tensorOfValue());
			tensorEq(grd[2], [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]].tensorOfValue());
		});
		add("split2", () -> {
			var t = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]].tensorOfValue();
			var spl = WebDL.split(t, 0, [1, 2, 1]);
			var grd = spl.map((s) -> s.gradients([t])[0]);
			grd.run();
			tensorEq(spl[0], [[1, 2, 3]].tensorOfValue());
			tensorEq(spl[1], [[4, 5, 6], [7, 8, 9]].tensorOfValue());
			tensorEq(spl[2], [[10, 11, 12]].tensorOfValue());
			tensorEq(grd[0], [[1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]].tensorOfValue());
			tensorEq(grd[1], [[0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 0]].tensorOfValue());
			tensorEq(grd[2], [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1]].tensorOfValue());
		});
		add("merge1", () -> {
			var t = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]].tensorOfValue();
			var spl = [
				[[1], [5], [9]].tensorOfValue(),
				[[2], [6], [10]].tensorOfValue(),
				[[3, 4], [7, 8], [11, 12]].tensorOfValue()
			];
			var t2 = spl.merge(1);
			var grd2 = t2.gradients(spl, t);
			grd2.run();
			tensorEq(t2, t);
			tensorEq(grd2[0], spl[0]);
			tensorEq(grd2[1], spl[1]);
			tensorEq(grd2[2], spl[2]);
		});
		add("merge2", () -> {
			var t = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]].tensorOfValue();
			var spl = [
				[[1, 2, 3]].tensorOfValue(),
			    [[4, 5, 6], [7, 8, 9]].tensorOfValue(),
			    [[10, 11, 12]].tensorOfValue()
			];
			var t2 = spl.merge(0);
			var grd2 = t2.gradients(spl, t);
			grd2.run();
			tensorEq(t2, t);
			tensorEq(grd2[0], spl[0]);
			tensorEq(grd2[1], spl[1]);
			tensorEq(grd2[2], spl[2]);
		});
		add("split_merge1", () -> {
			var t = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]].tensorOfValue();
			var spl = WebDL.split(t, 1, [1, 1, 2]);
			var t2 = spl.merge(1);
			var grd = t2.gradients([t]);
			grd.run();
			tensorEq(t2, t);
			tensorEq(grd[0], [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]].tensorOfValue());
		});
		add("split_merge2", () -> {
			var t = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]].tensorOfValue();
			var spl = WebDL.split(t, 0, [1, 2, 1]);
			var t2 = spl.merge(0);
			var grd = t2.gradients([t]);
			grd.run();
			tensorEq(t2, t);
			tensorEq(grd[0], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]].tensorOfValue());
		});
	}

	function testBroadCasting():Void {
		add("broadcasting1", () -> {
			var f = WebDL.mul;
			equalBinOp(f,
				[[0, 1], [2, 3], [4, 5]],
				2,
				[[0, 2], [4, 6], [8, 10]],
				[[2, 2], [2, 2], [2, 2]],
				15
			);
		});
		add("broadcasting2", () -> {
			var f = WebDL.mul;
			equalBinOp(f,
				[[0, 1], [2, 3], [4, 5]],
				[[2, 3]],
				[[0, 3], [4, 9], [8, 15]],
				[[2, 3], [2, 3], [2, 3]],
				[[6, 9]]
			);
		});
		add("broadcasting3", () -> {
			var f = WebDL.mul;
			equalBinOp(f,
				[[0, 1], [2, 3], [4, 5]],
				[[2], [3], [4]],
				[[0, 2], [6, 9], [16, 20]],
				[[2, 2], [3, 3], [4, 4]],
				[[1], [5], [9]]
			);
		});
	}

	function testAssign():Void {
		add("assign", () -> {
			var a = [1, 2, 3, 4].tensorOfValue();
			var b = [0, 0, 0, 0].tensorOfValue();
			var c = b.assign(a);
			var d = a.add(b);
			var e = b.add(c);
			var grd = c.gradients([a, b]);
			[grd[0], grd[1], c, d, e].run();
			tensorEq(b, [1, 2, 3, 4].tensorOfValue());
			tensorEq(c, [1, 2, 3, 4].tensorOfValue());
			tensorEq(d, [1, 2, 3, 4].tensorOfValue());
			tensorEq(e, [1, 2, 3, 4].tensorOfValue());
			tensorEq(grd[0], [1, 1, 1, 1].tensorOfValue());
			tensorEq(grd[1], [0, 0, 0, 0].tensorOfValue());
		});
	}

	function testAdam():Void {
		add("adam", () -> {
			var a = WebDL.tensorOfShape([-1, 2]);
			var b = a;
			var dl;
			for (i in 0...3) {
				dl = new DenseLayer(b, 16, Linear);
				dl.init();
				b = dl.output.activation(Relu);
			}
			dl = new DenseLayer(b, 1);
			dl.init();
			b = dl.output.activation(Sigmoid);

			var teacher = [-1, 1].tensorOfShape();
			var loss = b.sub(teacher).square().reduceMean(0);
			var trainables = [loss].getTrainableVariables();

			var func:Float -> Float -> Float = (x, y) -> {
				var btoi:Bool -> Int = (b) -> b ? 1 : 0;
				return btoi(x * x + y * y < 1) ^ btoi(x < 0) ^ btoi(y < 0);
			};

			var op = new AdamOptimizer(trainables, loss.gradients(trainables));
			var num:Int = 1000;
			var minibatch:Int = 1024;
			var lss:Array<Float> = [];

			for (i in 0...num) {
				var as:Array<Array<Float>> = [];
				var ts:Array<Array<Float>> = [];
				for (i in 0...minibatch) {
					var x = Math.random() * 2 - 1;
					var y = Math.random() * 2 - 1;
					var v = func(x, y);
					as.push([x, y]);
					ts.push([v]);
				}
				a.set2D(as);
				teacher.set2D(ts);

				op.run();
				if (i % 10 == 0) {
					lss.push(loss.get1D()[0]);
				}
			};
			trace("loss: " + lss.join(", ") + "");
		});
	}

	function testBatchNorm(enabled:Bool):Void {
		var a = WebDL.tensorOfShape([-1, 2]);
		var b = a;
		var bn = enabled;
		var dl:Layer = new DenseLayer(b, 16, Linear, null, !enabled);
		var isTraining = WebDL.tensorOfValue(1);
		dl.init();
		if (bn) {
			dl = new BatchNormalizationLayer(dl.output, isTraining, 0);
			dl.init();
		}
		b = dl.output.activation(Relu);
		dl = new DenseLayer(b, 16, Linear, null, !enabled);
		dl.init();
		if (bn) {
			dl = new BatchNormalizationLayer(dl.output, isTraining, 0);
			dl.init();
		}
		b = dl.output.activation(Relu);
		dl = new DenseLayer(b, 1);
		dl.init();
		b = dl.output.activation(Sigmoid);

		var teacher = WebDL.tensorOfShape([-1, 1]);
		var loss = b.sub(teacher).square().reduceMean(0);
		var trainables = [loss].getTrainableVariables();

		var func:Float -> Float -> Float = (x, y) -> {
			var btoi:Bool -> Int = (b) -> b ? 1 : 0;
			return btoi(x * x + y * y < 1) ^ btoi(x < 0) ^ btoi(y < 0);
		};

		trace("training...");
		var op = new AdamOptimizer(trainables, loss.gradients(trainables));
		var lss:Array<Float> = [];
		var num = 1000;
		var i = 0;
		var f:Void -> Void;
		f = function():Void {
			if (i == num) {
				trace("BN = " + bn);
				trace("[" + lss.join(",") + "];");
				return;
			}
			var as:Array<Array<Float>> = [];
			var ts:Array<Array<Float>> = [];
			for (i in 0...64) {
				var x = Math.random() * Math.random() * 2 - 1;
				var y = Math.random() * Math.random() * 2 - 1;
				var v = func(x, y);
				as.push([x, y]);
				ts.push([v]);
			}
			a.set2D(as);
			teacher.set2D(ts);

			op.run();
			lss.push(loss.get1D()[0]);
			if (i % 100 == 0) trace([for (j in 0...Std.int(i / 100) + 1) "."].join(""));
			i++;
			Timer.delay(f, 0);
		};
		f();

		/*
		var a = WebDL.tensorOfValue([
			[for (i in 0...1000) RandUtil.normal(10, 0.1)],
			[for (i in 0...1000) RandUtil.normal(0, 10)],
			[for (i in 0...1000) RandUtil.normal(5, 3)],
			[for (i in 0...1000) RandUtil.normal(-2, 1)]
		]);
		var po = WebDL.tensorOfValue(1);
		var bn = new BatchNormalizationLayer(a, po, 1);
		for (i in 0...100) bn.updates.run();
		po.set0D(0);
		[bn.output].run();
		trace(a.getArray().join(","));
		trace(bn.output.getArray().join(","));
		*/
	}

	function equal1D(a:Array<Float>, b:Array<Float>):Void {
		var res:Bool = true;
		do {
			if (a.length != b.length) {
				res = false;
				break;
			}
			for (i in 0...a.length) {
				var dif = a[i] - b[i];
				if (dif < -EPS || dif > EPS) {
					res = false;
					break;
				}
			}
		} while (false);
		if (!res) {
			throw '$a should equal to $b';
		}
	}

	function equal2D(a:Array<Array<Float>>, b:Array<Array<Float>>):Void {
		if (a.length != b.length) throw '$a should equal to $b';
		equal1D(flatten(a), flatten(b));
	}

	function equal3D(a:Array<Array<Array<Float>>>, b:Array<Array<Array<Float>>>):Void {
		if (a.length != b.length) throw '$a should equal to $b';
		equal2D(flatten(a), flatten(b));
	}

	function equal4D(a:Array<Array<Array<Array<Float>>>>, b:Array<Array<Array<Array<Float>>>>):Void {
		if (a.length != b.length) throw '$a should equal to $b';
		equal3D(flatten(a), flatten(b));
	}

	function equalBinOp(f:Tensor -> Tensor -> Tensor, a:Any, b:Any, c:Any, dadc:Any, dbdc:Any):Void {
		var a2 = a.tensorOfValue();
		var b2 = b.tensorOfValue();
		var c2 = f(a2, b2);
		var grads = c2.gradients([a2, b2]);
		grads.run();
		tensorEq(c2, c.tensorOfValue());
		tensorEq(grads[0], dadc.tensorOfValue());
		tensorEq(grads[1], dbdc.tensorOfValue());
	}

	function equalUnOp(f:Tensor -> Tensor, a:Any, b:Any, dadb:Any):Void {
		var a2 = a.tensorOfValue();
		var b2 = f(a2);
		var grads = b2.gradients([a2]);
		grads.run();
		tensorEq(b2, b.tensorOfValue());
		tensorEq(grads[0], dadb.tensorOfValue());
	}

	function tensorEq(a:Tensor, b:Tensor):Void {
		switch ([a.rank, b.rank]) {
			case [0, 0]: equal1D([a.get0D()], [b.get0D()]);
			case [1, 1]: equal1D(a.get1D(), b.get1D());
			case [2, 2]: equal2D(a.get2D(), b.get2D());
			case [3, 3]: equal3D(a.get3D(), b.get3D());
			case [4, 4]: equal4D(a.get4D(), b.get4D());
			case _: throw "tensor ranks mismatch: " + a.rank + " and " + b.rank;
		}
	}

	function flatten<T>(a:Array<Array<T>>):Array<T> {
		var res:Array<T> = [];
		for (a2 in a) {
			for (e in a2) {
				res.push(e);
			}
		}
		return res;
	}

	function add(name:String, f:Void -> Void):Void {
		tests.push(new Test(name, f));
	}

	function run():Void {
		var passed:Int = 0;
		var failed:Int = 0;
		for (t in tests) {
			if (t.run()) {
				passed++;
			} else {
				failed++;
			}
		}
		trace("\n--------------------------------");
		trace('passed: $passed');
		trace('failed: $failed');
		if (failed == 0) {
			trace("all tests passed");
		} else {
			trace("test failed");
		}
	}

	static function main():Void {
		new WebDLTest();
	}

}
