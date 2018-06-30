package webdl.core.nn;

/**
 * The list of types of activation functions.
 */
@:enum
abstract Activation(Int) to Int {
	var Linear = 0;
	var Tangent = 1;
	var Sigmoid = 2;
	var Relu = 3;
}
