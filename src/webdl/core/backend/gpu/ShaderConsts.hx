package webdl.core.backend.gpu;

/**
 * GLSL shader constants.
 */
class ShaderConsts {
	// attributes
	public static inline var A_POS:String = "aPos";
	public static inline var A_UV:String = "aUv";

	// varyings
	public static inline var V_UV:String = "vUv";

	// uniforms
	public static inline var U_SRC_TEX:String = "uSrcTex";
	public static inline var U_SRC_TEX_SIZE:String = "uSrcTexSize";
	public static inline var U_SRC_INV_TEX_SIZE:String = "uSrcInvTexSize";
	public static inline var U_SRC_RANK:String = "uSrcRank";
	public static inline var U_SRC_SHAPE:String = "uSrcShape";
	public static inline var U_SRC_STRIDE:String = "uSrcStride";

	public static inline var            U_SRC_TEX1:String = U_SRC_TEX          + "1";
	public static inline var       U_SRC_TEX_SIZE1:String = U_SRC_TEX_SIZE     + "1";
	public static inline var   U_SRC_INV_TEX_SIZE1:String = U_SRC_INV_TEX_SIZE + "1";
	public static inline var           U_SRC_RANK1:String = U_SRC_RANK         + "1";
	public static inline var          U_SRC_SHAPE1:String = U_SRC_SHAPE        + "1";
	public static inline var         U_SRC_STRIDE1:String = U_SRC_STRIDE       + "1";

	public static inline var            U_SRC_TEX2:String = U_SRC_TEX          + "2";
	public static inline var       U_SRC_TEX_SIZE2:String = U_SRC_TEX_SIZE     + "2";
	public static inline var   U_SRC_INV_TEX_SIZE2:String = U_SRC_INV_TEX_SIZE + "2";
	public static inline var           U_SRC_RANK2:String = U_SRC_RANK         + "2";
	public static inline var          U_SRC_SHAPE2:String = U_SRC_SHAPE        + "2";
	public static inline var         U_SRC_STRIDE2:String = U_SRC_STRIDE       + "2";

	public static inline var            U_SRC_TEX3:String = U_SRC_TEX          + "3";
	public static inline var       U_SRC_TEX_SIZE3:String = U_SRC_TEX_SIZE     + "3";
	public static inline var   U_SRC_INV_TEX_SIZE3:String = U_SRC_INV_TEX_SIZE + "3";
	public static inline var           U_SRC_RANK3:String = U_SRC_RANK         + "3";
	public static inline var          U_SRC_SHAPE3:String = U_SRC_SHAPE        + "3";
	public static inline var         U_SRC_STRIDE3:String = U_SRC_STRIDE       + "3";

	public static inline var            U_SRC_TEX4:String = U_SRC_TEX          + "4";
	public static inline var       U_SRC_TEX_SIZE4:String = U_SRC_TEX_SIZE     + "4";
	public static inline var   U_SRC_INV_TEX_SIZE4:String = U_SRC_INV_TEX_SIZE + "4";
	public static inline var           U_SRC_RANK4:String = U_SRC_RANK         + "4";
	public static inline var          U_SRC_SHAPE4:String = U_SRC_SHAPE        + "4";
	public static inline var         U_SRC_STRIDE4:String = U_SRC_STRIDE       + "4";

	public static inline var U_DST_TEX:String = "uDstTex";
	public static inline var U_DST_TEX_SIZE:String = "uDstTexSize";
	public static inline var U_DST_INV_TEX_SIZE:String = "uDstInvTexSize";
	public static inline var U_DST_RANK:String = "uDstRank";
	public static inline var U_DST_SHAPE:String = "uDstShape";
	public static inline var U_DST_STRIDE:String = "uDstStride";

	public static inline var U_DST_ACTUAL_SIZE:String = "uDstActualSize";

	public static inline var MAX_DIMENSION_SIZE:Int = 1000000;

	public static inline var SRC_ACCESS1:String = "texture2D(" + U_SRC_TEX1 + ", " + V_UV + ")";
	public static inline var SRC_ACCESS2:String = "texture2D(" + U_SRC_TEX2 + ", " + V_UV + ")";
	public static inline var SRC_ACCESS3:String = "texture2D(" + U_SRC_TEX3 + ", " + V_UV + ")";
	public static inline var SRC_ACCESS4:String = "texture2D(" + U_SRC_TEX4 + ", " + V_UV + ")";
}
