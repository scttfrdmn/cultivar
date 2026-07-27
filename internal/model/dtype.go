package model

import "strings"

// DTypeBytes returns the bytes per parameter for a Hugging Face safetensors dtype
// key, and whether the key is recognized.
//
// This exists because `safetensors.parameters` is a map keyed by dtype, not a
// count. Summing its values and multiplying by one width is the single most
// consequential sizing error available: openai/gpt-oss-120b reports
// {BF16: 2.17e9, U8: 118.2e9}, so a naive sum times 2 bytes gives 224 GiB against
// a correct 114 GiB. That is the difference between "needs two p5.48xlarge" and
// "fits on one".
//
// An unrecognized dtype must not be guessed. A wrong width produces a confident
// number that is wrong by an integer factor, and the caller can say "unknown
// dtype" instead — see [Model.WeightBytes].
func DTypeBytes(dtype string) (float64, bool) {
	switch strings.ToUpper(strings.TrimSpace(dtype)) {
	case "F64", "FLOAT64", "I64", "U64", "INT64":
		return 8, true
	case "F32", "FLOAT32", "I32", "U32", "INT32":
		return 4, true
	case "F16", "FLOAT16", "BF16", "BFLOAT16", "I16", "U16", "INT16":
		return 2, true
	// 8-bit: quantized weights and FP8 checkpoints. U8 is what MXFP4 models like
	// gpt-oss report their packed expert weights as.
	case "F8_E4M3", "F8_E5M2", "FP8", "I8", "U8", "INT8", "UINT8", "BOOL":
		return 1, true
	// 4-bit: two parameters per byte.
	case "F4", "FP4", "I4", "U4", "INT4", "UINT4", "MXFP4", "NF4":
		return 0.5, true
	default:
		return 0, false
	}
}
