from enum import Enum


class observation_spaces(Enum):
    Ir = "Ir"
    BitcodeFile = "BitcodeFile"
    InstCount = "InstCount"
    Autophase = "Autophase"
    Programl = "Programl"
    CpuInfo = "CpuInfo"
    IrInstructionCount = "IrInstructionCount"
    IrInstructionCountO0 = "IrInstructionCountO0"
    IrInstructionCountO3 = "IrInstructionCountO3"
    IrInstructionCountOz = "IrInstructionCountOz"
    ObjectTextSizeBytes = "ObjectTextSizeBytes"
    ObjectTextSizeO0 = "ObjectTextSizeO0"
    ObjectTextSizeO3 = "ObjectTextSizeO3"
    ObjectTextSizeOz = "ObjectTextSizeOz"
    Inst2vecPreprocessedText = "Inst2vecPreprocessedText"
    Inst2vecEmbeddingIndices = "Inst2vecEmbeddingIndices"
    Inst2vec = "Inst2vec"
    InstCountDict = "InstCountDict"
    InstCountNorm = "InstCountNorm"
    InstCountNormDict = "InstCountNormDict"
    AutophaseDict = "AutophaseDict"


class reward_spaces(Enum):
    IrInstructionCount = "IrInstructionCount"
    IrInstructionCountNorm = "IrInstructionCountNorm"
    IrInstructionCountO3 = "IrInstructionCountO3"
    IrInstructionCountOz = "IrInstructionCountOz"
    ObjectTextSizeBytes = "ObjectTextSizeBytes"
    ObjectTextSizeNorm = "ObjectTextSizeNorm"
    ObjectTextSizeO3 = "ObjectTextSizeO3"
    ObjectTextSizeOz = "ObjectTextSizeOz"
