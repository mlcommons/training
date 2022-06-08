#!/bin/bash

COM_DIR="/lustre/fsw/mlperf/mlperft-llm/c4/preprocessed"
C4_0="${COM_DIR}/c4_en_0_GPT2BPETokenizer_text_document"
C4_1="${COM_DIR}/c4_en_1_GPT2BPETokenizer_text_document"
C4_2="${COM_DIR}/c4_en_2_GPT2BPETokenizer_text_document"
C4_3="${COM_DIR}/c4_en_3_GPT2BPETokenizer_text_document"
C4_4="${COM_DIR}/c4_en_4_GPT2BPETokenizer_text_document"
C4_5="${COM_DIR}/c4_en_5_GPT2BPETokenizer_text_document"
C4_6="${COM_DIR}/c4_en_6_GPT2BPETokenizer_text_document"
C4_7="${COM_DIR}/c4_en_7_GPT2BPETokenizer_text_document"
DATA_BLEND="0.125 ${C4_0} 0.125 ${C4_1} 0.125 ${C4_2} 0.125 ${C4_3} 0.125 ${C4_4} 0.125 ${C4_5} 0.125 ${C4_6} 0.125 ${C4_7}"
