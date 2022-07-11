#!/bin/bash

COM_DIR="/c4/preprocessed_c4_googlespm"
C4_0="${COM_DIR}/c4_en_0_c4_spm_text_document"
C4_1="${COM_DIR}/c4_en_1_c4_spm_text_document"
C4_2="${COM_DIR}/c4_en_2_c4_spm_text_document"
C4_3="${COM_DIR}/c4_en_3_c4_spm_text_document"
C4_4="${COM_DIR}/c4_en_4_c4_spm_text_document"
C4_5="${COM_DIR}/c4_en_5_c4_spm_text_document"
C4_6="${COM_DIR}/c4_en_6_c4_spm_text_document"
C4_7="${COM_DIR}/c4_en_7_c4_spm_text_document"
DATA_BLEND="0.125 ${C4_0} 0.125 ${C4_1} 0.125 ${C4_2} 0.125 ${C4_3} 0.125 ${C4_4} 0.125 ${C4_5} 0.125 ${C4_6} 0.125 ${C4_7}"
VALID_C4="${COM_DIR}/c4_en_validation_c4_spm_text_document"
VALID_DATA_BLEND="1.00 ${VALID_C4}"
