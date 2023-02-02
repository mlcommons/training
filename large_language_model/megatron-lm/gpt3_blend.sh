#!/bin/bash

COM_DIR="/c4/preprocessed_c4_googlespm"
C4_0="${COM_DIR}/c4_en_6_c4_spm_text_document"
C4_1="${COM_DIR}/c4_en_7_c4_spm_text_document"
DATA_BLEND="0.5 ${C4_0} 0.5 ${C4_1}"
VALID_C4="${COM_DIR}/c4_en_validation_subset_c4_spm_text_document"
VALID_DATA_BLEND="1.00 ${VALID_C4}"
