#!/bin/bash
MODEL=$1
OUTPUT=$2
PARSE_ID=$3
#PARSE_OOD=$4
echo test ood score: UF1, LF1
python main.py --save_dir saves/$MODEL/UnlabelGraphParserNetwork run --output_dir $OUTPUT/$MODEL $PARSE_ID
python scripts/semdep_eval.py $PARSE_ID $OUTPUT/$MODEL/`basename $PARSE_ID`
#echo test ood score: UF1, LF1
#python main.py --save_dir saves/$MODEL/UnlabelGraphParserNetwork run --output_dir $OUTPUT/$MODEL $PARSE_OOD
#python scripts/semdep_eval.py $PARSE_OOD $OUTPUT/$MODEL/`basename $PARSE_OOD`
