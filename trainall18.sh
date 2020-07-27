# !/bin/bash
# This goes in ..

export CUDA_VISIBLE_DEVICES=$1
source ../tf3/bin/activate

STACK=stack

mkdir -p $STACK
while read -r LANGUAGE TREEBANK LC TB
do
  echo $LANGUAGE $TREEBANK $LC $TB
  if [ ! -e $STACK/$LANGUAGE-$TREEBANK ] 
  then
    touch $STACK/$LANGUAGE-$TREEBANK

    # For Chinese, the vectors are named differently
    # For Norwegian, there are two sets of vectors
    if [ "$LC" == "zh" ]
    then
      pretrained_file=data/word2vec/"$LANGUAGE"T/"$LC".vectors.xz
    elif [ "$LC" == "no" ]
    then
      if [ "$TB" == "bokmaal" ]
      then
        pretrained_file=data/word2vec/"$LANGUAGE"-Bokmaal/"$LC"_bokmaal.vectors.xz
      else
        pretrained_file=data/word2vec/"$LANGUAGE"-Nynorsk/"$LC"_nynorsk.vectors.xz
      fi
    else
      pretrained_file=data/word2vec/$LANGUAGE/"$LC".vectors.xz
    fi
  
  
    # Some languages have composite XPOS tags that can be treated like Features
    XPOSFeatureVocabTreebanks=(Ancient_Greek-Perseus Arabic-PADT Czech-CAC Czech-FicTree Czech-PDT Latin-Perseus_XV)
    if [[ " ${XPOSFeatureVocabTreebanks[*]} " == *" $LANGUAGE-$TREEBANK "* ]]
    then
      TaggerNetworkFlags="--TaggerNetwork output_vocab_classes=UPOSTokenVocab:XPOSFeatureVocab:UFeatsFeatureVocab --FormSubtokenVocab cased=False"
      ParserNetworkFlags="--ParserNetwork input_vocab_classes=FormMultivocab:UPOSTokenVocab:XPOSFeatureVocab:UFeatsFeatureVocab:LemmaTokenVocab --FormSubtokenVocab cased=False"
    else
      TaggerNetworkFlags="--TaggerNetwork share_layer=True"
      ParserNetworkFlags=""
    fi
  
    # We want to save the XV datasets without the XV at the end (to avoid downstream issues)
    basename=CoNLL18/UD_$LANGUAGE-$TREEBANK
    if [[ $TREEBANK == *_XV ]]
    then
      basename1=CoNLL18_Tim/UD_$LANGUAGE-${TREEBANK:0:-3}
    else
      basename1=CoNLL18_Tim/UD_$LANGUAGE-$TREEBANK
    fi
    train_conllus=data/$basename/"$LC"_$TB-ud-train.conllu
    dev_conllus=data/$basename/"$LC"_$TB-ud-dev.conllu
  
    # If the validation files don't exist, use the training files
    if [ ! -e $dev_conllus ]
    then
      dev_conllus=$train_conllus
    fi
  
    if [ $LANGUAGE == 'Czech' ]
    then
      XPOSFeatureVocabFlags="--XPOSFeatureVocab cliplen=15"
    else
      XPOSFeatureVocabFlags=""
    fi
  
  
    # Train the tagger and make sure it runs fine
     python main.py --save_dir saves/$basename1/Tagger train TaggerNetwork --DEFAULT train_conllus=$train_conllus dev_conllus=$dev_conllus LANG=$LANGUAGE TREEBANK=$TREEBANK LC=$LC TB=$TB --FormPretrainedVocab pretrained_file=$pretrained_file --force --noscreen $TaggerNetworkFlags $XPOSFeatureVocabFlags --BaseNetwork max_steps=7500
    python main.py --save_dir saves/$basename1/Tagger run $train_conllus $dev_conllus
  
    # Grab the re-tagged files and add the comments/compounds back in
    tagged_train_conllus=saves/$basename1/Tagger/parsed/$train_conllus
    tagged_dev_conllus=saves/$basename1/Tagger/parsed/$dev_conllus
    python scripts/reinsert_compounds.py $train_conllus $tagged_train_conllus
    if [[ "$train_conllus" != "$dev_conllus" ]]; then
      python scripts/reinsert_compounds.py $dev_conllus $tagged_dev_conllus
    fi
  
    # Train the Parser and make sure it runs fine
    python main.py --save_dir saves/$basename1/Parser train ParserNetwork --DEFAULT train_conllus=$tagged_train_conllus dev_conllus=$tagged_dev_conllus LANG=$LANGUAGE TREEBANK=$TREEBANK LC=$LC TB=$TB --FormPretrainedVocab pretrained_file=$pretrained_file --force --noscreen $ParserNetworkFlags $XPOSFeatureVocabFlags --BaseNetwork max_steps=7500 recur_size=200
    python main.py --save_dir saves/$basename1/Parser run $tagged_train_conllus $tagged_dev_conllus
  
    parsed_train_conllus=saves/$basename1/Parser/parsed/$tagged_train_conllus
    parsed_dev_conllus=saves/$basename1/Parser/parsed/$tagged_dev_conllus
    python scripts/reinsert_compounds.py $train_conllus $parsed_train_conllus
    if [[ "$train_conllus" != "$dev_conllus" ]]; then
      python scripts/reinsert_compounds.py $dev_conllus $parsed_dev_conllus
    fi
  
    # Save the eval output to the treebank save directory
    python scripts/conll18_ud_eval.py -v $dev_conllus $parsed_dev_conllus > saves/$basename1/evaluation.txt

    if [ $? -eq 0 ] 
    then 
      echo "Success" > $STACK/$LANGUAGE-$TREEBANK
    else 
      echo "Failure" > $STACK/$LANGUAGE-$TREEBANK
    fi 
  fi
done < conll18_treebanks.txt
