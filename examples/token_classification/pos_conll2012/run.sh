#!/usr/bin/env bash

# nohup sh run.sh --system_version centos --stage -2 --stop_stage 2 &
# sh run.sh --system_version windows --stage -2 --stop_stage -2
# sh run.sh --system_version windows --stage -1 --stop_stage -1
# sh run.sh --system_version windows --stage 0 --stop_stage 1
# sh run.sh --system_version windows --stage 1 --stop_stage 2

# params
system_version="centos";
verbose=true;
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5


pretrained_bert_model_name=chinese-bert-wwm-ext


# parse options
while true; do
  [ -z "${1:-}" ] && break;  # break if there are no arguments
  case "$1" in
    --*) name=$(echo "$1" | sed s/^--// | sed s/-/_/g);
      eval '[ -z "${'"$name"'+xxx}" ]' && echo "$0: invalid option $1" 1>&2 && exit 1;
      old_value="(eval echo \\$$name)";
      if [ "${old_value}" == "true" ] || [ "${old_value}" == "false" ]; then
        was_bool=true;
      else
        was_bool=false;
      fi

      # Set the variable to the right value-- the escaped quotes make it work if
      # the option had spaces, like --cmd "queue.pl -sync y"
      eval "${name}=\"$2\"";

      # Check that Boolean-valued arguments are really Boolean.
      if $was_bool && [[ "$2" != "true" && "$2" != "false" ]]; then
        echo "$0: expected \"true\" or \"false\": $1 $2" 1>&2
        exit 1;
      fi
      shift 2;
      ;;

    *) break;
  esac
done

$verbose && echo "system_version: ${system_version}"

work_dir="$(pwd)"
data_dir="$(pwd)/data_dir"
pretrained_models_dir="${work_dir}/../../../pretrained_models";

mkdir -p "${data_dir}"
mkdir -p "${pretrained_models_dir}"


export PYTHONPATH="${work_dir}/../../.."

if [ $system_version == "windows" ]; then
  alias python3='C:/Users/tianx/PycharmProjects/virtualenv/AllenNLP/Scripts/python.exe'
elif [ $system_version == "centos" ]; then
  source /data/local/bin/AllenNLP/bin/activate
  alias python3='/data/local/bin/AllenNLP/bin/python3'
elif [ $system_version == "ubuntu" ]; then
  source /data/local/bin/AllenNLP/bin/activate
  alias python3='/data/local/bin/AllenNLP/bin/python3'
fi


declare -A pretrained_bert_model_dict
pretrained_bert_model_dict=(
  ["chinese-bert-wwm-ext"]="https://huggingface.co/hfl/chinese-bert-wwm-ext"
  ["bert-base-uncased"]="https://huggingface.co/bert-base-uncased"
  ["bert-base-japanese"]="https://huggingface.co/cl-tohoku/bert-base-japanese"
  ["bert-base-vietnamese-uncased"]="https://huggingface.co/trituenhantaoio/bert-base-vietnamese-uncased"

)
pretrained_model_dir="${pretrained_models_dir}/${pretrained_bert_model_name}"


if [ ${stage} -le -2 ] && [ ${stop_stage} -ge -2 ]; then
  $verbose && echo "stage -2: download pretrained models"
  cd "${work_dir}" || exit 1;

  if [ ! -d "${pretrained_model_dir}" ]; then
    mkdir -p "${pretrained_models_dir}"
    cd "${pretrained_models_dir}" || exit 1;

    repository_url="${pretrained_bert_model_dict[${pretrained_bert_model_name}]}"
    git clone "${repository_url}"

    cd "${pretrained_model_dir}" || exit 1;
    rm flax_model.msgpack && rm pytorch_model.bin && rm tf_model.h5
    wget "${repository_url}/resolve/main/pytorch_model.bin"
  fi
fi


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  $verbose && echo "stage -1: download data"
  cd "${data_dir}" || exit 1;
  # source1
  # https://data.mendeley.com/datasets/zmycy7t9h9/2

  wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zmycy7t9h9-2.zip
  unzip zmycy7t9h9-2.zip
  rm -rf zmycy7t9h9-2.zip

  mv zmycy7t9h9-2/conll-2012.zip conll-2012.zip
  unzip conll-2012.zip
  rm -rf conll-2012.zip

  rm -rf zmycy7t9h9-2

  # source2
  # https://cemantix.org/conll/2012/data.html

  # wget -c http://conll.cemantix.org/2012/download/conll-2012-train.v4.tar.gz
  # wget -c http://conll.cemantix.org/2012/download/conll-2012-development.v4.tar.gz
  #
  # tar -zxvf conll-2012-train.v4.tar.gz
  # tar -zxvf conll-2012-development.v4.tar.gz
  #
  # rm -rf conll-2012-train.v4.tar.gz
  # rm -rf conll-2012-development.v4.tar.gz

fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  $verbose && echo "stage 0: prepare data"
  cd "${work_dir}" || exit 1;

  python3 1.prepare_data.py \
  --pretrained_model_path "${pretrained_model_dir}" \
  --file_dir "${data_dir}/conll-2012/v4/data/train" \
  --output_file "${data_dir}/train.json"

  python3 1.prepare_data.py \
  --pretrained_model_path "${pretrained_model_dir}" \
  --file_dir "${data_dir}/conll-2012/v4/data/development" \
  --output_file "${data_dir}/valid.json"

fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  $verbose && echo "stage 1: make vocabulary"
  cd "${work_dir}" || exit 1;

  python3 2.make_vocabulary.py \
  --pretrained_model_path "${pretrained_model_dir}" \

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  $verbose && echo "stage 2: train model"
  cd "${work_dir}" || exit 1;

  python3 3.train_model.py \
  --pretrained_model_path "${pretrained_model_dir}" \

fi
