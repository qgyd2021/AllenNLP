#!/usr/bin/env bash

# nohup sh run.sh --system_version centos --stage -2 --stop_stage 2 &
# sh run.sh --system_version windows --stage -2 --stop_stage -2
# sh run.sh --system_version windows --stage -1 --stop_stage -1
# sh run.sh --system_version windows --stage 0 --stop_stage 1
# sh run.sh --system_version windows --stage 1 --stop_stage 2
# sh run.sh --system_version windows --stage 0 --stop_stage 3
# sh run.sh --system_version windows --stage 6 --stop_stage 7

# params
system_version="centos";
verbose=true;
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5


trained_model_name=chinese_bilstm_ner_kbqa

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
trained_models_dir="${work_dir}/../../../trained_models";

serialization_dir="${data_dir}/serialization_dir";

mkdir -p "${data_dir}"
mkdir -p "${pretrained_models_dir}"
mkdir -p "${trained_models_dir}"
mkdir -p "${serialization_dir}"

pre_train_subset="${data_dir}/kbqa-train.json"
pre_valid_subset="${data_dir}/kbqa-valid.json"
pre_test_subset="${data_dir}/kbqa-test.json"

vocabulary_dir="${data_dir}/vocabulary"
train_subset="${data_dir}/train.json"
valid_subset="${data_dir}/valid.json"
test_subset="${data_dir}/test.json"


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
    rm -rf .git/
    wget "${repository_url}/resolve/main/pytorch_model.bin"
  fi
fi


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  $verbose && echo "stage -1: download data"
  cd "${data_dir}" || exit 1;

  # NLPCC-KBQA
  git clone https://github.com/nanduan/NLPCC-KBQA

  # CCKS
  git clone https://github.com/pkumod/CKBQA

fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  $verbose && echo "stage 0: preprocess data nlpcc"
  cd "${work_dir}" || exit 1;

  python3 1.preprocess_data_nlpcc.py \
  --data_path "${data_dir}/NLPCC-KBQA/nlpcc2016-2018.kbqa.train" \
  --pre_train_subset "${data_dir}/kbqa-train.json" \

fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  $verbose && echo "stage 1: preprocess data ccks"
  cd "${work_dir}" || exit 1;

  python3 2.preprocess_data_ccks2019.py \
  --data_path "${data_dir}/CKBQA/data/ccks2019" \
  --pre_train_subset "${pre_train_subset}" \
  --pre_valid_subset "${pre_valid_subset}" \
  --pre_test_subset "${pre_test_subset}" \

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  $verbose && echo "stage 2: prepare data"
  cd "${work_dir}" || exit 1;

  python3 3.prepare_data.py \
  --pretrained_model_path "${pretrained_model_dir}" \
  --pre_train_subset "${pre_train_subset}" \
  --pre_valid_subset "${pre_valid_subset}" \
  --train_subset "${train_subset}" \
  --valid_subset "${valid_subset}" \
  --test_subset "${test_subset}" \

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  $verbose && echo "stage 3: make vocabulary"
  cd "${work_dir}" || exit 1;

  python3 4.make_vocabulary.py \
  --pretrained_model_path "${pretrained_model_dir}" \
  --train_subset "${valid_subset}" \
  --valid_subset "${valid_subset}" \
  --vocabulary_dir "${vocabulary_dir}" \

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  $verbose && echo "stage 4: make json config"
  cd "${work_dir}" || exit 1;

  python3 5.make_json_config.py \
  --pretrained_model_path "${pretrained_model_dir}" \
  --train_subset "${train_subset}" \
  --valid_subset "${valid_subset}" \
  --vocabulary_dir "${vocabulary_dir}" \
  --serialization_dir "${serialization_dir}" \
  --json_config_dir "${data_dir}" \

fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  $verbose && echo "stage 5: train model"
  cd "${data_dir}" || exit 1;
  if [ ! -e weights.th ]; then
    wget "https://huggingface.co/spaces/qgyd2021/ChineseLstmPosConll2012/resolve/main/trained_models/chinese_lstm_pos_conll2012/weights.th"
  fi

  cd "${work_dir}" || exit 1;

  python3 6.train_model.py \
  --pretrained_pos_model_path "${data_dir}/weights.th" \
  --train_subset "${train_subset}" \
  --valid_subset "${valid_subset}" \
  --vocabulary_dir "${vocabulary_dir}" \
  --serialization_dir "${serialization_dir}" \

fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  $verbose && echo "stage 6: collect files"
  cd "${work_dir}" || exit 1;

  mkdir -p "${trained_models_dir}/${trained_model_name}"

  cp -r "${vocabulary_dir}" "${trained_models_dir}/${trained_model_name}/vocabulary/"
  cp "${serialization_dir}/best.th" "${trained_models_dir}/${trained_model_name}/weights.th"
  cp "${data_dir}/config.json" "${trained_models_dir}/${trained_model_name}/config.json"

fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  $verbose && echo "stage 8: predict by archive"
  cd "${work_dir}" || exit 1;

  python3 8.predict_by_archive.py \
  --archive_file "${trained_models_dir}/${trained_model_name}" \
  --pretrained_model_path "${pretrained_model_dir}" \

fi
