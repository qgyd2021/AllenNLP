#!/usr/bin/env bash

# sh run.sh --system_version windows --stage 1 --stop_stage 1
# sh run.sh --system_version windows --stage 3 --stop_stage 3
# bash run.sh --system_version windows --stage 4 --stop_stage 4
# sh run.sh --system_version windows --stage 5 --stop_stage 5

# nohup sh run.sh --system_version centos --stage 0 --stop_stage 3 &
# sh run.sh --system_version centos --stage 4 --stop_stage 4 --trained_model_name voc_post_domain_textcnn
# nohup sh run.sh --system_version centos --stage 5 --stop_stage 5 &

# params
system_version="windows";
verbose=true;
stage=0
stop_stage=5

pretrained_model_supplier=dienstag
pretrained_model_name=chinese-bert-wwm-ext

trained_model_name=voc_post_domain_textcnn

# parse options
while true; do
  [ -z "${1:-}" ] && break;
  case "$1" in
    --*) name=$(echo "$1" | sed s/^--// | sed s/-/_/g);
      eval '[ -z "${'"$name"'+xxx}" ]' && echo "$0: invalid option $1" 1>&2 && exit 1;
      old_value="(eval echo \\$$name)";
      if [ "${old_value}" == "true" ] || [ "${old_value}" == "false" ]; then
        was_bool=true;
      else
        was_bool=false;
      fi

      eval "${name}=\"$2\"";

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
serialization_dir="${data_dir}/serialization_dir"

pretrained_models_dir="${work_dir}/../../../pretrained_models"
trained_models_dir="${work_dir}/../../../trained_models/"

mkdir -p "${data_dir}"
mkdir -p "${serialization_dir}"
mkdir -p "${pretrained_models_dir}"
mkdir -p "${trained_models_dir}"

vocabulary_dir="${data_dir}/vocabulary"
train_subset="${data_dir}/train.jsonl"
valid_subset="${data_dir}/valid.jsonl"
evaluation_output_file="${data_dir}/evaluation.xlsx"

export PYTHONPATH="${work_dir}/../../.."

if [ $system_version == "windows" ]; then
  alias python3='D:/Users/Administrator/PycharmProjects/virtualenv/AllenNLP/Scripts/python.exe'
elif [ $system_version == "centos" ]; then
  source /data/local/bin/AllenNLP/bin/activate
  alias python3='/data/local/bin/AllenNLP/bin/python3'
elif [ $system_version == "ubuntu" ]; then
  source /data/local/bin/AllenNLP/bin/activate
  alias python3='/data/local/bin/AllenNLP/bin/python3'
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  $verbose && echo "stage 0: download pretrained model"
  cd "${work_dir}" || exit 1;

  python3 step_0_download_model.py \
  --output_dir "${pretrained_models_dir}/${pretrained_model_supplier}/${pretrained_model_name}" \

fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  $verbose && echo "stage 1: prepare data"
  cd "${work_dir}" || exit 1;

  python3 step_1_prepare_data.py \
  --train_subset "${train_subset}" \
  --valid_subset "${valid_subset}" \

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  $verbose && echo "stage 2: make vocabulary"
  cd "${work_dir}" || exit 1;

  python3 step_2_make_vocabulary.py \
  --pretrained_model_path "${pretrained_models_dir}/${pretrained_model_supplier}/${pretrained_model_name}" \
  --train_subset "${train_subset}" \
  --valid_subset "${valid_subset}" \
  --vocabulary_dir "${vocabulary_dir}" \

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  $verbose && echo "stage 3: train model"
  cd "${work_dir}" || exit 1;

  python3 step_3_train_model.py \
  --pretrained_model_path "${pretrained_models_dir}/${pretrained_model_supplier}/${pretrained_model_name}" \
  --train_subset "${train_subset}" \
  --valid_subset "${valid_subset}" \
  --vocabulary_dir "${vocabulary_dir}" \
  --serialization_dir "${serialization_dir}" \

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  $verbose && echo "stage 4: collect files"
  cd "${work_dir}" || exit 1;

  mkdir -p "${trained_models_dir}/${trained_model_name}"
  mkdir -p "${trained_models_dir}/${trained_model_name}/tokenizer"

  cp -r "${vocabulary_dir}" "${trained_models_dir}/${trained_model_name}/vocabulary/"
  cp "${serialization_dir}/best.th" "${trained_models_dir}/${trained_model_name}/weights.th"
  for tokenizer_file in config.json tokenizer.json tokenizer_config.json special_tokens_map.json added_tokens.json; do
    cp "${pretrained_models_dir}/${pretrained_model_supplier}/${pretrained_model_name}/${tokenizer_file}" \
       "${trained_models_dir}/${trained_model_name}/tokenizer/"
  done
  cp "config.json" "${trained_models_dir}/${trained_model_name}/config.json"

fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  $verbose && echo "stage 5: evaluation"
  cd "${work_dir}" || exit 1;

  python3 step_4_evaluation.py \
  --archive_file "${trained_models_dir}/${trained_model_name}" \
  --train_subset "${train_subset}" \
  --valid_subset "${valid_subset}" \
  --evaluation_output_file "${evaluation_output_file}" \

fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  $verbose && echo "stage 6: predict by archive"
  cd "${work_dir}" || exit 1;

  python3 step_5_predict_by_archive.py \
  --archive_file "${trained_models_dir}/${trained_model_name}" \

fi
