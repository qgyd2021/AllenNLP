#!/usr/bin/env bash

# sh run.sh --system_version windows --stage 0 --stop_stage 0
# sh run.sh --system_version windows --stage 3 --stop_stage 3
# sh run.sh --system_version windows --stage 4 --stop_stage 4
# sh run.sh --system_version windows --stage 5 --stop_stage 5
# sh run.sh --system_version windows --stage 6 --stop_stage 6

# nohup sh run.sh --system_version centos --stage 0 --stop_stage 3 &
# nohup sh run.sh --system_version centos --stage 3 --stop_stage 3 &

# sh run.sh --system_version centos --stage 2 --stop_stage 2

# params
system_version="windows";
verbose=true;
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5

trained_model_name=language_identification

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
serialization_dir="${data_dir}/serialization_dir"

trained_models_dir="${work_dir}/../../../trained_models/${trained_model_name}"


mkdir -p "${data_dir}"
mkdir -p "${trained_models_dir}"
mkdir -p "${serialization_dir}"

vocabulary_dir="${data_dir}/vocabulary"
train_subset="${data_dir}/train.jsonl"
valid_subset="${data_dir}/valid.jsonl"
bpe_corpus="${data_dir}/bpe_corpus.txt"
bpe_tokenizer_prefix="${data_dir}/bpe_tokenizer"


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


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  $verbose && echo "stage 1: prepare data"
  cd "${work_dir}" || exit 1;

  python3 step_1_prepare_data.py \
  --train_subset "${train_subset}" \
  --valid_subset "${valid_subset}" \

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  $verbose && echo "stage 2: make bpe corpus"
  cd "${work_dir}" || exit 1;

  python3 step_2_make_corpus.py \
  --corpus_file "${train_subset}" \
  --output_file "${bpe_corpus}" \

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  $verbose && echo "stage 3: train bpe"
  cd "${work_dir}" || exit 1;

  python3 step_3_train_bpe.py \
  --corpus_file "${bpe_corpus}" \
  --model_prefix "${bpe_tokenizer_prefix}" \

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  $verbose && echo "stage 4: make vocabulary"
  cd "${work_dir}" || exit 1;

  python3 step_4_make_vocabulary.py \
  --bpe_tokenizer_file "${bpe_tokenizer_prefix}.model" \
  --train_subset "${train_subset}" \
  --valid_subset "${valid_subset}" \
  --vocabulary_dir "${vocabulary_dir}" \

fi



if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  $verbose && echo "stage 5: train model"
  cd "${work_dir}" || exit 1;

  python3 step_5_train_model.py \
  --bpe_tokenizer_file "${bpe_tokenizer_prefix}.model" \
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
  cp "config.json" "${trained_models_dir}/${trained_model_name}/config.json"

fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  $verbose && echo "stage 7: predict by archive"
  cd "${work_dir}" || exit 1;

  python3 step_6_predict_by_archive.py \
  --archive_file "${trained_models_dir}/${trained_model_name}" \
  --pretrained_model_name_or_path "${pretrained_models_dir}/${pretrained_model_name}" \

fi
