#!/usr/bin/env bash

# nohup sh run.sh --system_version centos --stage 0 --stop_stage 5 &

# sh run.sh --system_version windows --stage -2 --stop_stage 8
# sh run.sh --system_version windows --stage 0 --stop_stage 0
# sh run.sh --system_version windows --stage 1 --stop_stage 1
# sh run.sh --system_version windows --stage 2 --stop_stage 2
# sh run.sh --system_version windows --stage 0 --stop_stage 5
# sh run.sh --system_version windows --stage 6 --stop_stage 6

# params
system_version="centos";
verbose=true;
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5


#trained_model_name=telemarketing_intent_classification_cn
#pretrained_bert_model_name=chinese-bert-wwm-ext
#dataset_fn="telemarketing_intent_cn.xlsx"

#trained_model_name=telemarketing_intent_classification_en
#pretrained_bert_model_name=bert-base-uncased
#dataset_fn="telemarketing_intent_en.xlsx"

#trained_model_name=telemarketing_intent_classification_jp
#pretrained_bert_model_name=bert-base-japanese
#dataset_fn="telemarketing_intent_jp.xlsx"

trained_model_name=telemarketing_intent_classification_vi
pretrained_bert_model_name=bert-base-vietnamese-uncased
dataset_fn="telemarketing_intent_vi.xlsx"

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

serialization_dir1="${data_dir}/serialization_dir1";
serialization_dir2="${data_dir}/serialization_dir2";

mkdir -p "${data_dir}"
mkdir -p "${pretrained_models_dir}"
mkdir -p "${trained_models_dir}"
mkdir -p "${serialization_dir1}"
mkdir -p "${serialization_dir2}"

vocabulary_dir="${data_dir}/vocabulary"
train_subset="${data_dir}/train.json"
valid_subset="${data_dir}/valid.json"
hierarchical_labels_pkl="${data_dir}/hierarchical_labels.pkl"
dataset_filename="${data_dir}/${dataset_fn}"

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

  wget "https://huggingface.co/datasets/qgyd2021/telemarketing_intent/resolve/main/${dataset_fn}"

fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  $verbose && echo "stage 0: prepare data without irrelevant domain (make train subset, valid subset file)"
  cd "${work_dir}" || exit 1;

  python3 1.prepare_data.py \
  --without_irrelevant_domain \
  --dataset_filename "${dataset_filename}" \
  --do_lowercase \
  --train_subset "${train_subset}" \
  --valid_subset "${valid_subset}" \

fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  $verbose && echo "stage 1: make hierarchical labels dictionary (make hierarchical_labels.pkl file)"
  cd "${work_dir}" || exit 1
  python3 2.make_hierarchical_labels.py \
  --dataset_filename "${dataset_filename}" \
  --hierarchical_labels_pkl "${hierarchical_labels_pkl}" \

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  $verbose && echo "stage 2: make vocabulary (make vocabulary directory)"
  cd "${work_dir}" || exit 1
  python3 3.make_vocabulary.py \
  --pretrained_model_path "${pretrained_model_dir}" \
  --hierarchical_labels_pkl "${hierarchical_labels_pkl}" \
  --vocabulary "${vocabulary_dir}" \

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  $verbose && echo "stage 3: train model without irrelevant domain"
  cd "${work_dir}" || exit 1

  python3 4.train_model.py \
  --pretrained_model_path "${pretrained_model_dir}" \
  --hierarchical_labels_pkl "${hierarchical_labels_pkl}" \
  --vocabulary_dir "${vocabulary_dir}" \
  --train_subset "${train_subset}" \
  --valid_subset "${valid_subset}" \
  --serialization_dir "${serialization_dir1}" \

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  $verbose && echo "stage 4: prepare data with irrelevant domain"
  cd "${work_dir}" || exit 1

  python3 1.prepare_data.py \
  --dataset_filename "${dataset_filename}" \
  --do_lowercase \
  --train_subset "${train_subset}" \
  --valid_subset "${valid_subset}" \

fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  $verbose && echo "stage 5: train model with irrelevant domain"
  cd "${work_dir}" || exit 1

  python3 4.train_model.py \
  --pretrained_model_path "${pretrained_model_dir}" \
  --hierarchical_labels_pkl "${hierarchical_labels_pkl}" \
  --vocabulary_dir "${vocabulary_dir}" \
  --train_subset "${train_subset}" \
  --valid_subset "${valid_subset}" \
  --serialization_dir "${serialization_dir2}" \
  --checkpoint_path "${serialization_dir1}/best.th"

fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  $verbose && echo "stage 6: make json config"
  cd "${work_dir}" || exit 1
  python3 6.make_json_config.py \
  --pretrained_model_path "${pretrained_model_dir}" \
  --hierarchical_labels_pkl "${hierarchical_labels_pkl}" \
  --vocabulary_dir "${vocabulary_dir}" \
  --train_subset "${train_subset}" \
  --valid_subset "${valid_subset}" \
  --serialization_dir "${serialization_dir2}" \
  --json_config_dir "${data_dir}" \

fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  $verbose && echo "stage 7: collect files"
  cd "${work_dir}" || exit 1;

  mkdir -p "${trained_models_dir}/${trained_model_name}"

  cp -r "${vocabulary_dir}" "${trained_models_dir}/${trained_model_name}/vocabulary/"
  cp "${serialization_dir2}/best.th" "${trained_models_dir}/${trained_model_name}/weights.th"
  cp "${data_dir}/config.json" "${trained_models_dir}/${trained_model_name}/config.json"
  cp "${hierarchical_labels_pkl}" "${trained_models_dir}/${trained_model_name}"

fi


if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  $verbose && echo "stage 8: predict by archive"
  cd "${work_dir}" || exit 1;

  python3 7.predict_by_archive.py \
  --archive_file "${trained_models_dir}/${trained_model_name}" \
  --pretrained_model_path "${pretrained_model_dir}" \

fi
