# PANGEA: A FULLY OPEN MULTILINGUAL MULTIMODAL LLM FOR 39 LANGUAGES

[Homepage](https://neulab.github.io/Pangea/) | [Pangea-7B](https://huggingface.co/neulab/Pangea-7B) | [PangeaIns](https://huggingface.co/datasets/neulab/PangeaInstruct) 
| [PangeaBench](https://huggingface.co/collections/neulab/pangea-6713c3b0d78a453906eb2ed8) | [Github](https://github.com/neulab/Pangea/tree/main)
| [Arxiv](https://arxiv.org/abs/2410.16153) | [PDF](https://arxiv.org/pdf/2410.16153) | [Demo](https://huggingface.co/spaces/neulab/Pangea)

This repository provides the necessary resources and guidelines for setting up, training, and evaluating PANGEA.

## About PANGEA

PANGEA is an open-source multilingual multimodal large language model designed to understand and generate content across diverse languages and cultural contexts.
PANGEA aims to bridge the gap in multilingual and multimodal language processing by providing an MLLM trained on the PANGEAINS dataset, a diverse corpus of 6 million multimodal instructions spanning 39 languages. The training data, PANGEAINS, includes high-quality English instructions, machine-translated instructions, and culturally relevant multimodal tasks to ensure comprehensive cross-cultural coverage. PANGEA is evaluated using PANGEABench, a comprehensive evaluation benchmark covering 47 languages through 14 datasets.

## Repository Structure

The repository is organized into the following directories:

- **train**: Contains scripts and instructions for pretraining and finetuning the PANGEA model. We have made modifications from the open-source [Llava-Next](https://github.com/LLaVA-VL/LLaVA-NeXT) repository.

- **evaluation**: Includes code and datasets to assess the model's performance across various tasks and languages. The code is modified from the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) repository for evaluation.

- **data**: Provides examples of the finetuning data used for PANGEA, facilitating understanding of the data format and structure.

- **predict**: Example Python code usage of Pangea-7B.

## Setting Up

To get started with PANGEA:

1. **Clone the Repository**: Use Git to clone the repository to your local environment.

2. **Install Dependencies**: Ensure you have the required dependencies installed. For training, you need to do 

```bash
cd train/LLaVA-NeXT
pip install -e ".[train]"
```

For evaluation, you need to do

```bash
cd evaluation/lmms-eval
pip install -e .
```

3. **Download Datasets**: Acquire the necessary pretraining and fine-tuning datasets. For pretraining, download the LLaVA-Pretrain dataset from [HuggingFace](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain). For finetuning, download the PangeaIns dataset from [HuggingFace](https://huggingface.co/datasets/neulab/PangeaInstruct).

## Quick Start
After installing the required packages in `train/LLaVA-NeXT`, you could go to `predict` and run example Python code using Pangea-7B.

```bash
cd predict
python predict_all.py # You could evaluate both multimodal inputs and text-only inputs with this script
python predict_multimodal.py # You could evaluate multimodal inputs with this script but not text-only inputs
python predict_text_only.py # You could evaluate text-only inputs with this script but not multimodal inputs
```

## Sample Data and Format

The `data/sample_data.json` file contains samples of the finetuning data used for training PANGEA. The `data/images` folder contains images referred to in the data sample.

Here is an example of one such data instance:

```json
{
   "id": 2433684022797.0,
   "image": "images/cultural/2433684022797.0.jpg",
   "conversations": [
      {
         "from": "human",
         "value": "<image>\n사진 속의 단풍잎들로 어떤 예술 작품을 만들 수 있을까? 예술 작품의 구체적인 아이디어와 그 이유를 상세히 설명해보세요."
      },
      {
         "from": "gpt",
         "value": "사진 속의 붉고 화려한 단풍잎들을 이용하여 '모자이크 예술 작품'을 만들어 보는 것은 어떨까요? 먼저, 단풍잎들을 작은 조각으로 잘라 내어 다양한 색상과 질감을 갖도록 준비합니다. 그런 다음, 이 조각들을 캔버스 위에 붙여 모자이크 패턴을 형성합니다. 예를 들어, 단풍잎 조각들을 이용하여 나뭇잎 모양의 패턴이나 추상적인 예술 작품을 만들 수 있습니다. 이러한 작업을 통해 자연의 아름다움과 계절의 순환을 상징하는 단풍을 예술 작품으로 재해석할 수 있습니다. 모자이크 작업은 단순한 예술 작품을 넘어, 환경 친화적이고 지속 가능한 예술 작업을 촉진하는 데에도 기여할 수 있습니다."
      },
      {
         "from": "human",
         "value": "사진 속의 단풍잎들을 이용하여 어떤 행사를 기획한다면, 어떤 행사를 기획하고 어떻게 준비해야 할지 구체적으로 설명해보세요."
      },
      {
         "from": "gpt",
         "value": "사진 속의 단풍잎들을 이용하여 '가을 단풍 축제'를 기획해 보는 것은 어떨까요? 먼저, 축제 장소를 한국의 아름다운 자연 경관이 있는 공원이나 산에서 열도록 준비합니다. 단풍잎들을 이용하여 장식을 만들고, 가을 분위기를 연출하는 데 사용할 수 있습니다. 또한, 단풍놀이, 산책, 사진 촬영 대회 등 다양한 이벤트를 포함하여 참가자들이 자연을 즐기고 가을의 아름다움을 느낄 수 있도록 합니다. 음식 및 음료 부스도 설치하여 현지 특산물이나 가을 음식을 제공합니다. 마케팅을 위해 소셜 미디어를 활용하여 축제를 홍보하고, 지역 공동체와의 협력을 통해 행사의 성공을 보장합니다. 이러한 준비를 통해 단풍이 가져다주는 자연의 아름다움과 문화적 의미를 참가자들과 함께 나눌 수 있는 기회를 제공합니다."
      }
   ],
   "language": "ko"
}
```
![ex](data/images/cultural/2433684022797.0.jpg)

The corresponding image file for this example is located at `data/images/cultural/2433684022797.0.jpg`.

### Data Structure:
- **id**: Unique identifier for the data sample.
- **image**: The path to the image file used in this instance.
- **conversations**: A series of conversations between the "human" and the model (in this case, referred to as "gpt").
   - **from**: Identifies the speaker (either "human" or "gpt").
   - **value**: The content of the message, which can include both text and image references.
- **language**: The language of the instruction and conversation (in this example, it is Korean).

## Training

### Stage 1: Pretraining

After setting up, initiate the pretraining phase:

1. **Run the Pretraining Script**:

```bash
cd pangea/train
./LLaVA-NeXT/scripts/train/pretrain_pangea.sh
```
This result in the creation of a `mm_projector.bin` file essential for the finetuning stage.

### Stage 2: Finetuning

Once pretraining is complete, proceed to finetune the model:

1. **Ensure Fine-tuning Data is Available**: Obtain the fine-tuning data and place it in the designated directory, as specified in the `finetune_pangea.sh` script. The training data will be publicly available on huggingface once we publish the paper.

2. **Run the Fine-tuning Script**:

```bash
cd pangea/train
./LLaVA-NeXT/scripts/train/finetune_pangea.sh
```

## Evaluation

To evaluate the model's capabilities:

1. **Navigate to the Evaluation Directory**:

```bash
cd pangea/evaluation
```

2. **Run the Evaluation Script**:

```bash
model=Pangea
model_type=llava
python3 -m accelerate.commands.launch \
         --num_processes=8 \
         -m lmms_eval \
         --model $model_type \
         --model_args pretrained=$model,conv_template=qwen_1_5 \
         --tasks ${task} \
         --batch_size 1 \
         --log_samples \
         --log_samples_suffix ${task} \
         --output_path eval_logs
```

If you would like to evaluate other models on PangeaBench, Replace `${model}` with the path to your model, `${model_type}` with you model type, and `${task}` with the specific evaluation task you wish to perform. Note that we use `conv_template=qwen_1_5` for Pangea, you could change this when using other models.

For detailed instructions and examples, refer to the `script.sh` file in the evaluation directory.

## Citation
```
@article{yue2024pangeafullyopenmultilingual,
  title={Pangea: A Fully Open Multilingual Multimodal LLM for 39 Languages},
  author={Xiang Yue and Yueqi Song and Akari Asai and Seungone Kim and Jean de Dieu Nyandwi and Simran Khanuja and Anjali Kantharuban and Lintang Sutawika and Sathyanarayanan Ramamoorthy and Graham Neubig},
  year={2024},
  journal={arXiv preprint arXiv:2410.16153},
  url={https://arxiv.org/abs/2410.16153}
}
```
