VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

all: help

# Create virtual environment and install dependencies

install:
	python3 -m venv $(VENV)
	. $(VENV)/bin/activate && \
	$(PIP) install --upgrade pip && \
	$(PIP) install -r requirements.txt

split:
	. $(VENV)/bin/activate && \
	$(PYTHON) src/split_dataset.py --dataset_path data/data.csv

train:
	. $(VENV)/bin/activate && \
	$(PYTHON) src/train.py --layers 24 24 24 --epochs 84  --batch_size 8 --learning_rate 0.0314

# predict:
# 	. $(VENV)/bin/activate && \
# 	$(PYTHON) predict.py --model saved_model.npy --data data_validation.csv

# clean:
# 	rm -rf src/__pycache__ *.pyc saved_model.npy *.log $(VENV)

help:
	@echo "Available targets:"
	@echo "  install     Create virtualenv and install dependencies"
	@echo "  split       Split the dataset"
	@echo "  train       Train the neural network"
	@echo "  predict     Run predictions"
	@echo "  clean       Clean build artifacts and environment"
