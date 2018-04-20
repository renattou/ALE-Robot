help:
	@cat Makefile

GPU?=0
DOCKER_FILE=Dockerfile
DOCKER=GPU=$(GPU) nvidia-docker
IMAGE=renato/atari-robot
SRC?=$(shell dirname `pwd`)
DST=/src/workspace
PARAMS=--csv_file ./results/pong.csv --save_weights_prefix pong

build:
	docker build -t $(IMAGE) -f $(DOCKER_FILE) .

bash: build
	$(DOCKER) run --rm -it -v $(SRC):$(DST) $(IMAGE) bash

train: build
	$(DOCKER) run --rm -it -v $(SRC):$(DST) -w $(DST) $(IMAGE) $(DST)/train.sh pong /src/vrep $(PARAMS)

play: build
	$(DOCKER) run --rm -it -v $(SRC):$(DST) -w $(DST) $(IMAGE) $(DST)/play.sh pong /src/vrep $(PARAMS)

train_ale: build
	$(DOCKER) run --rm -it -v $(SRC):$(DST) -w $(DST) $(IMAGE) $(DST)/train_ale.sh pong $(PARAMS)

play_ale: build
	$(DOCKER) run --rm -it -v $(SRC):$(DST) -w $(DST) $(IMAGE) $(DST)/play_ale.sh pong $(PARAMS)
