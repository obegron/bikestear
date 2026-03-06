SHELL := /bin/bash

PROFILE ?= supertuxkart
CAMERA ?= auto
BIKE ?= sim
HZ ?= 60
MONITOR_ARGS ?=
CALIBRATE_ARGS ?=
DEBUG_ARGS ?=
VISION_ARGS ?=

.PHONY: sync lock list-cameras list-bikes monitor run calibrate test

sync:
	uv sync

lock:
	uv lock

list-cameras:
	uv run ftms2pad list-cameras

list-bikes:
	uv run ftms2pad list-bikes

monitor:
	uv run ftms2pad monitor --profile $(PROFILE) --bike $(BIKE) --camera $(CAMERA) --hz $(HZ) $(VISION_ARGS) $(MONITOR_ARGS) $(DEBUG_ARGS)

run:
	uv run ftms2pad run --profile $(PROFILE) --bike $(BIKE) --camera $(CAMERA) --hz $(HZ) $(VISION_ARGS) $(DEBUG_ARGS)

calibrate:
	uv run ftms2pad calibrate --profile $(PROFILE) --camera $(CAMERA) --bike $(BIKE) $(VISION_ARGS) $(CALIBRATE_ARGS) $(DEBUG_ARGS)

test:
	uv run python -m unittest -v tests/test_ftms_parser.py
