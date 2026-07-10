SHELL := /bin/bash

PROFILE ?= supertuxkart
CAMERA ?= 0
BIKE ?= sim
HZ ?= 60
MONITOR_ARGS ?=
CALIBRATE_ARGS ?=

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
	uv run ftms2pad monitor --profile $(PROFILE) --bike $(BIKE) --camera $(CAMERA) --hz $(HZ) $(MONITOR_ARGS)

run:
	uv run ftms2pad run --profile $(PROFILE) --bike $(BIKE) --camera $(CAMERA) --hz $(HZ)

calibrate:
	uv run ftms2pad calibrate --profile $(PROFILE) --camera $(CAMERA) $(CALIBRATE_ARGS)

test:
	uv run python -m unittest discover -s tests -v
