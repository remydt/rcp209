.PHONY: build run

build:
	docker build --tag rcp209 .

run:
	docker run --interactive --rm --tty --volume="$(PWD)/src/:/usr/src/" rcp209
