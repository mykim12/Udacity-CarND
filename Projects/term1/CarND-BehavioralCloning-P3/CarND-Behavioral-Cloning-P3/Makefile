all:
	$(MAKE) run.model.x

train.x:
	python model.py

run.model.x:
	python drive.py model.h5

test.model.x:
	python drive.py $(m)

loadmodelNsave.x:
	python drive.py model.h5 $(rdir)

simulator.x:
	../linux_sim/run

makevideo.x:
	python video.py results --fps 60
