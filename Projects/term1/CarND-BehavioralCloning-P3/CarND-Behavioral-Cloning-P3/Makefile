all:
	$(MAKE) train.x

train.x:
	python model.py

loadmodel.x:
	python drive.py model.h5

loadmodelNsave.x:
	python drive.py model.h5 $(rdir)

simulate.x:
	../linux_sim/run

makevideo.x:
	python video.py results --fps 60
