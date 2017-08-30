env_dir = ~/Udacity/CarND/envs
env_name = carnd-term1
script_path = ./scripts

all : 
	$(make) createEnv.x

createEnv.x : 
	./scripts/createEnv.sh $(env_dir) $(env_name)

activateEnv.x : 
	bash $(script_path)/activateEnv.sh $(env_dir) $(env_name)


