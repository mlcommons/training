
=== TODO (brief notes) ===

Approximate Bleu score after 1 epoch: 5


Where e15846d665b2 is your docker image, and 77 is your random seed.
To run:
    sudo docker run -v /home/vbittorf/reference/transformer/raw_data:/raw_data --runtime=nvidia -t -i e15846d665b2 "./run_and_time.sh" 77 | tee benchmark.log


=== Install Setup ===

DO NOT COPY AND PASTE ALL AT ONCE.
Be careful ...


	# Install docker
	sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
	curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
	sudo apt-key fingerprint 0EBFCD88
	sudo add-apt-repository    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
	   $(lsb_release -cs) \
	   stable"
	sudo apt update
	sudo apt install docker-ce -y

	# Install nvidia-docker2
	curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey |   sudo apt-key add -
	distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list |   sudo tee /etc/apt/sources.list.d/nvidia-docker.list
	sudo apt-get update
	sudo apt install nvidia-docker2 -y


	sudo tee /etc/docker/daemon.json <<EOF
	{
	    "runtimes": {
		"nvidia": {
		    "path": "/usr/bin/nvidia-container-runtime",
		    "runtimeArgs": []
		}
	    }
	}
	EOF
	sudo pkill -SIGHUP dockerd

	sudo mount /dev/sdc /imn/

	sudo apt install bridge-utils
	sudo su
	service docker stop
	sleep 1;
	iptables -t nat -F
	sleep 1;
	ifconfig docker0 down
	sleep 1;
	brctl delbr docker0
	sleep 1;
	service docker start
	exit;

	sudo docker build .
