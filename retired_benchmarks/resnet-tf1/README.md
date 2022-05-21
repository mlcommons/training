Install
==========

In order to run this, you must first set stuff up... for now see Transformer's README.


Downlaoding Data
==========

Downloading data is TBD.


Processing Data
=============

TBD.


Running the Benchmark
============

You first must build the docker file;

    docker build .


Remember the image name/number.


1. Make sure /imn on the host contains the pre-processed data. (Scripts for this TODO).
2. Choose your random seed (below we use 77)
3. Enter your docker's image name (below we use 5ca81979cbc2 which you don't have)

Then, executute the following:

    sudo docker run -v /imn:/imn --runtime=nvidia -t -i 5ca81979cbc2 "./run_and_time.sh" 77 | tee benchmark.log


