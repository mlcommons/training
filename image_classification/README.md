TODO


Running:

Note: we assume that imagenet pre-processed has already been mounted at `/imn` ... In the future, we will have data download and pre-processing scripts. 


    IMAGE=`sudo docker build . | tail -n 1 | awk '{print $3}'`
    SEED=2
    NOW=`date "+%F-%T"`
    sudo docker run -v /imn:/imn --runtime=nvidia -t -i $IMAGE "./run_and_time.sh" $SEED | tee benchmark-$NOW.log
