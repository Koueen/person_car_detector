echo $(pwd)
container_id=$(docker run --rm -it -p 127.0.0.1:8080:8080 -p 127.0.0.1:8081:8081 \
-v $(pwd)/app/model_store:/home/model-server/model-store \
-v $(pwd)/app/torchserve_logs:/home/model-server/logs \
pytorch/torchserve:latest-cpu \
torchserve --start --ncs --model-store /home/model-server/model-store --models fcos_model=fcos_model.mar)

docker wait $container_id
rm app/torchserve_logs/*