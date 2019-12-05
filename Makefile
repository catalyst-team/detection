.PHONY: docker clean

docker: ./requirements.txt
	docker build -t catalyst-detection:latest . -f ./Dockerfile --no-cache

clean:
	rm -rf build/
	docker rmi -f catalyst-detection:latest
