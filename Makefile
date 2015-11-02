all:
	g++ `pkg-config --cflags opencv` -std=c++11 -Wall *.cpp -I /usr/include/opencv2 `pkg-config opencv --libs` -o exe

run: 
	./exe