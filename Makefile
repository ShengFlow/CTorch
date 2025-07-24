Tensor: include/dev/Tensor.cppm src/dev/Tensor.cpp
	clang++ -std=c++20 --precompile include/dev/Tensor.cppm -o Tensor.pcm
	clang++ -std=c++20 -c src/dev/Tensor.cpp -fmodule-file=Tensor="Tensor.pcm" -o Tensor.o

test: Tensor.o Tensor.pcm
	clang++ -std=c++20 -c src/dev/test.cpp -fmodule-file=Tensor="Tensor.pcm" -o test.o
	clang++ Tensor.o test.o -o test.exe

clean:
	del *.pcm
	del *.o