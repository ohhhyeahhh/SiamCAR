all:
    # install pycocotools locally
	python3 setup.py build_ext --inplace
	rm -rf build

install:
	# install pycocotools to the Python site-packages
	python3 setup.py build_ext install
	rm -rf build
clean:
	rm _mask.c _mask.cpython-36m-x86_64-linux-gnu.so
