install:
	pip3 install setuptools wheel twine

build:
	python3 setup.py sdist bdist_wheel

clean:
	$(RM) -rf build dist mcy_dist_ai.egg-info

upload-test:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

upload:
	twine upload dist/*
