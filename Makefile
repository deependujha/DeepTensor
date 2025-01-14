.PHONY: all docs setup test clean ntidy_commit

all:
	@echo "Run 'make setup' to setup the project"
	@echo "Run 'make test' to run the tests"
	@echo "Run 'make clean' to clean the run-generated files"

docs:
	@echo "TO BE UPDATED"

setup:
	pip install -e .

test:
	pytest tests

ctest:
	@mkdir -p build \
		&& cd build \
		&& cmake -DBUILD_TESTS=ON -DBUILD_PYBIND=ON .. \
		&& cmake --build . \
		&& ctest

pybind_intellisense:
	@mkdir -p build \
		&& cd build \
		&& cmake -DBUILD_PYBIND=ON .. \
		&& cmake --build .

# delete all gitignored files
clean:
	git clean -fdX

# not tidy (ignore clang-tidy error)
ntidy_commit:
	git add .
	IGNORE_CLANG_TIDY_ERROR=1 git commit
