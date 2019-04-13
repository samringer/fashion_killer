.PHONY: test unittest functest

test: unittest functest

unittest:
	pytest -v -s unittests

functest:
	pytest -v -s functests
