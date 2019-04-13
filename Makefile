.PHONY: unittest

unittest:
	pytest -v -s unittests

functest:
	pytest -v -s functests
