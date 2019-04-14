app_name=fashion_killer
cov_dir=htmlcov
pytest_args:= -v -s --color=yes --junit-xml ut_results.xml --cov-report=html:$(cov_dir) --cov-report=term --cov $(app_name)

.PHONY: check test unittest functest

check:
	pyflakes . unittests functests
	pycodestyle . unittests functests
	pylint -j2 -E . unittests functests

test: unittest functest

unittest:
	pytest -v -s unittests

functest:
	pytest -v -s functests
