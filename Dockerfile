FROM python
RUN pip install http://h2o-release.s3.amazonaws.com/h2o/rel-zorn/1/Python/h2o-3.36.0.1-py2.py3-none-any.whl
RUN pip install numpy