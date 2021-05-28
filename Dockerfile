FROM continuumio/miniconda3

ADD environment.yml /src/environment.yml
RUN conda env create -f /src/environment.yml
ENV PATH /opt/conda/envs/search-lab/bin:$PATH

ADD entrypoint.sh /src
ADD main.py /src
ADD gunicorn_conf.py /src
ADD app /src/app
ADD static/ /src/static
ADD config/ /src/config

RUN /bin/bash -c "source activate search-lab"

WORKDIR /src

EXPOSE 80

ENTRYPOINT ["./entrypoint.sh"]
